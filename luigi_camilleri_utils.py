import torch
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import json
from PIL import Image, ImageOps
import numpy as np
from pycocotools.cocoeval import COCOeval
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet


def get_device():
    """Use CUDA if available else CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_efficientdet(num_classes, model_name='tf_efficientdet_d2'):
    """
    Get an EfficientDet model configured for the specified number of classes.
    
    Args:
        num_classes: Number of classes (including background)
        model_name: EfficientDet variant to use (e.g., 'tf_efficientdet_d0', 'tf_efficientdet_d2')
    
    Returns:
        DetBenchTrain model for training
    """
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (512, 512)
    
    model = EfficientDet(config, pretrained_backbone=True)
    model.class_net = HeadNet(config, num_outputs=num_classes)
    
    return model, config


def get_efficientdet_train(num_classes, model_name='tf_efficientdet_d2'):
    """Get EfficientDet wrapped for training"""
    model, config = get_efficientdet(num_classes, model_name)
    return DetBenchTrain(model, config), config


def get_efficientdet_predict(model, config):
    """Wrap trained model for inference"""
    return DetBenchPredict(model.model, config)


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: DetBenchTrain model
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to use
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        Dictionary with average losses
    """
    from torch.amp import autocast
    
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    for images, targets in loader:
        images = torch.stack([img.to(device) for img in images])
        
        # Prepare targets for EfficientDet format
        # EfficientDet expects boxes in [x1, y1, x2, y2] format and batch dimension
        batch_size = images.shape[0]
        
        # Find max number of boxes in batch for padding
        max_boxes = max(len(t['boxes']) for t in targets)
        if max_boxes == 0:
            max_boxes = 1
        
        # Create padded tensors
        boxes_batch = torch.zeros(batch_size, max_boxes, 4, device=device)
        labels_batch = torch.zeros(batch_size, max_boxes, dtype=torch.float32, device=device)
        
        for i, t in enumerate(targets):
            num_boxes = len(t['boxes'])
            if num_boxes > 0:
                boxes_batch[i, :num_boxes] = t['boxes'].to(device)
                labels_batch[i, :num_boxes] = t['labels'].float().to(device)
        
        target_dict = {
            'bbox': boxes_batch,
            'cls': labels_batch,
        }

        optimizer.zero_grad()
        
        with autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, target_dict)
            loss = loss_dict['loss']
            cls_loss = loss_dict.get('class_loss', torch.tensor(0.0))
            box_loss = loss_dict.get('box_loss', torch.tensor(0.0))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if isinstance(cls_loss, torch.Tensor):
            total_cls_loss += cls_loss.item()
        if isinstance(box_loss, torch.Tensor):
            total_box_loss += box_loss.item()

    num_batches = len(loader)
    return {
        "total": total_loss / num_batches,
        "cls": total_cls_loss / num_batches,
        "box": total_box_loss / num_batches,
    }


def evaluate_map(model, data_loader, device, coco_gt, config):
    """
    Evaluate mAP using COCO evaluation.
    
    Args:
        model: Trained model (DetBenchTrain)
        data_loader: DataLoader for validation data
        device: Device
        coco_gt: COCO ground truth object
        config: EfficientDet config
    
    Returns:
        COCO evaluation stats
    """
    # Wrap for inference
    eval_model = DetBenchPredict(model.model, config)
    eval_model.to(device)
    eval_model.eval()
    
    coco_results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = torch.stack([img.to(device) for img in images])
            outputs = eval_model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                
                # EfficientDet outputs: [batch, num_detections, 6]
                # Format: [x1, y1, x2, y2, score, class]
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                
                for det in output:
                    if det[4] > 0.01:  # Filter very low scores
                        x1, y1, x2, y2, score, cls = det.cpu().numpy()
                        coco_results.append({
                            "image_id": image_id,
                            "category_id": int(cls),
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                        })

    if len(coco_results) == 0:
        print("No detections found!")
        return [0.0] * 12
    
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats


def f1_score_by_iou(model, loader, device, config, iou_threshold=0.5, score_threshold=0.1):
    """
    Calculate F1 score based on IoU matching.
    
    Args:
        model: Trained model (DetBenchTrain)
        loader: DataLoader
        device: Device
        config: EfficientDet config
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for filtering predictions
    
    Returns:
        F1 score
    """
    # Wrap for inference
    eval_model = DetBenchPredict(model.model, config)
    eval_model.to(device)
    eval_model.eval()
    
    tp = fp = fn = 0

    with torch.no_grad():
        for images, targets in loader:
            images = torch.stack([img.to(device) for img in images])
            outputs = eval_model(images)

            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)
                
                # Parse EfficientDet output: [num_detections, 6]
                # Format: [x1, y1, x2, y2, score, class]
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                
                # Filter by score
                keep = out[:, 4] > score_threshold
                pred_boxes = out[keep, :4]
                pred_labels = out[keep, 5].long()

                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue

                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched = set()

                for i in range(len(pred_boxes)):
                    max_iou, idx = ious[i].max(0)
                    if (
                        max_iou >= iou_threshold
                        and idx.item() not in matched
                        and pred_labels[i] == gt_labels[idx]
                    ):
                        tp += 1
                        matched.add(idx.item())
                    else:
                        fp += 1

                fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return 2 * (precision * recall) / (precision + recall + 1e-6)


def visualize_predictions(img, prediction, class_names, threshold=0.1):
    """
    Visualize predictions on an image.
    
    Args:
        img: Image tensor (C, H, W)
        prediction: EfficientDet prediction tensor [num_dets, 6]
        class_names: Dictionary mapping class IDs to names
        threshold: Score threshold for visualization
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img.permute(1, 2, 0).cpu())
    
    if prediction.dim() == 1:
        prediction = prediction.unsqueeze(0)
    
    for det in prediction:
        x1, y1, x2, y2, score, label = det.cpu().numpy()
        if score > threshold:
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2
                )
            )
            class_name = class_names.get(int(label), "Unknown")
            plt.text(
                x1, y1 - 5,
                f"{class_name} {score:.2f}",
                color="red",
                fontsize=10,
                backgroundcolor="white"
            )
    plt.axis("off")
    plt.show()


class SignsDataset(Dataset):
    """Dataset class for traffic sign detection with COCO-format annotations."""
    
    def __init__(self, root, annFile, transforms, preload=True):
        self.root = root
        self.transforms = transforms

        with open(annFile) as f:
            data = json.load(f)

        self.preload = preload
        self.loaded_images = []
        self.images_info = []

        for img_info in data["images"]:
            img_name = os.path.basename(img_info["file_name"])
            img_path = os.path.join(root, img_name)

            if not os.path.exists(img_path):
                print(f"Image not found at {img_path}")
                continue

            self.images_info.append(img_info)

            if preload:
                with Image.open(img_path) as img:
                    self.loaded_images.append(img.convert("RGB").copy())
        
        self.imgToAnns = {img["id"]: [] for img in self.images_info}
        for ann in data["annotations"]:
            if ann["image_id"] in self.imgToAnns:
                self.imgToAnns[ann["image_id"]].append(ann)

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        image_id = img_info["id"]

        if self.preload:
            img = self.loaded_images[idx].copy()
        else:
            img_path = os.path.join(self.root, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)

        anns = self.imgToAnns[image_id]
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, w, h])
            labels.append(ann["category_id"])

        augmented = self.transforms(
            image=np.array(img),
            bboxes=boxes,
            labels=labels
        )

        img = augmented["image"]
        boxes = augmented["bboxes"]
        labels = augmented["labels"]

        # Handle empty boxes case
        if len(boxes) > 0:
            boxes = torch.tensor(
                [[x, y, x + w, y + h] for x, y, w, h in boxes],
                dtype=torch.float32
            )
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        return img, target

    def __len__(self):
        return len(self.images_info)
