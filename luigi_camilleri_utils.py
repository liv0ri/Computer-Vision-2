import torch
from torch.amp import autocast
from torch.utils.data import Dataset
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import os
import json
from PIL import Image, ImageOps
import numpy as np
from pycocotools.cocoeval import COCOeval
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet


def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_efficientdet_train(num_classes):
    """
    Create an EfficientDet model wrapped for training.
    
    Args:
        num_classes: Number of classes (including background if required)
    
    Returns:
        Tuple of (model, config) where model is DetBenchTrain wrapper
    """
    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = num_classes
    # Don't override image_size - use default from config (768 for D2)
    # The albumentations transforms handle resizing to match
    
    # Create base model
    net = EfficientDet(config, pretrained_backbone=True)
    
    # Reset classification head for new number of classes
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    
    # Wrap in training benchmark
    model = DetBenchTrain(net, config)
    
    return model, config


def get_efficientdet_predict(model, config):
    """
    Create an EfficientDet model wrapped for prediction/inference.
    
    Args:
        model: Trained DetBenchTrain model
        config: Model configuration
    
    Returns:
        DetBenchPredict wrapper for inference
    """
    return DetBenchPredict(model.model, config)


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: EfficientDet model (DetBenchTrain)
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to use
        scaler: Optional GradScaler for mixed precision
    
    Returns:
        Dictionary with 'total', 'cls', and 'box' losses
    """
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    for images, targets in loader:
        # Stack images into batch tensor
        images = torch.stack([img.to(device) for img in images])
        
        # Prepare targets for EfficientDet
        boxes = [t["boxes"].to(device) for t in targets]
        labels = [t["labels"].to(device) for t in targets]
        
        # Create target dict for EfficientDet
        target_dict = {
            "bbox": boxes,
            "cls": labels,
        }

        optimizer.zero_grad()
        
        with autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, target_dict)
            
            # EfficientDet returns loss dict with 'loss', 'class_loss', 'box_loss'
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
        total_cls_loss += cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss
        total_box_loss += box_loss.item() if isinstance(box_loss, torch.Tensor) else box_loss

    num_batches = len(loader)

    return {
        "total": total_loss / num_batches,
        "cls": total_cls_loss / num_batches,
        "box": total_box_loss / num_batches,
    }


def evaluate_map(model, data_loader, device, coco_gt, config):
    """
    Evaluate mAP using COCO evaluation metrics.
    
    Args:
        model: Trained model (DetBenchTrain)
        data_loader: DataLoader for validation data
        device: Device to use
        coco_gt: COCO ground truth object
        config: Model configuration
    
    Returns:
        COCO evaluation stats
    """
    # Create prediction wrapper
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
                
                # EfficientDet output: [num_dets, 6] -> [x1, y1, x2, y2, score, class]
                boxes = output[:, :4].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                labels = output[:, 5].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.01:  # Filter low confidence
                        coco_results.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1]),
                            ],
                            "score": float(score),
                        })

    if len(coco_results) == 0:
        print("No predictions to evaluate")
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
        device: Device to use
        config: Model configuration
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for filtering predictions
    
    Returns:
        F1 score
    """
    # Create prediction wrapper
    eval_model = DetBenchPredict(model.model)
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

                # EfficientDet output: [num_dets, 6] -> [x1, y1, x2, y2, score, class]
                scores = out[:, 4]
                keep = scores > score_threshold
                
                pred_boxes = out[keep][:, :4]
                pred_labels = out[keep][:, 5].long()

                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
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
        threshold: Score threshold for displaying predictions
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(img.permute(1, 2, 0).cpu())
    
    # EfficientDet output: [num_dets, 6] -> [x1, y1, x2, y2, score, class]
    for det in prediction:
        x1, y1, x2, y2, score, label = det
        if score > threshold:
            x1, y1, x2, y2 = x1.cpu(), y1.cpu(), x2.cpu(), y2.cpu()
            score = score.cpu()
            label = int(label.cpu())
            
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
            class_name = class_names.get(label, f"Class {label}")
            plt.text(
                x1, y1 - 5,
                f"{class_name} {score:.2f}",
                color="red",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5)
            )
    
    plt.axis("off")
    plt.tight_layout()
    plt.show()


class SignsDataset(Dataset):
    """Dataset class for loading traffic sign images with COCO-format annotations."""
    
    def __init__(self, root, annFile, transforms, preload=True):
        self.root = root
        self.transforms = transforms

        with open(annFile) as f:
            data = json.load(f)

        self.annotations = data["annotations"]

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

        boxes = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in boxes],
            dtype=torch.float32
        )
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        return img, target

    def __len__(self):
        return len(self.images_info)
