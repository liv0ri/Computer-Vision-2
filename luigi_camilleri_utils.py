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

# Gets the best available device (CUDA if available, else CPU).
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

# Creates an EfficientDet model for training.
def get_efficientdet_train(num_classes):

    config = get_efficientdet_config('tf_efficientdet_d2')
    config.num_classes = num_classes
    
    # Creates base model
    net = EfficientDet(config, pretrained_backbone=True)
    
    # Reset classification head for new number of classes
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    
    # Wraps in training benchmark
    model = DetBenchTrain(net, config)
    return model, config

# Trains the model for one epoch and returns loss statistics.
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    for images, targets in loader:
        # Stacks images into batch tensor
        images = torch.stack([img.to(device) for img in images])
        
        # Prepares targets for EfficientDet
        boxes = [t["boxes"].to(device) for t in targets]
        labels = [t["labels"].to(device) for t in targets]
        
        # Creates target dict for EfficientDet
        target_dict = {
            "bbox": boxes,
            "cls": labels,
        }

        # Zero gradients
        optimizer.zero_grad()
        
        # autocast for mixed precision
        with autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, target_dict)
            
            # EfficientDet returns loss dict with 'loss', 'class_loss', 'box_loss'
            loss = loss_dict['loss']
            cls_loss = loss_dict.get('class_loss', torch.tensor(0.0))
            box_loss = loss_dict.get('box_loss', torch.tensor(0.0))

        # Backpropagation
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

# Evaluates mAP using COCO evaluation metrics.
def evaluate_map(model, data_loader, device, coco_gt, config):

    # Creates prediction wrapper for evaluation
    eval_model = DetBenchPredict(model.model)
    eval_model.to(device)
    eval_model.eval()
    
    coco_results = []

    # Prediction loop
    with torch.no_grad():
        for images, targets in data_loader:
            images = torch.stack([img.to(device) for img in images])
            outputs = eval_model(images)

            # Loops each pair of ground truth target and model output in the batch
            for target, output in zip(targets, outputs):
                # Gets the image ID for this sample
                image_id = int(target["image_id"])
                
                # Extracts predicted bounding boxes, scores, and labels from the output tensor
                boxes = output[:, :4].cpu().numpy()   # Predicted box coordinates [x1, y1, x2, y2]
                scores = output[:, 4].cpu().numpy()   # Confidence scores for each prediction
                labels = output[:, 5].cpu().numpy()   # Predicted class labels

                # Loops for each predicted box, score, and label
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.01:  # Only keeps predictions with confidence above threshold
                        coco_results.append({
                            "image_id": image_id,           # The image this prediction belongs to
                            "category_id": int(label),      # The predicted class label
                            "bbox": [                       # Convert [x1, y1, x2, y2] to [x, y, width, height] (COCO format)
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1]),
                            ],
                            "score": float(score),          # The confidence score for this prediction
                        })

    # Handles case with no predictions
    if len(coco_results) == 0:
        print("No predictions to evaluate")
        return [0.0] * 12
        
    # Results evaluation stats
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

# Calculates F1 score based on IoU matching.
def f1_score_by_iou(model, loader, device, config, iou_threshold=0.5, score_threshold=0.1):
    # Creates prediction wrapper
    eval_model = DetBenchPredict(model.model)
    eval_model.to(device)
    eval_model.eval()
    
    # Initialisies true positives, false positives, false negatives
    tp = fp = fn = 0

    with torch.no_grad():  # Gets the present state where gradients are not tracked
        for images, targets in loader:  # Gets each batch of images and targets from the loader
            images = torch.stack([img.to(device) for img in images])
            outputs = eval_model(images)

            for out, tgt in zip(outputs, targets):  # Gets each prediction and its corresponding ground truth
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)  

                # Shows EfficientDet output format
                scores = out[:, 4]  # Gets the present prediction scores
                keep = scores > score_threshold  # Gets the present mask for predictions above the score threshold
                
                pred_boxes = out[keep][:, :4]  # Gets the present predicted boxes above threshold
                pred_labels = out[keep][:, 5].long()  # Gets the present predicted labels above threshold

                if len(pred_boxes) == 0:  # Shows the case where there are no predictions
                    fn += len(gt_boxes)

                ious = box_iou(pred_boxes, gt_boxes)  # Gets the present IoU matrix between predictions and ground truths
                matched = set() 

                for i in range(len(pred_boxes)):  
                    max_iou, idx = ious[i].max(0)  # Gets the present maximum IoU and its index for this prediction
                    # If the maximum IoU exceeds the threshold, it is a true positive else a false positive
                    if (
                        max_iou >= iou_threshold  
                        and idx.item() not in matched  
                        and pred_labels[i] == gt_labels[idx]  
                    ):
                        tp += 1  
                        matched.add(idx.item())  
                    else:
                        fp += 1

                fn += len(gt_boxes) - len(matched)  # Gets the present false negatives

    precision = tp / (tp + fp + 1e-6)  # Gets the precision
    recall = tp / (tp + fn + 1e-6)  # Gets the recall
    return 2 * (precision * recall) / (precision + recall + 1e-6)

# Visualises predictions on an image.
def visualize_predictions(img, prediction, class_names, threshold=0.1):
    plt.figure(figsize=(12, 8))
    plt.imshow(img.permute(1, 2, 0).cpu())
    

    for det in prediction:
        x1, y1, x2, y2, score, label = det
        # If the score exceeds the threshold, draw the bounding box and label
        if score > threshold:
            x1, y1, x2, y2 = x1.cpu(), y1.cpu(), x2.cpu(), y2.cpu()
            score = score.cpu()
            label = int(label.cpu())
            
            # Draws the bounding box as a rectangle on the image
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,  # Width of the box
                    y2 - y1,  # Height of the box
                    fill=False,
                    edgecolor="red",
                    linewidth=2
                )
            )
            class_name = class_names.get(label, f"Class {label}")
            # Draws the class name and score above the bounding box
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

# Dataset class for loading traffic sign images with COCO-format annotations.
class SignsDataset(Dataset):
    # Initializes the dataset with image root directory, annotation file, transformations, and preload option.
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

    # Retrieves an image and its corresponding target annotations.
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

    # Returns the length of the dataset.
    def __len__(self):
        return len(self.images_info)
