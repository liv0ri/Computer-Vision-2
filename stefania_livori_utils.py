import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
)
from functools import partial
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
# Array of anchor boxes
from torchvision.models.detection.rpn import AnchorGenerator
import os
import zipfile
from torch.amp import autocast
from torch.utils.data import Dataset
import json
from PIL import Image

def get_device():
    # Use CUDA if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_faster_rcnn(num_classes):
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=600,
        max_size=1000,
        rpn_anchor_generator=anchor_generator
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    # Return the modified model
    return model

def get_retinanet(num_classes):
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")  # using most recent weights

    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_channels=256, num_anchors=num_anchors, num_classes=num_classes, norm_layer=partial(torch.nn.GroupNorm, 32))
    return model

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0

    # Iterate over the data loader
    # Different order each time due to the shuffle = true in the train_loader
    for images, targets in loader:
        # Move the data to the device
        images = [img.to(device) for img in images]
        # Move targets to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        # Forward pass
        with autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, targets)

            # Works for Faster R-CNN and RetinaNet
            loss = sum(loss_dict.values())

            cls_loss = sum(
                v for k, v in loss_dict.items()
                if "class" in k
            )

            box_loss = sum(
                v for k, v in loss_dict.items()
                if "box" in k or "reg" in k
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update the total loss
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_box_loss += box_loss.item()

    num_batches = len(loader)

    return {
        "total": total_loss / num_batches,
        "cls": total_cls_loss / num_batches,
        "box": total_box_loss / num_batches,
    }

def f1_score_by_iou(model, loader, device, iou_threshold=0.5, score_threshold=0.1):
    # Set model to evaluation mode
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        # Iterate over the data loader - BATCH PROCESSING - batch of images and batch of targets
        for images, targets in loader:
            images = [img.to(device) for img in images]
            # Outputs for these batch of images
            outputs = model(images)

            # Iterate over the batch of predictions and targets 
            for out, tgt in zip(outputs, targets):
                # Move ground truth to device
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)
                # Get the predictions
                # scores = out["scores"]
                # For debugging: print max score
                # max_score = scores.max().item() if len(scores) > 0 else 0
                # if max_score > 0.1: # Only print if there's some confidence
                #     print(f"Image ID {tgt['image_id'].item()}: Max prediction score = {max_score:.4f}")

                # Threshold for filtering predictions
                keep = out["scores"] > score_threshold
                # Keep only the predictions that are above the threshold
                pred_boxes = out["boxes"][keep]
                pred_labels = out["labels"][keep]

                # If no predictions, all gt are false negatives
                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue

                # Compute IoU between predicted and ground truth boxes 
                # It is done between every box in A and every box in B
                ious = box_iou(pred_boxes, gt_boxes)
                # Find best match gt for each predicted box
                matched = set()

                for i in range(len(pred_boxes)):
                    max_iou, idx = ious[i].max(0)
                    if (
                        # If max iou of this box is greater than threshold
                        max_iou >= iou_threshold
                        # It is a new box - not a duplicate
                        and idx.item() not in matched
                        # And the label matches
                        and pred_labels[i] == gt_labels[idx]
                    ):
                        # Number of correct predictions
                        tp += 1
                        matched.add(idx.item())
                    elif (max_iou < iou_threshold):
                        # Number of incorrect predictions expected as correct
                        fp += 1

                fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    # F1 score
    return 2 * (precision * recall) / (precision + recall + 1e-6) 


def visualize_predictions(img, prediction, class_names, threshold=0.1):
    plt.imshow(img.permute(1, 2, 0))
    for box, score, label in zip(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"]
    ):
        if score > threshold:
            x1, y1, x2, y2 = box.cpu()
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
            plt.text(
                x1, y1 - 5,
                f"{class_names[label.item()]} {score:.2f}",
                color="red"
            )
    plt.axis("off")
    plt.show()

def unzip_folder(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)

# https://medium.com/@RobuRishabh/understanding-and-implementing-faster-r-cnn-248f7b25ff96
class SignsDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, preload=True):
        self.root = root
        self.transforms = transforms

        # Load the JSON file
        with open(annFile) as f:
            data = json.load(f)

        # Extract images and annotations
        self.annotations = data["annotations"]
        # For debugging: print number of images and annotations
        # print(f"Loaded {len(self.images_info)} images and {len(self.annotations)} annotations.")

        self.preload = preload
        self.loaded_images = []
        self.images_info = []

        for img_info in data["images"]:
            # Handle potential path differences if filename contains folders
            # Assuming images are directly in root or filename matches relative structure
            img_name = os.path.basename(img_info["file_name"])
            img_path = os.path.join(root, img_name)

            if not os.path.exists(img_path):
                print(f"Image not found at {img_path}")
                # Skip this image
                continue

            self.images_info.append(img_info)

            # If preload is True, load all images into memory
            # Preload the images into memory for speed
            # This might cause memory usage
            if preload:
                with Image.open(img_path) as img:
                    self.loaded_images.append(img.convert("RGB").copy())
        
        # Map image_id to annotations for faster access
        self.imgToAnns = {img["id"]: [] for img in self.images_info}
        for ann in data["annotations"]:
            if ann["image_id"] in self.imgToAnns:
                self.imgToAnns[ann["image_id"]].append(ann)


    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        image_id = img_info["id"]

        if self.preload:
            # load the image 
            img = self.loaded_images[idx].copy()
        else:
            img_path = os.path.join(self.root, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")

        anns = self.imgToAnns[image_id]
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Based on the coco definition
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            # Converted to pytorch tensor
            "image_id": torch.tensor([image_id])
        }

        if self.transforms:
            img = self.transforms(img)

        # For debugging: print target labels in torch tensor format
        # print(torch.unique(target["labels"]))

        return img, target

    def __len__(self):
        return len(self.images_info)

class MountingDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, preload=True):
        self.root = root
        self.transforms = transforms

        # Load the JSON file
        with open(annFile) as f:
            data = json.load(f)

        # Extract images and annotations
        self.annotations = data["annotations"]
        
        self.preload = preload
        self.loaded_images = []
        self.images_info = []

        # Mapping for mounting types
        self.mounting_map = {
            "Pole-mounted": 1,
            "Wall-mounted": 2
        }

        for img_info in data["images"]:
            img_name = os.path.basename(img_info["file_name"])
            img_path = os.path.join(root, img_name)

            if not os.path.exists(img_path):
                # print(f"Image not found at {img_path}")
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

        anns = self.imgToAnns[image_id]
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            
            # Extract mounting attribute
            mounting_attr = ann.get("attributes", {}).get("mounting", [])
            
            if mounting_attr:
                mounting_str = mounting_attr[0] # It is a list
                label = self.mounting_map.get(mounting_str)
                
                # Only add if it is a known mounting type (1 or 2)
                if label:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(label)

        # Handle case with no valid boxes
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images_info)
