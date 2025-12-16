# Import the necessary libraries
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.ops import box_iou
from PIL import Image
import matplotlib.pyplot as plt
import json, os

# Classes
# FasterRCNN uses background so background is set to 0 and we staart from 1
CLASS_NAMES = {
    1: "Stop",
    2: "No Entry (One Way)",
    3: "Pedestrian Crossing",
    4: "Roundabout Ahead",
    5: "No Through Road (T-Sign)",
    6: "Blind-Spot Mirror (Convex)"
}

 # + background
 # total of 7 classes
NUM_CLASSES = len(CLASS_NAMES) + 1 

# https://medium.com/@RobuRishabh/understanding-and-implementing-faster-r-cnn-248f7b25ff96
class SignsDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, preload=True):
        self.root = root
        self.transforms = transforms

        # Load the JSON file - the annotation file
        with open(annFile) as f:
            data = json.load(f)

        # Extract the images information
        self.images_info = data["images"]
        # Extract the annotations
        self.annotations = data["annotations"]

        # Create a dictionary to store the annotations for each image
        # this avoids slow scanning for each image
        self.imgToAnns = {img["id"]: [] for img in self.images_info}
        for ann in self.annotations:
            self.imgToAnns[ann["image_id"]].append(ann)

        self.preload = preload
        if preload:
            # Preload the images into memory for speed
            # This might cause memory usage
            self.loaded_images = []
            for img_info in self.images_info:
                img_path = os.path.join(root, img_info["file_name"])
                with Image.open(img_path) as img:
                    self.loaded_images.append(img.convert("RGB").copy())

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

        boxes, labels = [], []

        for ann in anns:
            # covert coco into faster rcnn
            x, y, w, h = ann["bbox"]
            x1 = x
            y1 = y 
            x2 = (x + w)
            y2 = (y + h) 

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                # Adjust labels to start from 1
                # Since this uses 0 as background but I was getting it as a label/class
                labels.append(ann["category_id"]+1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # build target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        # apply transforms - convert image to tensor
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images_info)

# Transforms - convert image to float tensor
transform = T.Compose([T.ToTensor()])

# mAP@0.5 Evaluation
def evaluate_map(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(device)
                gt_labels = target["labels"].to(device)

                pred_boxes = output["boxes"]
                pred_labels = output["labels"]
                scores = output["scores"]

                keep = scores > 0.05
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                if len(pred_boxes) == 0:
                    total_fn += len(gt_boxes)
                    continue

                # Compute IoU between predicted and ground truth boxes
                ious = box_iou(pred_boxes, gt_boxes)
                # Find best match gt for each predicted box
                matched_gt = set()

                for i in range(len(pred_boxes)):
                    max_iou, gt_idx = ious[i].max(0)

                    if (
                        max_iou >= iou_threshold
                        and gt_idx.item() not in matched_gt
                        and pred_labels[i] == gt_labels[gt_idx]
                    ):
                        total_tp += 1
                        matched_gt.add(gt_idx.item())
                    else:
                        total_fp += 1

                total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    return precision * recall

# Dataset & Loaders
dataset = SignsDataset(
    "stefania_livori/images",
    "stefania_livori/result.json",
    transforms=transform
)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# https://github.com/GirinChutia/FasterRCNN-Torchvision-FineTuning/blob/main/train.py
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
model = fasterrcnn_resnet50_fpn(
    pretrained=True,
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# Optimizer & Scheduler
# https://github.com/GirinChutia/FasterRCNN-Torchvision-FineTuning/blob/main/train.py
# optimizer logic based on the github link above
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# Might need to try different step sizes
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.1
)

# Training Loop
# https://medium.com/@RobuRishabh/understanding-and-implementing-faster-r-cnn-248f7b25ff96 
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    map50 = evaluate_map(model, val_loader, device)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {avg_loss:.4f} | mAP@0.5: {map50:.4f}"
    )

print("Training finished!")

# Inference + Visualization - Inference applying the trained model to unlabeled examples so that it can
# make the respective predictions.
model.eval()
img, target = val_dataset[0]

with torch.no_grad():
    pred = model([img.to(device)])[0]

print("GT labels:", target["labels"])
print("Pred labels:", pred["labels"][:5])
print("Scores:", pred["scores"][:5])

img, _ = dataset[0]
with torch.no_grad():
    prediction = model([img.to(device)])[0]

plt.imshow(img.permute(1, 2, 0))
for box, score, label in zip(
    prediction["boxes"],
    prediction["scores"],
    prediction["labels"]
):
    if score > 0.6:
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
            f"{CLASS_NAMES[label.item()]} {score:.2f}",
            color="red",
            fontsize=10
        )

plt.axis("off")
plt.show()
