import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.ops import box_iou
from PIL import Image
import matplotlib.pyplot as plt
import json, os

# -------------------------
# Classes
# -------------------------
CLASS_NAMES = {
    1: "Stop",
    2: "No Entry (One Way)",
    3: "Pedestrian Crossing",
    4: "Roundabout Ahead",
    5: "No Through Road (T-Sign)",
    6: "Blind-Spot Mirror (Convex)"
}

NUM_CLASSES = len(CLASS_NAMES) + 1  # + background

# -------------------------
# Dataset
# -------------------------
class SignsDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, preload=True):
        self.root = root
        self.transforms = transforms

        with open(annFile) as f:
            data = json.load(f)

        self.images_info = data["images"]
        self.annotations = data["annotations"]

        self.imgToAnns = {img["id"]: [] for img in self.images_info}
        for ann in self.annotations:
            self.imgToAnns[ann["image_id"]].append(ann)

        self.preload = preload
        if preload:
            self.loaded_images = []
            for img_info in self.images_info:
                img_path = os.path.join(root, img_info["file_name"])
                with Image.open(img_path) as img:
                    self.loaded_images.append(img.convert("RGB").copy())

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        image_id = img_info["id"]

        if self.preload:
            img = self.loaded_images[idx].copy()
        else:
            img_path = os.path.join(self.root, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")

        anns = self.imgToAnns[image_id]

        # Resize image
        w0, h0 = img.size
        img = T.functional.resize(img, (512, 512))
        w1, h1 = 512, 512

        scale_x = w1 / w0
        scale_y = h1 / h0

        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

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

# -------------------------
# Transforms
# -------------------------
def get_transform():
    return T.Compose([T.ToTensor()])

# -------------------------
# mAP@0.5 Evaluation
# -------------------------
def evaluate_map(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].to(device)
                pred_boxes = output["boxes"]
                scores = output["scores"]

                keep = scores > 0.5
                pred_boxes = pred_boxes[keep]

                if len(pred_boxes) == 0:
                    total_fn += len(gt_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                for i in range(len(pred_boxes)):
                    max_iou, gt_idx = ious[i].max(0)
                    if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                        total_tp += 1
                        matched_gt.add(gt_idx.item())
                    else:
                        total_fp += 1

                total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    map50 = precision * recall

    return map50

# -------------------------
# Dataset & Loaders
# -------------------------
dataset = SignsDataset(
    "stefania_livori/images",
    "stefania_livori/result.json",
    get_transform()
)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Model
# -------------------------
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# -------------------------
# Optimizer & Scheduler
# -------------------------
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

# -------------------------
# Training Loop
# -------------------------
num_epochs = 30

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

# -------------------------
# Inference + Visualization
# -------------------------
model.eval()

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
