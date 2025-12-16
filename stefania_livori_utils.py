import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.ops import box_iou
import matplotlib.pyplot as plt


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def get_faster_rcnn(num_classes):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_map50(model, loader, device, iou_threshold=0.5):
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)

                keep = out["scores"] > 0.5
                pred_boxes = out["boxes"][keep]
                pred_labels = out["labels"][keep]

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
    return precision * recall


def visualize_predictions(img, prediction, class_names, threshold=0.6):
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
