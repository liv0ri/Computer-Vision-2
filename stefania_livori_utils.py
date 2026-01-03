import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.ops import box_iou
import matplotlib.pyplot as plt
# Array of anchor boxes
from torchvision.models.detection.rpn import AnchorGenerator



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
        rpn_anchor_generator=anchor_generator
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    # Return the modified model
    return model


def train_one_epoch(model, loader, optimizer, device):
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

        # Forward pass
        loss_dict = model(images, targets)

        cls_loss = loss_dict["loss_classifier"]
        box_loss = loss_dict["loss_box_reg"]

        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
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
                    else:
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
