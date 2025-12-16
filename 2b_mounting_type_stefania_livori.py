import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import json, os


MOUNTING_CLASSES = {
    1: "Pole-mounted",
    2: "Wall-mounted",
}
NUM_CLASSES = len(MOUNTING_CLASSES) + 1  # +1 


class MountingDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(annFile) as f:
            self.data = json.load(f)

        self.tasks = self.data

        self.MOUNTING_MAP = {
            "Pole-mounted": 1,
            "Wall-mounted": 2
        }

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]

        img_path = os.path.join(self.root, task["data"]["image"])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in task["annotations"]:
            for res in ann["result"]:
                if res["type"] == "rectanglelabels":
                    x = res["value"]["x"]
                    y = res["value"]["y"]
                    w = res["value"]["width"]
                    h = res["value"]["height"]

                    img_w, img_h = img.size

                    # convert % → pixels
                    x1 = x / 100 * img_w
                    y1 = y / 100 * img_h
                    x2 = (x + w) / 100 * img_w
                    y2 = (y + h) / 100 * img_h

                    boxes.append([x1, y1, x2, y2])

                    # find mounting choice
                    for r in ann["result"]:
                        if r["from_name"] == "mounting":
                            mount = r["value"]["choices"][0]
                            labels.append(self.MOUNTING_MAP[mount])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

dataset = MountingDataset(
    "label-studio/label_studio/media/upload",
    "json_stefania.json",
    transforms=T
)


model = fasterrcnn_resnet50_fpn(
    pretrained=True,
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, NUM_CLASSES
)
