import torch
import json, os
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
from stefania_livori_utils import *

# Classes names to id mapping
MOUNTING_CLASSES = {
    1: "Pole-mounted",
    2: "Wall-mounted"
}

NUM_CLASSES = len(MOUNTING_CLASSES) + 1

class MountingDataset(Dataset):
    # root - folder containing images
    # ann_file - Label Studio JSON export
    # transforms - image preprocessing
    # preload - load images into memory for faster training
    def __init__(self, root, ann_file, transforms=None, preload=True):
        self.root = root
        self.transforms = transforms

        # Load the annotations
        with open(ann_file) as f:
            self.tasks = json.load(f)

        self.preload = preload
        # Preload images into memory for faster training
        if preload:
            self.loaded_images = []
            for task in self.tasks:
                img_rel_path = task["data"]["image"].replace("/data/upload/", "")
                img_path = os.path.join(self.root, img_rel_path)
                with Image.open(img_path) as img:
                    self.loaded_images.append(img.convert("RGB").copy())

        self.map = {
            "Pole-mounted": 1,
            "Wall-mounted": 2
        }

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]

        if self.preload:
            # load the image - at the current position
            img = self.loaded_images[idx].copy()
        else:
            img_rel_path = task["data"]["image"].replace("/data/upload/", "")
            img_path = os.path.join(self.root, img_rel_path)
            img = Image.open(
                img_path
            ).convert("RGB")

        # Load the bounding boxes and labels
        boxes, labels = [], []

        for ann in task["annotations"]:
            mount = None
            rects = []

            for r in ann["result"]:
                if r["from_name"] == "mounting":
                    mount = self.map[r["value"]["choices"][0]]
                if r["type"] == "rectanglelabels":
                    rects.append(r)

            for r in rects:
                x = r["value"]["x"]
                y = r["value"]["y"]
                w = r["value"]["width"]
                h = r["value"]["height"]
                iw, ih = img.size
                boxes.append([
                    x/100*iw, y/100*ih,
                    (x+w)/100*iw, (y+h)/100*ih
                ])
                labels.append(mount)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


transform = T.Compose([T.ToTensor()])

dataset = MountingDataset(
    root="label-studio/label-studio/media/upload",
    ann_file= "json_stefania.json",
    transforms=transform
)

total_size = len(dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, 4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_ds, 2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

device = get_device()
model = get_faster_rcnn(NUM_CLASSES).to(device)

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

for epoch in range(5):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    scheduler.step()
    map50 = evaluate_map50(model, val_loader, device)
    print(f"Epoch {epoch+1} | Loss {loss:.4f} | mAP@0.5 {map50:.4f}")
