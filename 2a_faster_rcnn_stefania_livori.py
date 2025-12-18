# Import the necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import json, os
from stefania_livori_utils import *

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

device  = get_device()
# Model 
model = get_faster_rcnn(NUM_CLASSES).to(device)

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
    avg_loss = train_one_epoch(model, train_loader, optimizer, device)
    scheduler.step()
    map50 = evaluate_map50(model, val_loader, device)

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

for img, target in val_loader:
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    
    visualize_predictions(img, prediction, CLASS_NAMES)


