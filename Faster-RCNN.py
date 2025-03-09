import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define dataset paths
data_root = "NAIRS-1.v1i.coco"  # Root directory containing train and validation folders
train_dir = os.path.join(data_root, "train")
valid_dir = os.path.join(data_root, "valid")

# Load datasets
train_dataset = CocoDetection(root=train_dir,
                              annFile=os.path.join(train_dir, "annotations.json"),
                              transform=transform)

valid_dataset = CocoDetection(root=valid_dir,
                              annFile=os.path.join(valid_dir, "annotations.json"),
                              transform=transform)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Print dataset sizes
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")

# Load Faster R-CNN model with pretrained weights
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Modify the model for the number of classes in your dataset
num_classes = 3  # Adjust based on dataset (background + classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to device (CPU only)
device = torch.device("cpu")
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Training loop
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        # Move images to device and stack them into a batch
        images = [img.to(device) for img in images]
        images = torch.stack(images)  # Stack list of tensors into a single batched tensor

        # Convert targets to the format expected by Faster R-CNN
        formatted_targets = []
        for t in targets:
            # Extract annotations
            annotations = t
            boxes = []
            labels = []
            for obj in annotations:
                # Extract bbox and category_id
                bbox = obj["bbox"]
                category_id = obj["category_id"]

                # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(category_id)

            # Create target dictionary
            formatted_targets.append({
                'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                'labels': torch.tensor(labels, dtype=torch.int64).to(device)
            })

        optimizer.zero_grad()
        loss_dict = model(images, formatted_targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Validation loop
def evaluate(model, valid_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in valid_loader:
            # Move images to device and stack them into a batch
            images = [img.to(device) for img in images]
            images = torch.stack(images)  # Stack list of tensors into a single batched tensor

            # Convert targets to the format expected by Faster R-CNN
            formatted_targets = []
            for t in targets:
                # Extract annotations
                annotations = t
                boxes = []
                labels = []
                for obj in annotations:
                    # Extract bbox and category_id
                    bbox = obj["bbox"]
                    category_id = obj["category_id"]

                    # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
                    x_min, y_min, width, height = bbox
                    x_max = x_min + width
                    y_max = y_min + height
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(category_id)

                # Create target dictionary
                formatted_targets.append({
                    'boxes': torch.tensor(boxes, dtype=torch.float32).to(device),
                    'labels': torch.tensor(labels, dtype=torch.int64).to(device)
                })

            outputs = model(images)
            all_preds.extend(outputs)
            all_targets.extend(formatted_targets)  # Use formatted_targets instead of targets
    return all_preds, all_targets


# Compute precision, recall, and mAP
def compute_metrics(preds, targets, iou_threshold=0.5):
    """
    Compute precision, recall, and mean Average Precision (mAP) for object detection.
    """
    aps = []  # List to store average precision for each class
    precisions = []  # List to store precision for each class
    recalls = []  # List to store recall for each class

    for class_id in range(num_classes):
        # Filter predictions and targets for the current class
        class_preds = []
        class_targets = []

        for pred, target in zip(preds, targets):
            # Filter predictions for the current class
            pred_class_mask = pred["labels"] == class_id
            pred_boxes = pred["boxes"][pred_class_mask]
            if len(pred_boxes) > 0:
                class_preds.append({
                    'boxes': pred_boxes,
                    'scores': pred["scores"][pred_class_mask]
                })

            # Filter targets for the current class
            target_class_mask = target["labels"] == class_id
            target_boxes = target["boxes"][target_class_mask]
            if len(target_boxes) > 0:
                class_targets.append({
                    'boxes': target_boxes
                })

        if not class_preds or not class_targets:
            # If no predictions or targets for this class, skip
            continue

        # Compute precision and recall for the current class
        correct_detections = 0
        total_detections = 0
        total_targets = sum(len(target["boxes"]) for target in class_targets)

        for pred, target in zip(class_preds, class_targets):
            pred_boxes = pred["boxes"].cpu()
            target_boxes = target["boxes"].cpu()
            iou = box_iou(pred_boxes, target_boxes)
            correct_detections += (iou > iou_threshold).sum().item()
            total_detections += len(pred_boxes)

        precision = correct_detections / total_detections if total_detections else 0
        recall = correct_detections / total_targets if total_targets else 0

        # Store precision and recall for the current class
        precisions.append(precision)
        recalls.append(recall)

        # Compute average precision for the current class
        ap = precision * recall
        aps.append(ap)

    # Compute mean precision, recall, and mAP
    mean_precision = sum(precisions) / len(precisions) if precisions else 0
    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    mAP = sum(aps) / len(aps) if aps else 0

    return mean_precision, mean_recall, mAP


# Training process
num_epochs = 30 #10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    preds, targets = evaluate(model, valid_loader, device)
    precision, recall, mAP = compute_metrics(preds, targets)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {mAP:.4f}")

# Save trained model
# torch.save(model.state_dict(), "fasterrcnn_trained.pth")