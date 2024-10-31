import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torchvision import models

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation for the test dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder('frames/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load DenseNet model structure
def get_densenet_model(num_classes):
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model

# Load the trained models
road_type_model = get_densenet_model(num_classes=2)
road_type_model.load_state_dict(torch.load('road_type_densenet_model.pth', map_location=device))
road_type_model.to(device)
road_type_model.eval()

road_quality_model = get_densenet_model(num_classes=2)
road_quality_model.load_state_dict(torch.load('road_quality_densenet_model.pth', map_location=device))
road_quality_model.to(device)
road_quality_model.eval()
video_path = 'C:/Users/HP/Desktop/Road_quality/1.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()
# Function to evaluate a model
def evaluate_model(model, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds

# Evaluate the road type model
print("Evaluating Road Type Model...")
labels, preds = evaluate_model(road_type_model, test_loader)
print("Classification Report for Road Type Model:")
print(classification_report(labels, preds, target_names=test_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))

# Evaluate the road quality model
print("\nEvaluating Road Quality Model...")
labels, preds = evaluate_model(road_quality_model, test_loader)
print("Classification Report for Road Quality Model:")
print(classification_report(labels, preds, target_names=test_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))



