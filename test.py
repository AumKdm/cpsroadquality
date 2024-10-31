import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the data
train_dataset = datasets.ImageFolder('frames/train', transform=transform)
val_dataset = datasets.ImageFolder('frames/val', transform=transform)
test_dataset = datasets.ImageFolder('frames/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 2: Load DenseNet Model
def get_densenet_model(num_classes):
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# Initialize the models
road_type_model = get_densenet_model(num_classes=2)
road_type_model.to(device)

road_quality_model = get_densenet_model(num_classes=2)
road_quality_model.to(device)

# Step 3: Train the Model Function
def train_model(model, criterion, optimizer, dataloader, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    return model

# Step 4: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_road_type = optim.Adam(road_type_model.parameters(), lr=0.001)
optimizer_road_quality = optim.Adam(road_quality_model.parameters(), lr=0.001)

# Train both models
print("Training road type model...")
road_type_model = train_model(road_type_model, criterion, optimizer_road_type, train_loader, num_epochs=10)
torch.save(road_type_model.state_dict(), 'road_type_densenet_model.pth')

print("Training road quality model...")
road_quality_model = train_model(road_quality_model, criterion, optimizer_road_quality, train_loader, num_epochs=10)
torch.save(road_quality_model.state_dict(), 'road_quality_densenet_model.pth')

# Step 5: Real-Time Video Classification (unchanged, ensure models are loaded and evaluated)
road_type_model.load_state_dict(torch.load('road_type_densenet_model.pth', map_location=device))
road_type_model.eval()

road_quality_model.load_state_dict(torch.load('road_quality_densenet_model.pth', map_location=device))
road_quality_model.eval()

# Real-time classification code goes here as per your original implementation
video_capture = cv2.VideoCapture('C:/Users/HP/Desktop/Road_quality/1.mp4')  # Replace with your custom video path

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    road_type, road_quality = classify_frame(frame)
    cv2.putText(frame, f'Road Type: {road_type}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Road Quality: {road_quality}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Road Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()