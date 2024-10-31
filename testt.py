import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

# Ensure you have these lines in the same file if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels for road type and road quality (replace these with your actual class names)
road_type_classes = ['asphalt', 'concrete']  # Replace these with your actual class names for road type
road_quality_classes = ['good', 'pothole']  # Replace these with your actual class names for road quality

# Function to get the model
def get_densenet_model(num_classes):
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# Load the models
road_type_model = get_densenet_model(num_classes=len(road_type_classes))
road_type_model.load_state_dict(torch.load('road_type_densenet_model.pth', map_location=device))
road_type_model.to(device)
road_type_model.eval()

road_quality_model = get_densenet_model(num_classes=len(road_quality_classes))
road_quality_model.load_state_dict(torch.load('road_quality_densenet_model.pth', map_location=device))
road_quality_model.to(device)
road_quality_model.eval()

# Transformation for frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Video path (ensure this path is correct)
video_path = 'C:/Users/HP/Downloads/istockphoto-1198609004-640_adpp_is.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Process each frame
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("End of video or failed to read the frame.")
        break

    # Convert frame to RGB format and apply transformations
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        type_output = road_type_model(input_tensor)
        quality_output = road_quality_model(input_tensor)

        _, type_pred = torch.max(type_output, 1)
        _, quality_pred = torch.max(quality_output, 1)

    # Use class labels for output
    road_type = road_type_classes[type_pred.item()]
    road_quality = road_quality_classes[quality_pred.item()]

    # Display results on frame
    cv2.putText(frame, f'Road Type: {road_type}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Road Quality: {road_quality}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Road Classification', frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
