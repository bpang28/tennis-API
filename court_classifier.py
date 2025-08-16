import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the model architecture
def build_court_classifier():
    backbone = models.mobilenet_v2(pretrained=False)
    model = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(backbone.last_channel, 1),
        nn.Sigmoid()
    )
    return model

# Load pretrained model weights
model = build_court_classifier().to(device)
model.load_state_dict(torch.load("court_classifier.pt", map_location=device))
model.eval()

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def is_full_court(frame, thresh):
    """
    Determines whether a frame shows the full tennis court.

    Args:
        frame (np.ndarray): Input BGR frame from the video.
        thresh (float): Threshold to decide if the prediction is confident.

    Returns:
        tuple[bool, float]: (is_not_full_court, probability)
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device) # type: ignore

    with torch.no_grad():
        probability = float(model(input_tensor).item())

    return (probability < thresh), probability