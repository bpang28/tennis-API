import torch
import torch.nn as nn
import numpy as np
import math

class BounceNetV3(nn.Module):
    """
    A feedforward neural network used for bounce classification in tennis tracking.

    The architecture consists of:
    - Input layer of size `input_dim`
    - Hidden layers with BatchNorm, ReLU, and Dropout for regularization
    - Final output layer with a single unit (no activation)

    This model is intended to be used with sigmoid activation **after** inference,
    so the last layer does not include Sigmoid.

    Parameters:
        input_dim (int): The number of features in the input vector.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # No sigmoid; apply during inference if needed
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Raw output logits of shape (batch_size, 1)
        """
        return self.net(x)

class BounceClassifier:
    """
    A bounce detection classifier using a pre-trained BounceNetV3 model.

    This class loads a trained model and uses it to classify whether a tennis
    ball is bouncing based on a window of trajectory data.

    Attributes:
        window_size (int): Number of frames used before and after the center frame.
        threshold (float): Classification threshold for deciding a bounce.
        model (BounceNetV3): Loaded neural network model for inference.

    Parameters:
        model_path (str): Path to the trained model (.pt file).
        window_size (int): Size of the frame window to analyze (must be odd).
        threshold (float): Probability threshold to classify as a bounce (default = 0.7).
    """

    def __init__(self, model_path: str, window_size: int, threshold: float = 0.7):
        self.window_size = window_size
        self.threshold = threshold

        # Compute the input dimension based on window size and feature set:
        # - 2 values (x, y) per frame
        # - (window_size - 1) speed values between adjacent pairs
        # - 3 aggregated features: avg speed before, avg speed after, avg direction change
        input_dim = (self.window_size * 2) + (self.window_size - 1) + 3

        # Initialize and load the trained model
        self.model = BounceNetV3(input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def extract_features_from_detections(self, detections, center_idx, missing_value=-1.0):
        """
        Extracts features from a window of ball (x, y) detections centered around a given frame.

        The extracted feature vector includes:
        - Flattened (x, y) coordinates for each frame in the window
        - Speeds between each pair of adjacent frames
        - Average speed before and after the center frame
        - Average angle of direction change (in degrees)

        Missing or invalid coordinates are replaced with `missing_value`.

        Parameters:
            detections (list[tuple[float, float] | None]):
                List of (x, y) coordinates per frame or None if undetected.
            center_idx (int):
                Index of the center frame to extract features around.
            missing_value (float, optional):
                Value used to fill in missing coordinates. Default is -1.0.

        Returns:
            list[float]: A single flattened feature vector combining positions, speeds, and direction metrics.
        """
        half = self.window_size // 2
        coords = []

        # Step 1: Collect (x, y) coordinates from the detection window
        for i in range(center_idx - half, center_idx + half + 1):
            if (
                0 <= i < len(detections)
                and detections[i] is not None
                and None not in detections[i]
            ):
                coords.append(detections[i])
            else:
                coords.append((missing_value, missing_value))

        # Step 2: Flatten coordinates list [(x, y), ...] → [x1, y1, x2, y2, ...]
        flat_coords = [value for point in coords for value in point]

        # Step 3: Compute speeds between each adjacent pair of points
        speeds = []
        for j in range(len(coords) - 1):
            pt1, pt2 = coords[j], coords[j + 1]
            if missing_value in pt1 or missing_value in pt2:
                speeds.append(0.0)
            else:
                pt1_np = np.array(pt1)
                pt2_np = np.array(pt2)
                speeds.append(np.linalg.norm(pt2_np - pt1_np))

        # Step 4: Compute average speeds before and after the center point
        avg_speed_before = np.mean(speeds[:half])
        avg_speed_after = np.mean(speeds[half:])

        # Step 5: Calculate angle change between three consecutive points
        direction_changes = []
        for j in range(1, len(coords) - 1):
            a, b, c = coords[j - 1], coords[j], coords[j + 1]
            if (
                missing_value in a
                or missing_value in b
                or missing_value in c
            ):
                direction_changes.append(0.0)
                continue
            v1 = np.array(b) - np.array(a)
            v2 = np.array(c) - np.array(b)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                direction_changes.append(0.0)
            else:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)
                direction_changes.append(np.degrees(angle))

        avg_dir_change = np.mean(direction_changes)

        # Step 6: Combine everything into a single feature vector
        return flat_coords + speeds + [avg_speed_before, avg_speed_after, avg_dir_change]


    def predict(self, detections, center_idx: int) -> float:
        """
        Predicts the probability that a bounce occurred at the given frame index.

        This method:
        1. Extracts features from the window of detections centered on `center_idx`.
        2. Converts those features into a PyTorch tensor.
        3. Passes the tensor through the trained bounce classifier model.
        4. Applies sigmoid activation to get the probability.

        Parameters:
            detections (list[tuple[float, float] | None]):
                List of (x, y) coordinates or None values per frame.
            center_idx (int):
                Index of the center frame to predict for.

        Returns:
            float: Probability that the center frame is a bounce frame (between 0.0 and 1.0).
        """
        features = self.extract_features_from_detections(detections, center_idx, self.window_size)
        x = torch.tensor(features).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()
        return prob
