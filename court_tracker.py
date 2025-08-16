import cv2
import numpy as np
from ultralytics import YOLO

class CourtTracker:
    def __init__(self, model_path):
        """
        Initialize the CourtTracker.

        Parameters:
            model_path (str): Path to the YOLO keypoint detection model file.
        
        Attributes:
            model (YOLO): The loaded YOLO model for detecting court keypoints.
            homographies (dict): Dictionary mapping frame indices to homography matrices.
        """
        self.model = YOLO(model_path)
        self.homographies = {}

    def predict(self, image):
        """
        Run the YOLO model to detect court keypoints in an image. 
        There are 14 keypoints total with 7 keypoints on each side. 
        - 4 on the baseline with 1 each on the left and right of the single sideline and 1 each on the left and right of double sideline
        - 3 on the serviceline with 1 each on the left and right of the single sideline and 1 on the center serviceline

        Parameters:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray | None:
                - NumPy array of shape (N, 2) containing detected keypoints (x, y) in pixel coordinates.
                - None if no keypoints are detected.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)

        if len(results) == 0 or len(results[0].keypoints.xy) == 0:
            return None

        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        return keypoints

    # Debugging tool to draw the keypoints. Not actually used in the actual analysis
    def draw_keypoints(self, image, keypoints):
        """
        Draw numbered keypoints on the image.

        Parameters:
            image (np.ndarray): The image (BGR format) on which to draw.
            keypoints (np.ndarray): Array of shape (N, 2) containing keypoint coordinates (x, y).

        Returns:
            np.ndarray: The image with keypoints and their indices drawn.
        """
        for i, (x, y) in enumerate(keypoints):
            x = int(x)
            y = int(y)

            cv2.putText(image, str(i), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw a red circle at the point
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        return image

    def store_homography(self, H, frame_idx):
        """
        Store a homography matrix for a specific frame index.

        Parameters:
            H (np.ndarray | None): The homography matrix to store. Ignored if None.
            frame_idx (int): The frame index associated with this homography.
        """
        if H is not None:
            self.homographies[frame_idx] = H

    def get_H_for_frame(self, frame_idx):
        """
        Retrieve the stored homography matrix for a given frame index.

        Parameters:
            frame_idx (int): The frame index whose homography should be retrieved.

        Returns:
            np.ndarray | None:
                - The stored homography matrix if available.
                - None if no homography is stored for this frame index.
        """
        return self.homographies.get(frame_idx, None)
