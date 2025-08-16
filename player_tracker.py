import cv2
import math
import pickle
import random
import numpy as np
from ultralytics import YOLO
from ultralytics import SAM

class PlayerTracker:
    def __init__(self, model_path):
        """
        Initialize the PlayerTracker.

        Parameters:
            model_path (str): Path to the YOLO model file used for player detection.

        Attributes:
            model (YOLO): The loaded YOLO model for tracking players in frames.
            prev_positions (dict): Stores previous player positions keyed by player ID.
            min_speeds (dict): Stores the minimum observed speed for each player ID.
            sam_model (SAM | None): (Optional) A loaded SAM model for player segmentation,
                                    currently commented out.
        """
        self.model = YOLO(model_path)
        # self.sam_model = SAM("sam2.1_b.pt")
        self.prev_positions = {}
        self.min_speeds = {}
        self.next_persistent_id = 1

    def calculate_speed(self, prev_center, curr_center):
        """
        Calculate the speed (pixel distance) between two positions.

        Parameters:
            prev_center (tuple[float, float]): The previous (x, y) position.
            curr_center (tuple[float, float]): The current (x, y) position.

        Returns:
            float: The Euclidean distance in pixels between the two positions.
        """
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        return math.sqrt(dx * dx + dy * dy)

    def detect_frame(self, frame, speed_threshold=1):
        """
        Detect and track players in a single frame using YOLO tracking.

        Parameters:
            frame (np.ndarray): The input video frame (BGR format) for detection.
            speed_threshold (float): Currently unused; reserved for future filtering based on player speed.

        Returns:
            tuple:
                - dict: Mapping of player track IDs to bounding boxes [x1, y1, x2, y2] in pixels.
                - float: Horizontal scale factor (currently fixed at 1).
                - float: Vertical scale factor (currently fixed at 1).
        """
        results = self.model.track(source=frame, stream=False, persist=True, verbose=False)[0]
        id_name_dict = results.names
        player_dict = {}

        if results.boxes is None or len(results.boxes) == 0:
            return player_dict

        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                object_cls_id = box.cls.tolist()[0]
                object_cls_name = id_name_dict[object_cls_id]

                if object_cls_name == "person":
                    bbox = box.xyxy.tolist()[0]
                    player_dict[track_id] = bbox

        return player_dict, 1, 1

    # For the SAM model which is currently not in use in the analysis
    def segment_players_in_frame(self, frame):
        """
        Detect and segment players in a single frame using the SAM model.

        Steps:
            1. Detect players and get their bounding boxes.
            2. Downscale the frame to 512×512 for faster segmentation.
            3. Scale bounding boxes to match the downscaled frame.
            4. Run the SAM model on the downscaled frame with scaled boxes.
            5. Resize the resulting segmentation masks back to the original frame size.

        Parameters:
            frame (np.ndarray): The input frame (BGR format).

        Returns:
            list: A list of boolean NumPy arrays (masks) where True indicates the
                  segmented player region.
        """
        # Step 1: detect players
        player_dict = self.detect_frame(frame)
        player_boxes = list(player_dict.values())

        # Step 2: downscale frame for faster SAM processing
        small_frame = cv2.resize(frame, (512, 512))

        # Step 3: scale boxes to match the new size
        orig_h, orig_w = frame.shape[:2]
        scale_x = 512 / orig_w
        scale_y = 512 / orig_h
        scaled_boxes = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for x1, y1, x2, y2 in player_boxes
        ]

        # Step 4: Run SAM on downscaled frame and bounding boxes
        results = self.sam_model(small_frame, bboxes=scaled_boxes)

        # Step 5: Resize masks back to original size
        masks = []
        if results and results[0].masks is not None:
            for i in range(len(scaled_boxes)):
                small_mask = results[0].masks.data[i].cpu().numpy().astype(np.uint8)
                full_mask = cv2.resize(small_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                masks.append(full_mask.astype(bool))

    # For the SAM model which is currently not in use in the analysis
    def draw_masks_on_frame(self, frame, masks):
        """
        Overlay segmentation masks on a frame with random colors.

        Parameters:
            frame (np.ndarray): The original frame (BGR format).
            masks (list[np.ndarray]): List of boolean masks where True indicates the segmented area.

        Returns:
            np.ndarray: Frame with colored masks blended on top.
        """
        frame_copy = frame.copy()

        for mask in masks:
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            colored_mask = np.zeros_like(frame_copy)
            colored_mask[mask] = color
            frame_copy = cv2.addWeighted(frame_copy, 1.0, colored_mask, 0.5, 0)

        return frame_copy

    def draw_bbox_on_frame(self, frame, player_dict):
        """
        Draw bounding boxes and player IDs on a frame.

        Parameters:
            frame (np.ndarray): The original frame (BGR format).
            player_dict (dict): Mapping of track IDs to bounding boxes [x1, y1, x2, y2].

        Returns:
            np.ndarray: Frame with bounding boxes and IDs drawn.
        """
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
