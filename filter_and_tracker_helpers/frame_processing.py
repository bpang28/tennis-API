import cv2
import numpy as np
from court_classifier import is_full_court
from player_analysis import (
    get_top_two_players,
    project_player_to_court,
    draw_players_on_minimap,
    draw_fixed_minimap
)

def resize_frame(frame, width, height):
    """
    Resize the input frame to the specified width and height, 
    only if it doesn't already match.

    Args:
        frame (np.ndarray): Input image (BGR).
        width (int): Desired width.
        height (int): Desired height.

    Returns:
        np.ndarray: Resized image (or original if already correct size).
    """
    current_height, current_width = frame.shape[:2]
    if current_width != width or current_height != height:
        frame = cv2.resize(frame, (width, height))
    return frame

def should_keep_frame(frame_idx, frame, sample_rate, thresh):
    """
    Determines whether a frame should be kept based on its index 
    and a full-court detection score.

    Args:
        frame_idx (int): Index of the current frame.
        frame (np.ndarray): BGR image frame.
        sample_rate (int): Frequency of sampling (e.g., 10 = every 10th frame).
        thresh (float): Probability threshold for full court detection.

    Returns:
        tuple[bool, float | None]: 
            - True if frame passes both sampling and detection.
            - Detection probability if evaluated, else None.
    """
    if frame_idx % sample_rate != 0:
        return False, None

    keep_frame, probability = is_full_court(frame, thresh)
    return keep_frame, probability

def draw_frame_number(frame, frame_idx, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(0, 0, 0), thickness=2, offset_x=400, offset_y=30):
    """
    Draw the current frame number as a label in the bottom-right corner of the image.

    Args:
        frame (np.ndarray): Input image (BGR).
        frame_idx (int): Frame number to overlay.
        font (int): Font type for the text.
        scale (float): Font scaling factor.
        color (tuple[int, int, int]): BGR color of the text.
        thickness (int): Thickness of the text.
        offset_x (int): Horizontal offset from the right edge.
        offset_y (int): Vertical offset from the bottom edge.

    Returns:
        np.ndarray: Image with frame number overlaid.
    """
    height, width = frame.shape[:2]
    pos = (width - offset_x, height - offset_y)
    cv2.putText(frame, f"Frame {frame_idx}", pos, font, scale, color, thickness)
    return frame

def process_warmup_frames(frame, keypoints, all_players, idx, minimap_scale, minimap_margin, extra_m, player_tracker):
    """
    Redraw a buffered warmup frame with minimap, bounding boxes, and player dots.

    Args:
        frame (np.ndarray): The current video frame (BGR format).
        keypoints (np.ndarray | None): Detected court keypoints or None.
        all_players (dict): Dictionary mapping player IDs to bounding boxes.
        idx (int): Frame index in the original video.
        minimap_scale (int): Scale for drawing the minimap.
        minimap_margin (int): Margin around the minimap.
        extra_m (float): Extra margin in meters for the minimap.
        player_tracker (PlayerTracker): Instance used to draw bounding boxes.

    Returns:
        np.ndarray: Frame with minimap, player bounding boxes, and projected player dots drawn.
    """
    # Compute homography matrix if we have at least 4 court keypoints
    H = None
    if keypoints is not None and len(keypoints) >= 4:
        pts_img = keypoints[[0, 1, 2, 3]]
        pts_m = np.array([
            [0.00,  0.00],
            [10.97, 0.00],
            [ 0.00, 23.77],
            [10.97, 23.77]
        ], dtype=np.float32)
        H, _ = cv2.findHomography(pts_img, pts_m)

    # Draw court minimap
    frame = draw_fixed_minimap(frame, scale=minimap_scale, margin=minimap_margin, extra_m=extra_m)

    # Select top two players
    player_dict = get_top_two_players(all_players, idx, H)

    # Overlay frame number text
    frame = draw_frame_number(frame, idx)

    # Draw player bounding boxes
    frame = player_tracker.draw_bbox_on_frame(frame, player_dict)

    # If homography is valid, project player coordinates and draw on minimap
    if H is not None:
        coords = [project_player_to_court(bbox, H) for bbox in player_dict.values()]
        frame = draw_players_on_minimap(frame, coords, scale=minimap_scale, margin=minimap_margin, extra_m=extra_m)

    return frame
