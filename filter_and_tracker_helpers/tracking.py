import cv2
import numpy as np

def detect_ball(frame, ball_tracker):
    """
    Detect the ball in the given frame using the provided ball tracker.

    Args:
        frame (np.ndarray): Input video frame.
        ball_tracker (BallTracker): Tracker instance with detect_ball_from_frame method.

    Returns:
        tuple or None: Detected ball position or None if not found.
    """
    return ball_tracker.detect_ball_from_frame(frame.copy())

def detect_players(frame, player_tracker):
    """
    Detect players in the given frame using the provided player tracker.

    Args:
        frame (np.ndarray): Input video frame.
        player_tracker (PlayerTracker): Tracker instance with detect_frame method.

    Returns:
        tuple: player_boxes (dict), scale_x (float), scale_y (float)
    """
    return player_tracker.detect_frame(frame)

def detect_court(frame, court_tracker):
    """
    Detect court keypoints and compute the image-to-court homography matrix.

    Args:
        frame (np.ndarray): BGR input frame from the video.
        court_tracker (CourtTracker): Instance capable of predicting court keypoints.

    Returns:
        tuple:
            - keypoints (np.ndarray or None): Detected court keypoints.
            - H (np.ndarray or None): 3×3 homography matrix from image to court space.
    """
    keypoints = court_tracker.predict(frame)

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

    return keypoints, H
