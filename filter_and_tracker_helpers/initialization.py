import cv2
import numpy as np
from ball_tracker import BallTracker
from player_tracker import PlayerTracker
from court_tracker import CourtTracker

def initialize_trackers(ball_model_path, player_model_path, court_model_path):
    """
    Initialize tracking models for the tennis pipeline.

    Parameters:
        ball_model_path (str): Path to the trained ball detection model.
        player_model_path (str): Path to the trained player detection model.
        court_model_path (str): Path to the court keypoint detection model.

    Returns:
        tuple: (ball_tracker, player_tracker, court_tracker) — 
               Instances of BallTracker, PlayerTracker, and CourtTracker.
    """
    ball_tracker = BallTracker(ball_model_path)
    player_tracker = PlayerTracker(player_model_path)
    court_tracker = CourtTracker(court_model_path)

    return ball_tracker, player_tracker, court_tracker

def initialize_video_io(input_path: str, output_path: str, codec: str = 'mp4v') -> tuple[cv2.VideoCapture, cv2.VideoWriter, float, int, int]:
    """
    Initializes video input and output streams.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        codec (str): Four-character code for video codec (default is 'mp4v').

    Returns:
        tuple: (cv2.VideoCapture object, cv2.VideoWriter object, fps, width, height)
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer with same properties
    fourcc = cv2.VideoWriter.fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise ValueError(f"Failed to open output writer: {output_path}")

    return cap, out, fps, width, height