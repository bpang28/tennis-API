import cv2
import numpy as np
import math

# Not currently used
def compute_ball_speeds(detections: list[tuple[float, float] | None]) -> list[float]:
    """
    Compute frame-to-frame ball speeds from 2D positions.

    Parameters:
        detections (list of (x, y) or None): List of 2D ball positions per frame.

    Returns:
        list of float: Estimated speed (pixel distance) between each pair of frames.
                       If either point is missing (None), speed is set to 0.
    """
    speeds = []

    for i in range(1, len(detections)):
        p1 = detections[i - 1]
        p2 = detections[i]

        if any(x is None for x in p1) or any(x is None for x in p2): # type: ignore
            speeds.append(0.0)
            continue

        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        speeds.append(dist)

    return speeds

# Not currently used
def detect_hit_frames(speeds: list[float], threshold: float = 30.0) -> list[int]:
    """
    Detect hit frames by finding speed drop patterns.

    A "hit" is defined as a frame where the speed drops from above the threshold
    to below the threshold.

    Parameters:
        speeds (list of float): List of computed speeds between frames.
        threshold (float): Speed threshold used to detect a drop (default = 30.0).

    Returns:
        list of int: Indices of frames where a hit likely occurred.
    """
    hit_frames = []

    for i in range(1, len(speeds) - 1):
        if speeds[i - 1] > threshold and speeds[i] < threshold:
            hit_frames.append(i)

    return hit_frames

def project_ball_to_court(ball_xy: tuple[float, float] | None, H: np.ndarray | None) -> tuple[float, float] | None:
    """
    Project the ball's (x, y) image coordinates into real-world court coordinates using homography.
    This should only be used for bounces as otherwise it is inaccurate since homography assumes the ball is always on the ground.

    Parameters:
        ball_xy (tuple or None): The (x, y) position of the ball in image coordinates.
        H (np.ndarray or None): The 3x3 homography matrix used for projection.

    Returns:
        tuple of float or None: The (x, y) position in court coordinates, or None if invalid input.
    """
    # print(f"ball_xy {ball_xy} and ball_xy[0] {ball_xy[0]}")

    if H is None or ball_xy is None or ball_xy[0] is None:
        return None

    x, y = ball_xy
    pt = np.array([[[x, y]]], dtype=np.float32)
    court_coords = cv2.perspectiveTransform(pt, H)[0][0]
    return tuple(court_coords)

def draw_ball_on_minimap(frame: np.ndarray,
                         ball_court_coord: tuple[float, float] | None,
                         scale: int = 20,
                         margin: int = 10,
                         extra_m: float = 4.0,
                         color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Draw the ball on the top-right minimap of the frame based on court coordinates.

    Parameters:
        frame (np.ndarray): Input video frame to draw on.
        ball_court_coord (tuple or None): (x, y) position of the ball in court coordinates.
        scale (int): Scale factor from meters to pixels.
        margin (int): Margin from the frame edge to the minimap.
        extra_m (float): Additional vertical court space to display.
        color (tuple of int): BGR color used to draw the ball.

    Returns:
        np.ndarray: The frame with the ball drawn on the minimap.
    """
    if ball_court_coord is None:
        return frame

    h, w = frame.shape[:2]
    mini_w = int(11 * scale)
    x_offset = w - mini_w - margin
    y_offset = margin + int(extra_m / 2 * scale)

    x_m, y_m = ball_court_coord
    x = int(x_m * scale + x_offset)
    y = int(y_m * scale + y_offset)

    cv2.circle(frame, (x, y), 5, color, -1)
    return frame

def draw_ball(cap, out, ball_tracker, court_tracker, game_tracker=None):
    """
    Annotate video frames with ball trail, bounce markers, and minimap ball position.

    This function reads frames from the input `cap`, overlays ball trails,
    marks interpolated vs real ball positions, and optionally shows bounce location
    on the minimap using the court homography. Final frames are written to `out`.

    Parameters:
        cap (cv2.VideoCapture): Input video capture object.
        out (cv2.VideoWriter): Output video writer object.
        ball_tracker (BallTracker): Tracker with `.detections` and `draw_ball_trail()` method.
        court_tracker: Object providing homographies per frame (e.g., `get_H_for_frame()`).
        game_tracker: Object with `.bounce_results` indicating "in"/"out" bounces per frame.
        top_two_players: (Unused in this function but retained in signature for consistency.)

    Returns:
        None. Writes annotated frames to `out` and releases both `cap` and `out`.
    """
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # If reading the frame was not successful 
            break

        # Draw ball trail for the current frame
        frame = ball_tracker.draw_ball_trail(frame.copy(), frame_num=idx, mark_num=5)
        final_pos = ball_tracker.detections[idx]

        """
        # Draw bounce on minimap if bounce info is available for this frame
        if hasattr(game_tracker, "bounce_results") and idx in game_tracker.bounce_results:
            H = court_tracker.get_H_for_frame(idx)
            court_coord = project_ball_to_court(final_pos, H)
            if court_coord:
                is_in = game_tracker.bounce_results[idx] == "in"
                color = (0, 255, 0) if is_in else (0, 0, 255)
                frame = draw_ball_on_minimap(frame, court_coord, scale=20, margin=10, extra_m=4.0, color=color)
        """

        # Draw final ball position on frame (color-coded by detection type)
        if final_pos[0] is not None:
            flag = None
            if hasattr(ball_tracker, 'interpolated_flags') and idx < len(ball_tracker.interpolated_flags):
                flag = ball_tracker.interpolated_flags[idx]

            if flag == 'interpolated':
                color = (0, 0, 255)
            elif flag == 'real':
                color = (0, 255, 0)
            else:
                color = (200, 200, 200)

            cv2.circle(frame, (int(final_pos[0]), int(final_pos[1])), 10, color, -1)

        # Optional text overlay (commented out)
        # if idx in hit_frames:
        #     cv2.putText(frame, "HIT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        # if idx in bounce_frames:
        #     cv2.putText(frame, "BOUNCE!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 165, 0), 3)

        out.write(frame)
        idx += 1

    cap.release()
    out.release()
