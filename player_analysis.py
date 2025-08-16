import cv2
import math
import numpy as np

# Global state tracking for court and players
_prev_court_coords = {}                     # Stores previously computed court coordinates
_player_history = {}                        # Stores recent court coordinates per player ID
_selected_player_ids = {"above": None, "below": None}  # Stores selected player IDs per side of the court
_last_known_bbox = {}                       # Stores the last known bounding box for each player ID
_last_seen_frame = {}                       # Stores the last frame index where each player was seen

# Unused 
def compute_distance_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculates the perpendicular distance from a point (px, py) to a line segment
    defined by endpoints (x1, y1) and (x2, y2).

    Parameters:
        px (float): X-coordinate of the point.
        py (float): Y-coordinate of the point.
        x1 (float): X-coordinate of the first endpoint of the line segment.
        y1 (float): Y-coordinate of the first endpoint of the line segment.
        x2 (float): X-coordinate of the second endpoint of the line segment.
        y2 (float): Y-coordinate of the second endpoint of the line segment.

    Returns:
        float: The shortest distance from the point to the line segment.
    """
    dx = x2 - x1
    dy = y2 - y1
    line_mag = dx * dx + dy * dy

    if line_mag == 0:
        # The line segment is a single point
        return math.hypot(px - x1, py - y1)

    # Project point onto the line segment using parameter t
    t = ((px - x1) * dx + (py - y1) * dy) / line_mag
    t = max(0, min(1, t))  # Clamp t to the [0, 1] segment range

    # Compute the projection point on the segment
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(px - proj_x, py - proj_y)

def project_player_to_court(bbox: list[float], H: np.ndarray) -> np.ndarray:
    """
    Projects a player's bounding box (bbox) onto real-world court coordinates
    using a given homography matrix. More specifically the bottom center of the bounding box. 

    Parameters:
        bbox (list[float]): A bounding box in image coordinates, defined as [x1, y1, x2, y2].
        H (np.ndarray): A 3x3 homography matrix used to transform image points to court coordinates.

    Returns:
        np.ndarray: A 2D point [x_m, y_m] representing the projected location in court space.
                    If the homography is None, returns [-1, -1].
    """
    if H is None:
        return np.array([-1, -1]) 

    # Compute the center-x and bottom-y of the bounding box
    cx = (bbox[0] + bbox[2]) / 2
    cy = bbox[3]  # Use the bottom edge (y2) of the bbox, not the center

    pt = np.array([[[cx, cy]]], dtype=np.float32)

    court_coords = cv2.perspectiveTransform(pt, H)[0][0]

    return court_coords

def draw_fixed_minimap(frame, scale=20, margin=10, extra_m=8.0):
    """
    Draws a fixed-size tennis court minimap overlay on the input frame on the top right. 

    Parameters:
        frame (np.ndarray): The image/frame on which to draw the minimap.
        scale (int): How many pixels represent 1 meter on the minimap.
        margin (int): Distance from frame edges to minimap.
        extra_m (float): Extra vertical meters beyond court boundaries.

    Returns:
        np.ndarray: Frame with the minimap overlay drawn.
    """
    court_m = np.array([
        [0, 0], [10.97, 0], [0, 23.77], [10.97, 23.77],
        [1.37, 0], [9.6, 0], [1.37, 23.77], [9.6, 23.77],
        [1.37, 5.48], [9.6, 5.48],
        [1.37, 18.29], [9.6, 18.29],
        [5.485, 5.48], [5.485, 18.29]
    ], dtype=np.float32)

    h, w = frame.shape[:2]
    mini_h = int((23.77 + extra_m) * scale)
    mini_w = int(11 * scale)
    x_offset = w - mini_w - margin
    y_offset = margin + int(extra_m / 2 * scale)

    court_px = court_m * scale
    court_px += np.array([x_offset, y_offset], dtype=np.float32)
    pts = court_px.astype(np.int32)

    # Draw outer court
    cv2.polylines(frame, [pts[[0, 1, 3, 2]]], isClosed=True, color=(0, 255, 0), thickness=1)

    # Draw singles sidelines
    for idx in [4, 5]:
        p1 = pts[idx]
        p2 = pts[idx + 2]
        cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 255), 1)

    # Draw net
    net_y = y_offset + int(23.77 / 2 * scale)
    cv2.line(frame, (x_offset, net_y), (x_offset + mini_w, net_y), (255, 255, 0), 1)

    # Draw service and center lines
    for a, b in [(8, 9), (10, 11), (12, 13)]:
        color = (255, 0, 255) if a < 10 else (0, 165, 255)
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 1)

    return frame

def draw_players_on_minimap(frame, player_court_coords, scale=20, margin=10, extra_m=4.0):
    """
    Draws red circles on the minimap to represent players' court positions.

    Parameters:
        frame (np.ndarray): The image/frame on which to draw player positions.
        player_court_coords (list[tuple[float, float]]): List of (x_m, y_m) coordinates for players in meters.
        scale (int): Pixels per meter for the minimap.
        margin (int): Distance from edge of frame to minimap.
        extra_m (float): Extra meters added beyond court length.

    Returns:
        np.ndarray: Frame with player locations drawn on the minimap.
    """
    h, w = frame.shape[:2]
    mini_w = int(11 * scale)
    x_offset = w - mini_w - margin
    y_offset = margin + int(extra_m / 2 * scale)

    for x_m, y_m in player_court_coords:
        x = int(x_m * scale + x_offset)
        y = int(y_m * scale + y_offset)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    return frame

# Unused
def compute_player_baseline_distances(all_players, H):
    """
    Compute the distance from each player to the nearest baseline on the court.

    Parameters:
        all_players (dict[int, list[float]]): A dictionary mapping player ID (pid) to their bounding box [x1, y1, x2, y2].
        H (np.ndarray): Homography matrix used to project player position from image to court coordinates.

    Returns:
        tuple:
            - distances (dict[int, float]): Distance (in meters) from each player to the closest baseline.
            - player_court_coords (dict[int, np.ndarray]): Projected (x, y) court coordinates for each player.
    """
    distances = {}
    player_court_coords = {}

    for pid, bbox in all_players.items():
        coord = project_player_to_court(bbox, H)
        dist_to_near_baseline = abs(coord[1])
        dist_to_far_baseline = abs(coord[1] - 23.77)
        distances[pid] = min(dist_to_near_baseline, dist_to_far_baseline)
        player_court_coords[pid] = coord

    return distances, player_court_coords

def get_top_two_players(all_players, frame_idx, H, max_missed_frames=20):
    """
    Select the top two players on the court — one from each side (above and below the net).

    This function uses average speed and proximity to the baseline to score each player.
    It preserves ID consistency over time, even if a player temporarily disappears.

    Parameters:
        all_players (dict[int, list[float]]): Dictionary of player ID → bounding box [x1, y1, x2, y2].
        frame_idx (int): Current frame index.
        H (np.ndarray): Homography matrix to project image coords to court coords.
        max_missed_frames (int): Maximum number of frames a player can be unseen before being untracked.

    Returns:
        dict[int, list[float]]: Dictionary of the top 2 players' IDs → their latest known bounding boxes.
    """
    global _player_history, _selected_player_ids, _last_known_bbox, _last_seen_frame

    halfway_m = 23.77 / 2
    above = []  # candidates for the far side of the court
    below = []  # candidates for the near side

    for pid, bbox in all_players.items():
        if H is None:
            continue

        # Project player position onto court
        coord = project_player_to_court(bbox, H)
        x_m, y_m = coord

        # Update tracking history
        history = _player_history.setdefault(pid, [])
        history.append((x_m, y_m, frame_idx))
        if len(history) > 10:
            history = history[-10:]
        _player_history[pid] = history

        # Require enough history for speed calculation
        if len(history) < 2:
            continue

        # Compute average speed over history
        total_dist = 0.0
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            total_dist += math.hypot(dx, dy)

        time_delta = history[-1][2] - history[0][2]
        if time_delta == 0:
            continue

        avg_speed = total_dist / time_delta

        # Compute distance to nearest baseline
        dist1 = math.hypot(x_m - 5.485, y_m - 0.0)
        dist2 = math.hypot(x_m - 5.485, y_m - 23.77)
        dist_base = min(dist1, dist2)

        # Scoring formula
        score = avg_speed - 0.5 * dist_base

        # Categorize player based on court half
        if y_m < halfway_m:
            above.append((score, pid))
        else:
            below.append((score, pid))

        _last_known_bbox[pid] = bbox
        _last_seen_frame[pid] = frame_idx

    selected = {}

    # Select top player for each side
    for side, candidates in [("above", above), ("below", below)]:
        current_id = _selected_player_ids[side]

        # Determine if current ID should be replaced
        missing_too_long = (
            current_id not in _last_seen_frame or
            frame_idx - _last_seen_frame[current_id] > max_missed_frames
        )

        if current_id is None or missing_too_long:
            if candidates:
                best_pid = max(candidates)[1]  # pick player with highest score
                _selected_player_ids[side] = best_pid

        pid = _selected_player_ids[side]
        if pid in _last_known_bbox:
            selected[pid] = _last_known_bbox[pid]

    return selected

def draw_bbox_with_threshold(frame: np.ndarray, player_dict: dict[int, tuple], proximity_scale: float = 0.75, color_bbox: tuple = (0, 255, 0), color_thresh: tuple = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """
    Draws bounding boxes and threshold zones around players.

    This function overlays two rectangles for each player:
    1. A regular bounding box (green by default).
    2. A larger "proximity" box (red by default), which is based on the player's bbox width.

    Parameters:
        frame (np.ndarray): The original video frame.
        player_dict (dict[int, tuple]): Dictionary mapping player IDs to bounding boxes (x1, y1, x2, y2).
        proximity_scale (float): Fraction of bbox width used to expand the proximity box.
        color_bbox (tuple): BGR color for the normal bounding box.
        color_thresh (tuple): BGR color for the threshold/proximity box.
        thickness (int): Thickness of the rectangles.

    Returns:
        np.ndarray: The updated frame with rectangles drawn.
    """
    height, width = frame.shape[:2]

    for player_id, (x1, y1, x2, y2) in player_dict.items():
        # Draw the standard bounding box
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color_bbox,
            thickness
        )

        # Compute margin based on bbox width
        bbox_width = x2 - x1
        margin = bbox_width * proximity_scale

        # Define the expanded proximity box (bounded by image size)
        xt1 = max(0, int(x1 - margin))
        yt1 = max(0, int(y1 - margin))
        xt2 = min(width - 1, int(x2 + margin))
        yt2 = min(height - 1, int(y2 + margin))

        # Draw the proximity box
        cv2.rectangle(
            frame,
            (xt1, yt1),
            (xt2, yt2),
            color_thresh,
            thickness
        )

    return frame
