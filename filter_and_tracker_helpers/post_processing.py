# import joblib
from game_tracker import Game 
from ball_analysis import *
from player_analysis import project_player_to_court
import os, cv2, time
import numpy as np

def post_process_ball(ball_tracker):
    """
    Post-process the detected ball positions:
    1. Validate detections using context-aware logic.
    2. Mark valid points as 'real', others remain None.
    3. Interpolate missing points with increasing max_gap.
    4. Print labeled results for all frames.

    Args:
        ball_tracker: an instance of BallTracker with .detections and .interpolated_flags
    """

    # Validate using context-aware method
    validated = ball_tracker.validate_detections_with_context()
    valid_count = sum(1 for x, y in validated if x is not None)
    print(f"Valid after validation: {valid_count} / {len(validated)}")

    # Store validated points
    ball_tracker.detections = validated
    ball_tracker.interpolated_flags = [
        'real' if x is not None else None for x, y in validated
    ]

    # Interpolate missing points in two passes
    ball_tracker.interpolate_missing_detections(max_gap=25)
    ball_tracker.interpolate_missing_detections(max_gap=50)

    # Uncomment to save detections to file
    """
    ball_detections_dict = {
        i: pt for i, pt in enumerate(ball_tracker.detections)
        if pt[0] is not None and pt[1] is not None
    }

    joblib.dump(ball_detections_dict, "ball_detections.joblib")
    print("Saved to ball_detections.joblib")
    """

    # Print results
    print("\nFinal Ball Positions with Labels:")
    for idx, (x, y) in enumerate(ball_tracker.detections):
        if x is not None:
            label = ball_tracker.interpolated_flags[idx] if idx < len(ball_tracker.interpolated_flags) else None
            label_str = (label or "Unknown").title()
            marker = "Real" if label == "real" else "Interpolated" if label == "interpolated" else "Unknown"
            print(f"Frame {idx}: {marker} Ball at ({int(x)}, {int(y)})")
        else:
            print(f"Frame {idx}: Ball not detected")

def winner_xy_from_top2(frame_idx: int, side: str, top_two_players_dicts: dict[int, dict[int, tuple[float, float, float, float]]],
                         court_tracker) -> tuple[float, float] | None:
    """
    Projects the bottom-center of each player's bounding box to real-world court coordinates,
    and returns the (x_m, y_m) position of the player on the specified side ("near" or "far").
    This is a callable function.

    Args:
        frame_idx (int): The current frame index to extract player detections from.
        side (str): "near" or "far", indicating which side of the court to find the player on.
        top_two_players_dicts (dict): Dictionary mapping frame index to two player bounding boxes.
            Format: {frame_idx: {player_id: (x1, y1, x2, y2), ...}}
        court_tracker: An object that provides homography matrices via `get_H_for_frame(frame_idx)`.

    Returns:
        tuple[float, float] | None: The projected court coordinates (x_m, y_m) of the selected player.
        Returns None if players are missing, projection fails, or homography is unavailable.
    """
    # Get the top 2 players for this frame
    players = top_two_players_dicts.get(frame_idx)
    if not players:
        return None

    # Get the homography matrix for this frame
    H = court_tracker.get_H_for_frame(frame_idx)
    if H is None:
        return None

    projected_players = []

    # Project each player's bottom-center to court coordinates
    for player_id, (x1, y1, x2, y2) in players.items():
        bbox = [x1, y1, x2, y2]
        court_coords = project_player_to_court(bbox, H)

        if court_coords is None or np.any(np.isnan(court_coords)):
            continue

        x_m, y_m = float(court_coords[0]), float(court_coords[1])
        projected_players.append((player_id, (x_m, y_m)))

    if not projected_players:
        return None

    # Select the player closest to the "near" or "far" side based on y_m
    if side == "near":
        return max(projected_players, key=lambda t: t[1][1])[1]
    else:
        return min(projected_players, key=lambda t: t[1][1])[1]


def player_xy_from_top2(frame_idx: int, side: str, top_two_players_dicts: dict[int, dict[int, tuple[float, float, float, float]]], court_tracker) -> tuple[float, float] | None:
    """
    Projects the bottom-center of each player's bounding box to real-world court coordinates,
    and returns the (x_m, y_m) position of the player on the specified side ("near" or "far").
    This is a callable function.

    Args:
        frame_idx (int): The frame index to process.
        side (str): Which side of the court to select the player from ("near" or "far").
        top_two_players_dicts (dict): A dictionary mapping frame index to two player bounding boxes.
            Format: {frame_idx: {player_id: (x1, y1, x2, y2), ...}}
        court_tracker: Object providing homography matrices via `get_H_for_frame(frame_idx)`.

    Returns:
        tuple[float, float] | None: Projected (x_m, y_m) coordinates of the player on the specified side.
        Returns None if projection or data is invalid.
    """
    # Get the two detected players for this frame
    players = top_two_players_dicts.get(frame_idx)
    if not players:
        return None

    # Get the homography matrix used to project to court space
    H = court_tracker.get_H_for_frame(frame_idx)
    if H is None:
        return None

    projected_players = []

    # Loop through each player and project their position
    for player_id, (x1, y1, x2, y2) in players.items():
        bbox = [x1, y1, x2, y2]
        court_coords = project_player_to_court(bbox, H)

        if court_coords is None or np.any(np.isnan(court_coords)):
            continue

        x_m, y_m = float(court_coords[0]), float(court_coords[1])
        projected_players.append((player_id, (x_m, y_m)))

    if not projected_players:
        return None

    # Select based on y_m: max for "near", min for "far"
    return max(projected_players, key=lambda t: t[1][1])[1] if side == "near" else min(projected_players, key=lambda t: t[1][1])[1]


def run_game_tracking(
        ball_tracker,
        court_tracker,
        bounce_classifier_path,
        top_two_players_dicts,
        output_path,
        fps,
        width,
        height,
    ):
    """
    Run the full tennis match tracking/analysis pipeline on an already-processed video.

    Parameters:
        ball_tracker: Your BallTracker instance (already run on video).
        court_tracker: CourtTracker instance.
        bounce_classifier_path: Path to trained bounce classifier model.
        top_two_players_dicts: Dict mapping frame_idx to player bboxes.
        output_path: Path to the raw tracked video file (for annotation overlay).
        fps: Video FPS.
        width, height: Video dimensions.

    Returns:
        (zone_stats, return_stats): Two dictionaries with per-side/per-zone stats.
    """
    # -- Annotate video (unchanged) --
    annotated_path = output_path.replace(".mp4", "_annotated.mp4")
    cap2 = cv2.VideoCapture(output_path)
    out2 = cv2.VideoWriter(annotated_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))
    # If draw_ball is your existing annotation function:
    game_tracker = Game(ball_tracker, bounce_classifier_path)
    draw_ball(cap2, out2, ball_tracker, court_tracker, game_tracker)

    game_tracker.determine_initial_hitter()
    game_tracker.detect_hits_and_bounces(top_two_players=top_two_players_dicts)
    game_tracker.filter_bounces_by_hits()
    game_tracker.detect_bounces()
    game_tracker.score_single_pass(court_tracker)

    print(f"HITS: {game_tracker.hits}")

    # --- Zone stats
    zone_stats = game_tracker.compute_scoring_zone_stats(
        court_tracker,
        lambda frame_idx, side: winner_xy_from_top2(frame_idx, side, top_two_players_dicts, court_tracker),
        debug=False
    )

    return_stats = game_tracker.compute_return_rate_by_zone(
        get_player_xy_fn=lambda frame_idx, side: player_xy_from_top2(frame_idx, side, top_two_players_dicts, court_tracker),
        court_tracker=court_tracker
    )

    out2.release()
    cap2.release()

    return zone_stats, return_stats, annotated_path, game_tracker.hits