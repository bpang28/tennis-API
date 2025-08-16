import os
import cv2
import torch
import numpy as np
import traceback
from typing import Optional, Tuple, Dict

from bounce_classifier import BounceClassifier, BounceNetV3
from ball_analysis import compute_ball_speeds, detect_hit_frames, project_ball_to_court, draw_ball_on_minimap
from heatmap import *
from game_utils import _classify_zone_from_xy

class Game:
    """
    Manages ball hit/bounce detection and scoring.

    ball_tracker : object
        Provides `.detections`.
    proximity_thresh : float
    last_hitter : str | None
    hits : set[int]

    bounces : set[int]
    bounce_model : BounceClassifier
    bounces_array : list[int]
    bounce_results : dict[int,str]

    heatmap_mgr: HeatmapManager
    """
    def __init__(self, ball_tracker, bounce_classifier_path, proximity_thresh: float = 50.0):
        self.ball_tracker = ball_tracker
        self.proximity_thresh = proximity_thresh
        self.last_hitter = None
        self.hits = set()

        self.bounces = set()
        self.bounce_model = BounceClassifier(model_path=bounce_classifier_path, window_size=7)
        self.bounces_array = []
        self.bounce_results = {}

        self.heatmap_mgr = HeatmapManager(self.ball_tracker, lambda: self.last_hitter)

    def determine_initial_hitter(self) -> None:
        """
        Determines which player (near or far) hit the ball first using the first 5 valid y-coordinate differences.
        
        Updates:
            self.last_hitter: 'far' if dy is increasing, 'near' if dy is decreasing, or None if ambiguous.
        """
        detections = self.ball_tracker.detections
        y_values = []

        # Loop through consecutive detection pairs
        for i in range(1, len(detections)):
            if detections[i - 1] and detections[i]:
                y0 = detections[i - 1][1]
                y1 = detections[i][1]
                if y0 is not None and y1 is not None:
                    y_values.append(y1 - y0)
            if len(y_values) >= 5:
                break

        if len(y_values) < 3:
            print("⚠️ Not enough valid frames to determine hitter.")
            self.last_hitter = None
            return

        avg_dy = sum(y_values) / len(y_values)

        if avg_dy > 0:
            self.last_hitter = 'far'  # Ball moving downward → hit by far player
        elif avg_dy < 0:
            self.last_hitter = 'near'  # Ball moving upward → hit by near player
        else:
            self.last_hitter = None  # No movement or ambiguous

        print(f"Initial hitter inferred: {self.last_hitter}")

    # Manual version not used right now
    def detect_hits_and_bounces(self, top_two_players: dict[int, dict[int, tuple]], dy_thresh: float = 1.0,
                                 pre_window: int = 5, post_window: int = 5, bounce_window: int = 5,
                                 drop_factor: float = 0.5, min_spacing: int = 10, cluster_window: int = 15,
                                 proximity_scale: float = 0.75):
        """
        Manual hit and bounce detection for tennis.

        Performs both hit and bounce detection in a single loop using hand-crafted rules.

        - Hits are detected using a velocity slope change (dy before vs after).
        - Hits are clustered, and the best one is selected using proximity logic.
        - Bounces are detected by:
            1. Local maxima (in Y position)
            2. Slope-drop (sharp slope reduction in Y)

        Args:
            top_two_players: dict mapping frame index to player bounding boxes.
            dy_thresh: Threshold for detecting slope change in hit detection.
            pre_window: Number of frames before candidate hit to average.
            post_window: Number of frames after candidate hit to average.
            bounce_window: Window size for slope-drop bounce detection.
            drop_factor: Slope-drop threshold for bounce detection.
            min_spacing: Minimum spacing between hits or bounces.
            cluster_window: Max frames between hit candidates in a cluster.
            proximity_scale: Scale for validating hit proximity to players.

        Updates:
            self.hits: set of frame indices classified as hits.
            self.bounces: set of frame indices classified as bounces.
        """
        pts = self.ball_tracker.detections
        n = len(pts)
        if n < 2 * bounce_window + 1:
            self.hits = set()
            self.bounces = set()
            return [], []

        dys = [None] * n
        for i in range(1, n):
            y0 = pts[i - 1][1] if pts[i - 1] else None
            y1 = pts[i][1] if pts[i] else None
            if y0 is not None and y1 is not None:
                dys[i] = y1 - y0

        final_hits = []
        cluster = []  # [(frame, strength)]
        last_accepted_hit = -min_spacing

        def resolve_cluster(cluster):
            nonlocal last_accepted_hit
            if not cluster:
                return

            frames = [f for f, _ in cluster]
            strengths = {f: s for f, s in cluster}
            is_consecutive = all(frames[i+1] - frames[i] == 1 for i in range(len(frames) - 1))

            if is_consecutive:
                first_hit = frames[0]
                final_hits.append(first_hit)
                if first_hit - last_accepted_hit >= min_spacing:
                    self.last_hitter = 'near' if self.last_hitter == 'far' else 'far'
                    last_accepted_hit = first_hit
                return

            valid = []
            for hit, _ in cluster:
                x_b, y_b = None, None
                for offset in range(20):
                    for j in [hit - offset, hit + offset]:
                        if 0 <= j < n:
                            pos = pts[j]
                            if pos and pos[0] is not None:
                                x_b, y_b = pos
                                break
                    if x_b is not None:
                        break
                if x_b is None:
                    continue

                bboxes = top_two_players.get(hit, {})
                for (x1, y1, x2, y2) in bboxes.values():
                    w = x2 - x1
                    h = y2 - y1
                    if w > 2.5 * h:
                        valid.append(hit)
                        break
                    base = (w + 0.5 * h) / 2
                    margin = base * proximity_scale
                    if x1 - margin <= x_b <= x2 + margin and y1 - margin <= y_b <= y2 + margin:
                        valid.append(hit)
                        break

            if not valid:
                return

            best = max(valid, key=lambda h: strengths[h])
            final_hits.append(best)
            if best - last_accepted_hit >= min_spacing:
                self.last_hitter = 'near' if self.last_hitter == 'far' else 'far'
                last_accepted_hit = best

        last_bounce = -min_spacing
        bounce_candidates = []
        bounce_raw = []

        for i in range(n):
            # HIT detection
            pre_vals, pre_idxs = [], []
            j = i - 1
            while j >= 0 and len(pre_vals) < pre_window:
                if dys[j] is not None:
                    pre_vals.append(dys[j])
                    pre_idxs.append(j)
                j -= 1

            post_vals, post_idxs = [], []
            k = i + 1
            while k < n and len(post_vals) < post_window:
                if dys[k] is not None:
                    post_vals.append(dys[k])
                    post_idxs.append(k)
                k += 1

            if len(pre_vals) == pre_window and len(post_vals) == post_window:
                pre_avg = sum(pre_vals) / pre_window
                post_avg = sum(post_vals) / post_window
                down_up = pre_avg > dy_thresh and post_avg < -dy_thresh
                up_down = pre_avg < -dy_thresh and post_avg > dy_thresh
                if down_up or up_down:
                    hit_idx = (pre_idxs[0] + post_idxs[0]) // 2
                    strength = abs(pre_avg - post_avg)
                    if cluster and hit_idx - cluster[-1][0] > cluster_window:
                        resolve_cluster(cluster)
                        cluster = []
                    cluster.append((hit_idx, strength))
                    if len(cluster) == 1 and hit_idx - last_accepted_hit >= min_spacing:
                        self.last_hitter = 'near' if self.last_hitter == 'far' else 'far'
                        last_accepted_hit = hit_idx

            # Bounce detection: local max
            if pts[i] and pts[i][1] is not None:
                prev_j = next((j for j in range(i - 1, -1, -1)
                              if pts[j] and pts[j][1] is not None), None)
                next_k = next((k for k in range(i + 1, n)
                              if pts[k] and pts[k][1] is not None), None)
                if prev_j is not None and next_k is not None:
                    if pts[i][1] > pts[prev_j][1] and pts[i][1] > pts[next_k][1]:
                        bounce_candidates.append(i)

            # Bounce detection: slope-drop
            if self.last_hitter == 'far' and i >= bounce_window and i + bounce_window < n:
                p0, p1, p2 = pts[i - bounce_window], pts[i], pts[i + bounce_window]
                if all(p and p[1] is not None for p in (p0, p1, p2)):
                    dy1 = p1[1] - p0[1]
                    dy2 = p2[1] - p1[1]
                    if dy1 * dy2 >= 0:
                        dx1 = p1[0] - p0[0]
                        dx2 = p2[0] - p1[0]
                        if abs(dx1) > 1e-3 and abs(dx2) > 1e-3:
                            s1 = abs(dy1 / dx1)
                            s2 = abs(dy2 / dx2)
                            if s1 > 0 and s2 < s1 * drop_factor:
                                bounce_raw.append(i)

        if cluster:
            resolve_cluster(cluster)

        clusters = []
        for idx in sorted(bounce_raw):
            if not clusters or idx - clusters[-1][-1] > 1:
                clusters.append([idx])
            else:
                clusters[-1].append(idx)
        cluster_centers = [run[len(run) // 2] for run in clusters]
        all_candidates = sorted(set(bounce_candidates + cluster_centers))

        final_bounces = []
        for idx in all_candidates:
            if idx - last_bounce >= min_spacing:
                final_bounces.append(idx)
                last_bounce = idx

        self.hits = set(final_hits)
        self.bounces = set(final_bounces)

    def filter_bounces_by_hits(self) -> None:
        """
        Post-process bounce detection by matching bounces to hits.

        For each hit in self.hits:
            - Find the closest bounce BEFORE the hit, skipping ±1 frame.
            - Keep only matched bounces.

        Additionally:
            - Keep unmatched bounces that occur AFTER the last hit.

        Result:
            - Updates self.bounces to only contain filtered ones.
            - Stores hit → bounce mapping in self.hit_to_bounce.
        """
        if not self.hits or not self.bounces:
            self.bounces = set()
            self.hit_to_bounce = {}
            return

        sorted_hits = sorted(self.hits)
        sorted_bounces = sorted(self.bounces)

        filtered_bounces = set()
        hit_to_bounce = {}

        last_hit = sorted_hits[-1]

        for hit in sorted_hits:
            # Find nearest bounce before hit (skipping ±1)
            candidate = None
            for b in reversed(sorted_bounces):
                if b < hit and abs(b - hit) > 1:
                    candidate = b
                    break
            if candidate is not None:
                filtered_bounces.add(candidate)
                hit_to_bounce[hit] = candidate

        # Keep unmatched bounces that are after the last hit
        for b in sorted_bounces:
            if b > last_hit + 1:
                filtered_bounces.add(b)

        self.bounces = filtered_bounces
        self.hit_to_bounce = hit_to_bounce

    def detect_bounces(self, min_spacing=8):
        """
        Predict bounces using the trained model on ball trajectory from bounce classifier.

        Keeps only the strongest bounce in any min_spacing-sized window.
        Updates:
            - self.bounces_array: list of accepted bounce frame indices.
        """
        detections = self.ball_tracker.detections
        all_candidates = []

        # Gather all bounce candidates
        for i in range(self.bounce_model.window_size // 2, len(detections) - self.bounce_model.window_size // 2):
            if detections[i] is None:
                continue
            prob = self.bounce_model.predict(detections, center_idx=i)
            if prob >= self.bounce_model.threshold:
                all_candidates.append((i, prob))

        # Sort candidates by descending confidence
        all_candidates.sort(key=lambda x: x[1], reverse=True)

        accepted = []
        occupied = set()

        for i, prob in all_candidates:
            # Check if too close to already accepted bounce
            if any(abs(i - a) < min_spacing for a in accepted):
                continue
            accepted.append(i)

        self.bounces_array = sorted(accepted)
        print(f"Total bounces detected with MODEL (non-clustered): {self.bounces_array}")

    def detect_hit_inline(self, i, pts, dys, pre_window, post_window, dy_thresh,
                          cluster, cluster_window, min_spacing,
                          last_hitter, last_accepted_hit, final_hits):
        """
        Detects a ball "hit" at the current frame index using motion patterns in the trajectory.

        This function performs per-frame hit detection by analyzing a window of vertical
        velocity values (dy) both before and after the current frame. It looks for rapid
        direction changes in the ball's movement as a proxy for a racket hit, clustering
        nearby detections, and applying spacing constraints to avoid double-counting.

        Args:
            i (int): Current frame index to analyze.
            pts (list): List of (x, y) tuples for ball detections.
            dys (list): List of precomputed vertical speeds (dy) per frame.
            pre_window (int): Number of frames to look back for averaging dy.
            post_window (int): Number of frames to look ahead for averaging dy.
            dy_thresh (float): Minimum change in dy to consider as a hit.
            cluster (list): List of candidate hits in the current cluster [(frame, strength), ...].
            cluster_window (int): Max distance (frames) between candidates to keep in one cluster.
            min_spacing (int): Minimum frame distance required between accepted hits.
            last_hitter (str | None): Side ("near" or "far") of the last detected hit.
            last_accepted_hit (int): Frame index of the last accepted hit.
            final_hits (list): Output list to record accepted hit frame indices.

        Returns:
            tuple:
                - last_hitter (str | None): Updated hitter after this detection.
                - last_accepted_hit (int): Updated index after this detection.

        Side Effects:
            - Mutates self.hit_owner_map to record owner ('near'/'far') for each accepted hit.
            - Appends to final_hits (in-place) as hits are accepted.
            - Mutates the provided `cluster` in-place (should be maintained between calls).

        Hit Detection Logic:
            - Gathers dy values before and after frame i.
            - Computes average dy in pre and post windows.
            - If a significant direction flip (down→up or up→down) is found, records a candidate hit.
            - Clusters close candidates together.
            - On cluster break (gap too large), resolves the strongest candidate in cluster.
            - Accepts a hit only if min_spacing is satisfied, and flips last_hitter.

        This function is optimized for single-pass loop usage within your main tracking pipeline.
        """
        # ensure map exists
        if not hasattr(self, "hit_owner_map"):
            self.hit_owner_map = {}

        n = len(pts)

        # Gather pre values
        pre_vals, pre_idxs = [], []
        j = i - 1
        while j >= 0 and len(pre_vals) < pre_window:
            if dys[j] is not None:
                pre_vals.append(dys[j])
                pre_idxs.append(j)
            j -= 1

        # Gather post values
        post_vals, post_idxs = [], []
        k = i + 1
        while k < n and len(post_vals) < post_window:
            if dys[k] is not None:
                post_vals.append(dys[k])
                post_idxs.append(k)
            k += 1

        # Local helper to accept a hit and record owner
        def _accept_hit(hit_idx, last_hitter, last_accepted_hit):
            """
            Accept hit if spacing allows; flip hitter and record owner.
            Returns updated (last_hitter, last_accepted_hit) and whether we accepted.
            """
            if hit_idx - last_accepted_hit >= min_spacing:
                # hitter flips when a new hit is accepted
                new_hitter = 'near' if last_hitter == 'far' else 'far'
                last_accepted_hit = hit_idx
                final_hits.append(hit_idx)
                self.hit_owner_map[hit_idx] = new_hitter
                return new_hitter, last_accepted_hit, True
            return last_hitter, last_accepted_hit, False

        # Check for hit pattern
        if len(pre_vals) == pre_window and len(post_vals) == post_window:
            pre_avg = sum(pre_vals) / pre_window
            post_avg = sum(post_vals) / post_window
            down_up = pre_avg > dy_thresh and post_avg < -dy_thresh
            up_down = pre_avg < -dy_thresh and post_avg > dy_thresh

            if down_up or up_down:
                hit_idx = (pre_idxs[0] + post_idxs[0]) // 2
                strength = abs(pre_avg - post_avg)

                # Cluster break → resolve previous cluster by picking strongest
                if cluster and hit_idx - cluster[-1][0] > cluster_window:
                    best = max(cluster, key=lambda h: h[1])[0]
                    # accept cluster's best (if spacing ok)
                    last_hitter, last_accepted_hit, accepted = _accept_hit(best, last_hitter, last_accepted_hit)
                    cluster.clear()

                # Add to current cluster
                cluster.append((hit_idx, strength))

                # First in cluster → opportunistically accept immediately (spacing gated)
                if len(cluster) == 1:
                    last_hitter, last_accepted_hit, _ = _accept_hit(hit_idx, last_hitter, last_accepted_hit)

        return last_hitter, last_accepted_hit

    def evaluate_bounce_in_court_for_frame(self, frame_idx, court_tracker, line_margin_m=0.10):
        """
        Determines whether the ball bounce at a specific frame is inside or outside the court boundaries.

        Parameters:
            frame_idx (int): Index of the frame to evaluate.
            court_tracker: Object providing the court homography via get_H_for_frame(frame_idx).
            line_margin_m (float, optional): Margin in meters to tolerate around court boundaries for in/out decision (default is 0.10).

        Returns:
            str: 
                - "in" if the bounce is inside the court (with margin).
                - "out" if the bounce is outside the court (with margin).
                - "unknown" if the projection fails or the ball position is missing.

        Side Effects:
            Updates self.bounce_results[frame_idx] with the result ("in", "out", or "unknown").
        """
        pts = self.ball_tracker.detections
        ball_xy = pts[frame_idx]
        if ball_xy is None:
            self.bounce_results[frame_idx] = "unknown"
            return "unknown"
        H = court_tracker.get_H_for_frame(frame_idx)
        if H is None:
            self.bounce_results[frame_idx] = "unknown"
            return "unknown"
        court_coord = project_ball_to_court(ball_xy, H)
        if court_coord is None:
            self.bounce_results[frame_idx] = "unknown"
            return "unknown"

        x_m, y_m = court_coord
        in_x = -line_margin_m <= x_m <= 10.97 + line_margin_m
        in_y = -line_margin_m <= y_m <= 23.77 + line_margin_m
        result = "in" if in_x and in_y else "out"
        self.bounce_results[frame_idx] = result
        return result

    def compute_scoring_zone_stats(self, court_tracker, get_player_xy_fn, debug: bool = False):
        """
        Computes the return success rate (%) for each of the 4 scoring zones on both sides ('near' and 'far').

        For each hit in self.hits, determines the scoring zone of the hitter and whether the point was won by that side after their hit. 
        Tallies the number of successful returns (winning the point) out of the total number of hits in each zone.

        Parameters:
            get_player_xy_fn (Callable): 
                Function with signature get_player_xy_fn(frame_idx: int, side: str) -> tuple[float, float] | None.
                Returns the real-world court (x_m, y_m) for a player ('near' or 'far') at a given frame index, or None if not available.
            court_tracker: 
                Court tracking object, used to determine y-axis direction (required for zone classification).

        Returns:
            dict: Nested dictionary with structure:
                {
                    "near": {
                        "left-back": float | None,
                        "right-back": float | None,
                        "left-front": float | None,
                        "right-front": float | None,
                    },
                    "far": {
                        ...
                    }
                }
            Each value is the return success percentage (float, rounded to 1 decimal), or None if no data for that zone.
        
        Details:
            - self.hits: Set of frames where hits were detected.
            - self.point_events: List of point outcome events (must include last_hit_frame and winner_side).
            - self.hit_owner_map: Maps each hit frame to the player side ("near" or "far").
            - Uses _classify_zone_from_xy(x_m, y_m, y_inc_toward_near) to classify zones.
            - Handles empty zones by returning None for those keys.
        """
        if not getattr(self, "point_events", None):
            return {'near': {}, 'far': {}}

        # --- Determine court orientation (y-axis increasing toward 'near') ---
        y_inc_toward_near = True
        try:
            for (pf, win_side, evt, last_hit_f) in self.point_events:
                xy_near = get_player_xy_fn(last_hit_f, "near") if last_hit_f is not None else None
                xy_far = get_player_xy_fn(last_hit_f, "far") if last_hit_f is not None else None
                if xy_near and xy_far:
                    y_inc_toward_near = (xy_near[1] > xy_far[1])
                    break
        except Exception:
            pass

        if debug:
            print(f"[zones] y_inc_toward_near = {y_inc_toward_near}")

        # --- Initialize tallies ---
        zone_keys = ['left-back', 'right-back', 'left-front', 'right-front']
        tally = {side: {k: 0 for k in zone_keys} for side in ['near', 'far']}
        counts = {'near': 0, 'far': 0}

        # --- Count occurrences in each zone ---
        for point_frame, winner_side, event_type, winner_last_hit_frame in self.point_events:
            if winner_side not in ("near", "far") or winner_last_hit_frame is None:
                continue

            xy = get_player_xy_fn(winner_last_hit_frame, winner_side)
            if xy is None:
                continue

            x_m, y_m = xy
            zone = _classify_zone_from_xy(x_m, y_m, y_inc_toward_near)
            tally[winner_side][zone] += 1
            counts[winner_side] += 1

            if debug:
                print(f"[zones] Point @ frame {point_frame} | Winner={winner_side} | "
                      f"Last hit frame={winner_last_hit_frame} | XY=({x_m:.3f},{y_m:.3f}) | "
                      f"Event={event_type} → Zone={zone}")

        # --- Convert to percentage results ---
        result = {'near': {}, 'far': {}}
        for side in ['near', 'far']:
            total = counts[side]
            for zone in zone_keys:
                result[side][zone] = round(100.0 * tally[side][zone] / total, 1) if total > 0 else None

        return result

    def compute_return_rate_by_zone(self, get_player_xy_fn, court_tracker) -> dict:
        """
        Calculate return success rate (%) for each of the four scoring zones ("left-back", "right-back", "left-front", "right-front")
        on both 'near' and 'far' sides of the court, based on hit and point outcome data.

        Parameters:
            get_player_xy_fn (Callable[[int, str], tuple[float, float] | None]):
                Function that returns (x_m, y_m) court coordinates for the requested frame index and side ("near" or "far").
            court_tracker: 
                Court tracking object (used only to infer court orientation).

        Returns:
            dict: Nested dictionary in the form:
                {
                    "near": {"left-back": float | None, "right-back": float | None, "left-front": float | None, "right-front": float | None},
                    "far":  {"left-back": float | None, "right-back": float | None, "left-front": float | None, "right-front": float | None}
                }
            Each value is the percentage of hits from that zone which led to the player winning the point (float, rounded to 1 decimal), or None if no samples.

        Details:
            - Uses self.hits (frames where a hit occurred) and self.point_events (point results).
            - self.hit_owner_map maps each hit frame to the player side ("near" or "far").
            - _classify_zone_from_xy(x_m, y_m, y_inc_toward_near) is used for zone classification.
            - y_inc_toward_near is inferred by comparing the y-coordinates of "near" and "far" players.
            - Returns None for a zone if there are no hits in that zone.
        """
        zones = ["left-back", "right-back", "left-front", "right-front"]
        outcome = {
            "near": {z: {"total": 0, "won": 0} for z in zones},
            "far":  {z: {"total": 0, "won": 0} for z in zones},
        }

        # --- Determine y-axis orientation ---
        y_inc_toward_near = True
        for ev in self.point_events:
            if len(ev) >= 5:
                frame, side, _, hit_frame, _ = ev
                near_xy = get_player_xy_fn(hit_frame, "near")
                far_xy = get_player_xy_fn(hit_frame, "far")
                if near_xy and far_xy:
                    y_inc_toward_near = near_xy[1] > far_xy[1]
                    break

        # --- Tally hit outcomes ---
        hits_sorted = sorted(self.hits)
        for hit_frame in hits_sorted:
            side = self.hit_owner_map.get(hit_frame)
            if side not in ("near", "far"):
                continue

            xy = get_player_xy_fn(hit_frame, side)
            if xy is None:
                continue

            x_m, y_m = xy
            zone = _classify_zone_from_xy(x_m, y_m, y_inc_toward_near)
            if zone is None:
                continue

            outcome[side][zone]["total"] += 1

            # Check if player won the point after this hit
            for ev in self.point_events:
                if len(ev) >= 5:
                    point_frame, winner_side, _, last_hit_frame, _ = ev
                    if last_hit_frame == hit_frame and winner_side == side:
                        outcome[side][zone]["won"] += 1
                        break

        # --- Convert counts to percentages ---
        result = {"near": {}, "far": {}}
        for side in ("near", "far"):
            for zone in zones:
                stats = outcome[side][zone]
                if stats["total"] == 0:
                    result[side][zone] = None
                else:
                    result[side][zone] = round(100.0 * stats["won"] / stats["total"], 1)

        return result

    def score_single_pass(self, court_tracker, dy_thresh: float = 1.0, pre_window: int = 5, post_window: int = 5,
                          min_spacing: int = 10, cluster_window: int = 15, line_margin_m: float = 0.10,
                          heatmap_out_path: str = None): # type: ignore
        """
        Run a single-pass match scoring routine: detects hits, scores bounces, manages rally state,
        and triggers heatmap video rendering via HeatmapManager, all in a single O(N) pass.

        Parameters:
            court_tracker: CourtTracker instance (provides per-frame homography for court projection).
            dy_thresh (float, optional): Minimum vertical speed (Δy) threshold for hit detection. Default: 1.0.
            pre_window (int, optional): Number of frames to look back when checking for hit signature. Default: 5.
            post_window (int, optional): Number of frames to look ahead when checking for hit signature. Default: 5.
            min_spacing (int, optional): Minimum frame spacing allowed between accepted hits/bounces. Default: 10.
            cluster_window (int, optional): Max frame gap for clustering candidate hits. Default: 15.
            line_margin_m (float, optional): Margin (in meters) for in/out bounce calls (to tolerate court line noise). Default: 0.10.
            heatmap_out_path (str, optional): Output path for the heatmap video. If None, a default is used.

        Main Steps:
            - Precomputes dy (vertical) deltas and caches court homographies for efficiency.
            - Inline hit detection with cluster/spacing logic, using dy signatures.
            - Rally state management (prevents double-counting; only one point per rally).
            - Bounce scoring using precomputed or detected bounces and in/out logic.
            - Records point outcomes in self.point_events and updates self.scores, self.hits.
            - Calls HeatmapManager to render heatmap videos after scoring.

        Returns:
            None. Side effects:
                - Updates self.scores (dict), self.hits (set), self.point_events (list), self.last_hitter (str).
                - Triggers two heatmap videos via self.heatmap_mgr (all bounces & service-box only).

        Notes:
            - Expects that self.heatmap_mgr (HeatmapManager) is already attached to this instance.
            - Designed for efficiency: O(N) per frame; minimizes repeated court projections.
            - Robust to missing detections or missing homographies.
        """
        pts = self.ball_tracker.detections
        n = len(pts)
        if n < 2:
            print("No detections available.")
            return

        # Init state
        self.scores = {"near": 0, "far": 0}
        self.point_events = []
        self.bounce_results = getattr(self, "bounce_results", {})
        self.hit_owner_map = {}

        final_hits = []
        cluster = []
        last_accepted_hit = -min_spacing
        last_hitter = getattr(self, "last_hitter", None)

        # Precompute dy (vertical deltas)
        dys = [None] * n
        for i in range(1, n):
            p0, p1 = pts[i - 1], pts[i]
            if p0 and p1 and p0[1] is not None and p1[1] is not None:
                dys[i] = p1[1] - p0[1]

        # Cache homographies once (avoid repeated Python calls)
        H_cache = [court_tracker.get_H_for_frame(i) for i in range(n)]

        bounce_set = set(getattr(self, "bounces_array", []))  # can be empty

        # Ensure heatmap buffers exist (HeatmapManager owns them)
        self.heatmap_mgr.init_ball_heatmaps(frame_shape=(720, 1280, 3), scale=10, extra_m=2)

        NET_Y = 23.77 / 2.0
        def opponent(side: str) -> str: return "near" if side == "far" else "far"

        rally_open = True
        last_point_frame = -10**9
        expect_return_from = None
        double_bounce_armed = False

        # MAIN LOOP
        for i in range(n):
            prev_hitter = last_hitter

            # HIT DETECTION (unchanged)
            last_hitter, last_accepted_hit = self.detect_hit_inline(
                i, pts, dys, pre_window, post_window, dy_thresh,
                cluster, cluster_window, min_spacing,
                last_hitter, last_accepted_hit, final_hits
            )

            # reopen rally after new accepted hit that occurs after point was decided
            if not rally_open and last_accepted_hit > last_point_frame:
                rally_open = True
                expect_return_from = None
                double_bounce_armed = False

            # expected return cleared if the expected side actually hits next
            if prev_hitter != last_hitter and last_hitter in ("near", "far") and expect_return_from == last_hitter:
                expect_return_from = None
                double_bounce_armed = False

            # BOUNCE SCORING (only when rally is open) – unchanged decisions
            if rally_open and bounce_set and i in bounce_set:
                # in/out decision
                result = self.evaluate_bounce_in_court_for_frame(i, court_tracker, line_margin_m)

                if result == "out":
                    if last_hitter in ("near", "far"):
                        win = opponent(last_hitter)
                        self.scores[win] += 1
                        self.point_events.append((i, win, "bounce_out", last_accepted_hit))
                        rally_open = False
                        last_point_frame = i
                        expect_return_from = None
                        double_bounce_armed = False
                    continue

                if result != "in" or last_hitter not in ("near", "far"):
                    continue

                hitter_side = last_hitter
                opp_side = opponent(hitter_side)

                pt = pts[i]
                H = H_cache[i]
                court_coord = project_ball_to_court(pt, H) if (H is not None and pt is not None) else None
                if court_coord is None:
                    continue

                _, y_m = court_coord
                bounce_side = "near" if y_m > NET_Y else "far"

                if bounce_side == opp_side:
                    if double_bounce_armed:
                        self.scores[hitter_side] += 1
                        self.point_events.append((i, hitter_side, "double_bounce_winner", last_accepted_hit))
                        rally_open = False
                        last_point_frame = i
                        expect_return_from = None
                        double_bounce_armed = False
                    else:
                        expect_return_from = opp_side
                        double_bounce_armed = True
                else:
                    win = opp_side
                    self.scores[win] += 1
                    self.point_events.append((i, win, "same_side_bounce_after_hit", last_accepted_hit))
                    rally_open = False
                    last_point_frame = i
                    expect_return_from = None
                    double_bounce_armed = False

        # Finalize trailing cluster
        if cluster:
            best = max(cluster, key=lambda h: h[1])[0]
            if best - last_accepted_hit >= min_spacing:
                new_hitter = "near" if last_hitter == "far" else "far"
                last_accepted_hit = best
                final_hits.append(best)
                self.hit_owner_map[best] = new_hitter
                last_hitter = new_hitter
            else:
                final_hits.append(best)

        self.hits = set(final_hits)
        self.last_hitter = last_hitter

        # Render heatmap videos via HeatmapManager (same outputs as before)
        try:
            # hand the detections list explicitly to avoid manager reading internals again
            pts_local = self.ball_tracker.detections
            self.heatmap_mgr.bounces_array = getattr(self, "bounces_array", [])  # forward precomputed bounces if any

            self.heatmap_mgr.render_minimap_heatmap_video(
                out_path=heatmap_out_path or "/tmp/minimap_heatmap.mp4",
                court_tracker=court_tracker,
                ball_xy=pts_local,
                size=(720, 1280), fps=30,
                scale=10, extra_m=2, line_margin_m=line_margin_m
            )
            self.heatmap_mgr.render_minimap_heatmap_video_service(
                out_path="/tmp/minimap_heatmap_weak.mp4",
                court_tracker=court_tracker,
                ball_xy=pts_local,
                size=(720, 1280), fps=30,
                scale=10, extra_m=2, line_margin_m=line_margin_m
            )
        except Exception:
            traceback.print_exc()

        # Debug summary
        print("\n==== SCORE SINGLE PASS DEBUG ====")
        print(f"Total frames processed: {n}")
        print(f"Hits detected ({len(self.hits)}): {sorted(self.hits)}")
        print(f"Last hitter: {self.last_hitter}")
        print(f"Scores: {self.scores}")
        print(f"Point events ({len(self.point_events)}): {self.point_events}")
        print("=================================\n")



