import cv2
import math
import numpy as np
import torch
from scipy.stats import median_abs_deviation

def combine_three_frames(frame1, frame2, frame3, width, height):
    """
    Resize and combine three image frames into a single 3-channel tensor.

    Parameters:
        frame1 (np.ndarray): First frame as a NumPy array (e.g., the current frame).
        frame2 (np.ndarray): Second frame as a NumPy array (e.g., previous frame).
        frame3 (np.ndarray): Third frame as a NumPy array (e.g., next frame).
        width (int): The target width to resize each frame to.
        height (int): The target height to resize each frame to.

    Returns:
        np.ndarray: A NumPy array of shape (3 * C, H, W) where C is the number of channels (typically 3).
                    The result is obtained by stacking the three resized frames along the channel dimension.
    """
    resized_frame1 = cv2.resize(frame1, (width, height)).astype(np.float32)
    resized_frame2 = cv2.resize(frame2, (width, height)).astype(np.float32)
    resized_frame3 = cv2.resize(frame3, (width, height)).astype(np.float32)

    stacked_frames = np.concatenate((resized_frame1, resized_frame2, resized_frame3), axis=2)
    stacked_frames = np.rollaxis(stacked_frames, 2, 0)

    return np.array(stacked_frames)

class BallTracker:
    def __init__(self, model_path):
        """
        Initialize the BallTracker object.

        Parameters:
            model_path (str): Path to the TorchScript model file (.pt or .pth).

        Attributes:
            device (torch.device): The computation device ('cuda' if available, else 'cpu').
            model (torch.jit.ScriptModule): Loaded PyTorch model set to evaluation mode.
            before_last_frame (np.ndarray | None): Frame at t-2.
            last_frame (np.ndarray | None): Frame at t-1.
            current_frame (np.ndarray | None): Current frame (t).
            model_input_width (int): Width to resize frames before model input (default 640).
            model_input_height (int): Height to resize frames before model input (default 360).
            video_width (int | None): Actual width of the input video.
            video_height (int | None): Actual height of the input video.
            detections (list): List of detected ball coordinates per frame as (x, y) or (None, None).
            interpolated_flags (list): Flags indicating whether each detection was interpolated.

            roi_miss_count (int): Number of consecutive frames with no valid ball detection.
            roi_initial_size (int): Initial size of the region of interest for heatmap search.
            roi_max_size (int): Maximum size to grow the region of interest.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).to(self.device).eval()

        self.before_last_frame = None
        self.last_frame = None
        self.current_frame = None

        self.model_input_width = 640
        self.model_input_height = 360

        self.video_width = None
        self.video_height = None

        self.detections = []
        self.interpolated_flags = []

        # Region of Interest (ROI)
        self.roi_miss_count = 0
        self.roi_initial_size = 60
        self.roi_max_size = 160

    def detect_ball_from_frame(self, frame):
        """
        Update the ball tracker with a new frame.

        This method:
        - Stores the new frame and shifts older frames
        - Prepares a 3-frame input tensor for the model from the function combine_three_frames
        - Runs the model to produce a heatmap to detect the ball (not to be confused with the heatmap that is produced as a statistic)
        - Locates the ball using the heatmap and recent position
        - Appends the result to self.detections

        Parameters:
            frame (np.ndarray): The current video frame (BGR image)

        Returns:
            tuple:
                (x, y): Detected ball coordinates in original frame scale,
                      or (None, None) if no valid detection
        """

        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]

        self.before_last_frame = self.last_frame
        self.last_frame = self.current_frame
        self.current_frame = frame.copy()

        if self.before_last_frame is None or self.last_frame is None:
            self.detections.append((None, None))
            return (None, None)

        input_tensor = combine_three_frames(
            self.current_frame,
            self.last_frame,
            self.before_last_frame,
            self.model_input_width,
            self.model_input_height
        )
        input_tensor = (torch.from_numpy(input_tensor) / 255).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            heatmap = probs[0, 1].cpu().numpy()

        if heatmap.shape != (360, 640):
            heatmap = cv2.resize(heatmap, (640, 360))

        # Find the last valid detection for ROI guidance
        last_valid = next(
            (
                (
                    int(x * self.model_input_width / self.video_width),
                    int(y * self.model_input_height / self.video_height)
                )
                for x, y in reversed(self.detections)
                if x is not None
            ),
            None
        )

        roi_size = min(self.roi_initial_size + self.roi_miss_count * 20, self.roi_max_size)
        x, y = self.get_center_ball(
            heatmap,
            roi_center=last_valid if last_valid else None,
            roi_size=roi_size,
            fallback_to_full=True
        )

        if x is None or y is None:
            self.roi_miss_count += 1
            self.detections.append((None, None))
            return (None, None)
        else:
            self.roi_miss_count = 0
            x *= self.video_width / self.model_input_width
            y *= self.video_height / self.model_input_height # type: ignore
            self.detections.append((x, y))
            return (x, y)

    def interpolate_missing_detections(
            self,
            max_gap=25,
            max_angle_deg=120,
            weight_power=2,
            allow_near_straight=True,
            max_anchor_distance=300
        ):
        """
        Fill in missing ball detections by interpolating between nearby valid detections.

        This method scans through the detection list (`self.detections`) and replaces
        missing entries (None, None) with interpolated coordinates, based on valid
        detections before and after the gap. It can also reject interpolation if the
        distance between anchor points is too large or if the angle change is too sharp.

        Parameters:
            max_gap (int): Maximum number of frames to look back/forward for interpolation anchors.
            max_angle_deg (float): Maximum allowed angle change between anchors for interpolation.
            weight_power (float): Power used in distance-based weighting for interpolation.
            allow_near_straight (bool): Whether to allow interpolation for nearly straight paths.
            max_anchor_distance (float): Maximum allowed distance (in pixels) between interpolation anchors.

        Returns:
            list: Updated list of detections with interpolated values filled in where possible.
        """
        filled = self.detections.copy()
        flags = self.interpolated_flags.copy()

        def compute_angle(p1, p2, p3):
            """
            Compute the angle (in degrees) formed at point p2 by three points p1, p2, and p3.

            The angle is measured between the vector from p1 to p2 (v1) and the vector
            from p2 to p3 (v2). This is useful for checking trajectory direction changes.

            Parameters:
                p1 (tuple[float, float]): The first point (x1, y1).
                p2 (tuple[float, float]): The middle point where the angle is measured.
                p3 (tuple[float, float]): The third point (x3, y3).

            Returns:
                float: The angle in degrees between the two vectors, from 0° to 180°.
                      Returns 0 if either vector has zero length (overlapping points).

            Steps:
                1. Create vector v1 from p1 to p2, and vector v2 from p2 to p3.
                2. Compute the lengths (norms) of v1 and v2.
                3. If either vector length is zero, return 0 (cannot form an angle).
                4. Compute the cosine of the angle using the dot product formula:
                      cos(theta) = (v1 · v2) / (||v1|| * ||v2||)
                5. Clip the cosine value to [-1, 1] to avoid floating-point rounding errors.
                6. Convert the angle from radians to degrees and return it.
            """
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            return math.degrees(math.acos(cos_angle))

        # Loop through all detections to find and fill missing ones
        for i in range(len(filled)):
            if filled[i][0] is not None:
                continue

            prev_index = next(
                (j for j in range(i - 1, max(i - max_gap - 1, -1), -1)
                if filled[j][0] is not None),
                None
            )
            next_index = next(
                (j for j in range(i + 1, min(i + max_gap + 1, len(filled)))
                if filled[j][0] is not None),
                None
            )

            if prev_index is None or next_index is None:
                # print(f"Frame {i}: Cannot interpolate — missing anchors")
                continue

            prev = filled[prev_index]
            next_ = filled[next_index]

            # Reject interpolation if anchors are too far apart
            dist_between_anchors = math.hypot(next_[0] - prev[0], next_[1] - prev[1])
            if dist_between_anchors > max_anchor_distance:
                # print(f"Frame {i}: Anchor distance too large ({int(dist_between_anchors)}px) — skipping")
                continue

            # Check angle constraints
            allow_interpolation = True
            gap = next_index - prev_index
            adaptive_angle = min(max_angle_deg, 180 - 2 * gap)

            if prev_index > 0 and next_index < len(filled) - 1:
                prev_prev = filled[prev_index - 1] if filled[prev_index - 1][0] is not None else prev
                angle = compute_angle(prev_prev, prev, next_)

                # print(f"Frame {i}: Checking angle = {angle:.2f}° between [{prev_index}]→[{i}]→[{next_index}]")

                if abs(angle - 180.0) < 5.0 and allow_near_straight:
                    # print(f"Frame {i}: Accepting straight path (angle ≈ 180°)")
                    continue
                elif angle > adaptive_angle:
                    # print(f"Frame {i}: Angle too sharp ({angle:.2f}° > {adaptive_angle:.2f}°) — skipping")
                    allow_interpolation = False

            if not allow_interpolation:
                continue

            dist_prev = i - prev_index
            dist_next = next_index - i

            w_prev = 1 / (dist_prev ** weight_power)
            w_next = 1 / (dist_next ** weight_power)
            w_sum = w_prev + w_next

            interp_x = (w_prev * prev[0] + w_next * next_[0]) / w_sum
            interp_y = (w_prev * prev[1] + w_next * next_[1]) / w_sum
            filled[i] = (interp_x, interp_y)
            flags[i] = 'interpolated'

            # print(f"Frame {i}: Interpolated at ({int(interp_x)}, {int(interp_y)}) using [{prev_index}] and [{next_index}]")

        self.detections = filled
        self.interpolated_flags = flags
        return filled

    def draw_ball_trail(self, frame, mark_num=10, frame_num=None):
        """
        Draw the ball's trail on the given frame.

        This method overlays a trail of recent ball positions on the frame.
        The trail can either be drawn up to a specific frame index or for the
        most recent `mark_num` frames. Interpolated detections are shown in
        a different color.

        Parameters:
            frame (np.ndarray): The frame (BGR image) to draw on.
            mark_num (int): The number of recent points to include in the trail.
            frame_num (int | None): If given, the trail is drawn ending at this frame index.
                                    If None, the trail ends at the last available detection.

        Returns:
            np.ndarray: The frame with the ball trail overlay.
        """
        overlay = frame.copy()

        # Determine start and end frame indices for the trail
        if frame_num is not None:
            start = max(0, frame_num - mark_num + 1)
            end = frame_num + 1
        else:
            end = len(self.detections)
            start = max(0, end - mark_num)

        # Draw circles for each detection in the trail
        for i in range(start, end):
            x, y = self.detections[i]
            if x is None or y is None:
                continue

            label = self.interpolated_flags[i] if hasattr(self, 'interpolated_flags') else None
            is_interpolated = (label == 'interpolated')

            base_color = (0, 255, 0) if not is_interpolated else (0, 0, 255)

            alpha = (i - start) / (end - start + 1e-6)  # Fading effect factor
            radius = max(2, int(6 * (1 - alpha)))

            cv2.circle(overlay, (int(x), int(y)), radius, base_color, -1)

        # Blend the overlay with the original frame
        return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    def validate_detections_with_context_broken(
            self,
            window_size=5,
            traj_std_thresh=5,
            jump_thresh_factor=5,
            reappear_thresh_factor=8
        ):
        """
        Validate and clean the list of ball detections using three context-based checks.

        The method runs in three stages:
          1. Trajectory Fit Deviation Check:
            - Fits a local linear regression to the ball's neighbors.
            - Removes points with a Y-position that deviates too far from the fit.

          2. Sudden Jump Spike Detection:
            - Removes points where the frame-to-frame jump (ΔX, ΔY) is abnormally large
              compared to the historical median and MAD (Median Absolute Deviation).

          3. Reappearance Sanity Check:
            - After missing frames, ensures the ball doesn't reappear far from its last position.

        Parameters:
            window_size (int): Number of frames on each side to include in local trajectory fitting.
            traj_std_thresh (float): Multiplier for MAD in trajectory deviation thresholding.
            jump_thresh_factor (float): Multiplier for MAD when detecting large jumps.
            reappear_thresh_factor (float): Multiplier for MAD when validating reappearances.

        Returns:
            list: A cleaned list of detections where invalid points are replaced with (None, None).
        """
        detections = self.detections.copy()
        cleaned = detections.copy()
        n = len(detections)

        # Get valid neighbor indices for trajectory fitting
        def get_valid_neighbors(i, w):
            return [
                j
                for j in range(max(0, i - w), min(n, i + w + 1))
                if j != i and detections[j][0] is not None
            ]

        # --- 1. Trajectory fit deviation check ---
        for i in range(n):
            if detections[i][0] is None:
                continue

            neighbors = get_valid_neighbors(i, window_size)
            if len(neighbors) < 3:
                continue

            coords = np.array([detections[j] for j in neighbors])
            xi, yi = zip(*coords)
            A = np.vstack([xi, np.ones(len(xi))]).T

            try:
                # Fit y = a*x + b
                a, b = np.linalg.lstsq(A, yi, rcond=None)[0]
                predicted_y = a * detections[i][0] + b
                error = abs(predicted_y - detections[i][1])

                residuals = np.abs(np.array(yi) - (a * np.array(xi) + b))
                mad = median_abs_deviation(residuals)
                adaptive_thresh = traj_std_thresh * mad if mad > 0 else 50

                if error > adaptive_thresh:
                    print(f"Frame {i}: Y deviation from trajectory fit = {error:.1f} > {adaptive_thresh:.1f}")
                    cleaned[i] = (None, None)
            except Exception:
                continue  # Fail safe if fitting fails

        # --- 2. Sudden jump spike detection ---
        deltas_x, deltas_y = [], []

        # Build lists of frame-to-frame distances
        for i in range(1, n):
            if cleaned[i][0] is not None and cleaned[i - 1][0] is not None:
                dx = abs(cleaned[i][0] - cleaned[i - 1][0])
                dy = abs(cleaned[i][1] - cleaned[i - 1][1])
                deltas_x.append(dx)
                deltas_y.append(dy)

        # Compute adaptive thresholds using median + MAD
        median_dx = np.median(deltas_x) if deltas_x else 0
        mad_dx = median_abs_deviation(deltas_x) if deltas_x else 0
        jump_thresh_x = median_dx + jump_thresh_factor * mad_dx if mad_dx > 0 else 80

        median_dy = np.median(deltas_y) if deltas_y else 0
        mad_dy = median_abs_deviation(deltas_y) if deltas_y else 0
        jump_thresh_y = median_dy + jump_thresh_factor * mad_dy if mad_dy > 0 else 120

        # Remove points that jump too far from previous frame
        for i in range(1, n):
            if cleaned[i][0] is None or cleaned[i - 1][0] is None:
                continue

            dx = abs(cleaned[i][0] - cleaned[i - 1][0])
            dy = abs(cleaned[i][1] - cleaned[i - 1][1])

            if dx > jump_thresh_x or dy > jump_thresh_y:
                print(f"Frame {i}: Sudden jump ΔX={dx}, ΔY={dy} — removed")
                cleaned[i] = (None, None)

        # --- 3. Reappearance check after missing frames ---
        last_valid = None
        dists = []

        for i in range(n):
            curr = cleaned[i]
            if curr[0] is not None:
                if last_valid is not None:
                    dist = np.linalg.norm(np.array(curr) - np.array(last_valid))
                    dists.append(dist)

                    median_dist = np.median(dists) if dists else 10
                    mad_dist = median_abs_deviation(dists) if dists else 0
                    adaptive_reappear_thresh = median_dist + reappear_thresh_factor * mad_dist if mad_dist > 0 else 150

                    if dist > adaptive_reappear_thresh:
                        print(f"Frame {i}: Reappeared with huge jump {dist:.1f} > {adaptive_reappear_thresh:.1f} — removed")
                        cleaned[i] = (None, None)
                        continue  # Skip updating last_valid

                last_valid = curr

        return cleaned

    def validate_detections_with_context(self, max_deviation=60):
        """
        Use both backward/forward geometry and local average checks
        to reject bad ball detections.
        """
        detections = self.detections.copy()
        cleaned = detections.copy()

        # --- PART 1: geometric projection check ---
        for i in range(len(detections)):
            if detections[i][0] is None:
                continue  # Already None

            # Find previous valid
            prev_idx = next((j for j in range(i - 1, -1, -1) if detections[j][0] is not None), None)
            # Find next valid
            next_idx = next((j for j in range(i + 1, len(detections)) if detections[j][0] is not None), None)

            if prev_idx is None or next_idx is None:
                continue  # Not enough context to validate

            # Form a line from previous to next
            prev = np.array(detections[prev_idx])
            next_ = np.array(detections[next_idx])
            curr = np.array(detections[i])

            line_vec = next_ - prev
            if np.linalg.norm(line_vec) == 0:
                continue  # Same point, skip

            # Project curr onto line
            point_vec = curr - prev
            proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
            proj_point = prev + proj * line_vec
            deviation = np.linalg.norm(curr - proj_point)

            if deviation > max_deviation:
                print(f"❌ Frame {i}: Too far from line ({deviation:.1f}) — removed")
                cleaned[i] = (None, None)

        # --- PART 2: sliding Y-deviation window ---
        for i in range(len(cleaned)):
            x, y = cleaned[i]
            if x is None or y is None:
                continue

            window = [
                cleaned[j][1]
                for j in range(max(0, i - 3), min(len(cleaned), i + 4))
                if j != i and cleaned[j][1] is not None
            ]

            if len(window) >= 3:
                avg_y = sum(window) / len(window)
                if abs(y - avg_y) > 300:
                    print(f"🚫 Frame {i}: Y={int(y)} too far from local avg Y={int(avg_y)} — removed")
                    cleaned[i] = (None, None)

        return cleaned

    def get_center_ball(self, heatmap, threshold=0.5, roi_center=None, roi_size=60, fallback_to_full=True):
        """
        Find the center position of the ball in a heatmap using image moments.

        This method first applies a binary threshold to the heatmap. It can search
        either inside a defined region of interest (ROI) or across the full image.

        Parameters:
            heatmap (np.ndarray): Model output heatmap (2D array of probabilities).
            threshold (float): Probability threshold for binary conversion.
            roi_center (tuple[int, int] | None): (x, y) center of the ROI in heatmap coordinates.
                                                If None, the search is over the full heatmap.
            roi_size (int): Half-width of the square ROI (only used if roi_center is given).
            fallback_to_full (bool): If True, will search the entire heatmap if ROI detection fails.

        Returns:
            tuple[int, int] | (None, None):
                - (x, y): Center of the ball in heatmap coordinates.
                - (None, None): If no valid detection is found.
        """
        heatmap = heatmap.astype(np.float32)
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # If an ROI is provided, search within it first
        if roi_center is not None:
            cx, cy = roi_center
            x1 = max(0, int(cx - roi_size))
            y1 = max(0, int(cy - roi_size))
            x2 = min(heatmap.shape[1], int(cx + roi_size))
            y2 = min(heatmap.shape[0], int(cy + roi_size))

            cropped = binary[y1:y2, x1:x2]
            M = cv2.moments(cropped)

            if M["m00"] > 0:
                dx = int(M["m10"] / M["m00"])
                dy = int(M["m01"] / M["m00"])
                return x1 + dx, y1 + dy

            # ROI failed — optionally fall back to full image search
            if fallback_to_full:
                M = cv2.moments(binary)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return cx, cy

        else:
            # No ROI given — search over the full heatmap
            M = cv2.moments(binary)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy

        return None, None

    def estimate_radius_from_frame(
            self,
            frame,
            center,
            roi_size=80,
            dp=1.2,
            min_dist=20,
            param1=50,
            param2=15,
            min_radius=3,
            max_radius=50
        ):
        """
        Estimate the radius of the ball in pixels using the Hough Circle Transform.

        This method extracts a square Region of Interest (ROI) around the given
        center coordinate, applies preprocessing, and then uses OpenCV's
        `HoughCircles` function to find the most likely ball circle.

        Parameters:
            frame (np.ndarray): The full BGR frame (H × W × 3) from the video.
            center (tuple[float, float] | None): Ball center (x, y) in original frame pixels.
                                                If None or invalid, returns None.
            roi_size (int): Half-width of the square Region of Interest (ROI) crop centered at `center`.
            dp (float): Inverse ratio of the accumulator resolution to the image resolution.
            min_dist (int): Minimum distance between detected circle centers.
            param1 (float): Higher threshold for Canny edge detection in HoughCircles.
            param2 (float): Accumulator threshold for circle detection.
            min_radius (int): Minimum circle radius to detect.
            max_radius (int): Maximum circle radius to detect.

        Returns:
            int | None:
                - Estimated ball radius in pixels.
                - None if detection fails.
        """
        if center is None or center[0] is None:
            return None

        h, w = frame.shape[:2]
        cx, cy = int(center[0]), int(center[1])

        # Define ROI boundaries
        x1 = max(0, cx - roi_size)
        y1 = max(0, cy - roi_size)
        x2 = min(w, cx + roi_size)
        y2 = min(h, cy + roi_size)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Preprocess ROI: convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        # Detect circles in the ROI
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is None:
            return None

        circles = np.round(circles[0]).astype(int)

        # Pick the detected circle closest to the ROI center
        best = min(
            circles,
            key=lambda c: (c[0] - (x2 - x1) // 2) ** 2 + (c[1] - (y2 - y1) // 2) ** 2
        )
        _, _, r = best
        return int(r)

    # Currently not in use because it's too wonky
    def validate_segments(self, min_segment_length=5, min_total_movement=40, max_flat_y_range=20):
        """
        Validate ball trajectory in continuous segments instead of frame-by-frame.

        This method groups consecutive detections into segments (continuous runs of valid points)
        and removes entire segments that do not meet minimum movement or length requirements.
        It helps eliminate false positives, such as when tracking a stationary object.

        Parameters:
            min_segment_length (int): Minimum number of consecutive valid detections to keep a segment.
            min_total_movement (float): Minimum total Euclidean movement in pixels for a segment to be valid.
            max_flat_y_range (float): Maximum allowed vertical range in pixels; segments flatter than this are removed.

        Returns:
            list: Cleaned detections list where removed segments are replaced with (None, None).
        """
        detections = self.detections.copy()
        cleaned = [(None, None)] * len(detections)

        segments = []
        current_segment = []

        # Step 1: Group consecutive valid detections into segments
        for i, (x, y) in enumerate(detections):
            if x is not None and y is not None:
                current_segment.append((i, (x, y)))
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        if current_segment:
            segments.append(current_segment)

        # Step 2: Filter out bad segments based on movement rules
        for seg in segments:
            frame_indices, coords = zip(*seg)
            xs, ys = zip(*coords)

            segment_length = len(seg)
            total_y_movement = abs(ys[-1] - ys[0])
            total_xy_distance = sum(
                np.linalg.norm(np.array(coords[i]) - np.array(coords[i - 1]))
                for i in range(1, segment_length)
            )

            # Apply validation rules
            if segment_length < min_segment_length:
                print(f"Segment {frame_indices[0]}–{frame_indices[-1]} too short — removed")
                continue

            if total_xy_distance < min_total_movement:
                print(f"Segment {frame_indices[0]}–{frame_indices[-1]} not moving enough (total {total_xy_distance:.1f}) — removed")
                continue

            if max(ys) - min(ys) < max_flat_y_range:
                print(f"Segment {frame_indices[0]}–{frame_indices[-1]} is too flat in Y — removed")
                continue

            # Passed all checks — keep the segment
            for idx, pt in seg:
                cleaned[idx] = pt

        return cleaned

