import cv2
import numpy as np
from game_utils import *
from ball_analysis import project_ball_to_court, draw_ball_on_minimap
import os

def draw_fixed_minimap_centered(frame, scale=10, extra_m=2):
    """
    Draws a white-line tennis court onto the given `frame`, centered and scaled.

    Parameters:
        frame (np.ndarray): The image to draw on (usually black background).
        scale (int): Number of pixels per meter. Default is 10.
        extra_m (float): Extra vertical margin in meters added to top and bottom. Default is 2.

    Returns:
        np.ndarray: The input frame with court lines drawn on it.
    """

    # Real-world court coordinates (in meters)
    court_m = np.array([
        [0.0, 0.0], [10.97, 0.0], [0.0, 23.77], [10.97, 23.77],    # outer boundaries
        [1.37, 0.0], [9.6, 0.0], [1.37, 23.77], [9.6, 23.77],      # singles sidelines
        [1.37, 5.48], [9.6, 5.48],                                 # service line front
        [1.37, 18.29], [9.6, 18.29],                               # service line back
        [5.485, 5.48], [5.485, 18.29]                              # center service line
    ], dtype=np.float32)

    # Frame dimensions
    h, w = frame.shape[:2]

    # Minimap size in pixels
    mini_w = int(10.97 * scale)
    mini_h = int((23.77 + extra_m) * scale)

    # Offsets to center the minimap on the frame
    off_x = (w - mini_w) // 2
    off_y = (h - mini_h) // 2

    # Convert court meters to pixel coordinates and shift to center
    court_px = court_m * scale
    court_px += np.array([off_x, off_y], dtype=np.float32)
    pts = court_px.astype(np.int32)

    # Drawing settings
    WHITE = (255, 255, 255)
    TH = 2

    # Outer doubles court boundary
    cv2.polylines(frame, [pts[[0, 1, 3, 2]]], isClosed=True, color=WHITE, thickness=TH)

    # Singles sidelines
    for idx in [4, 5]:
        p1 = tuple(pts[idx])
        p2 = tuple(pts[idx + 2])
        cv2.line(frame, p1, p2, WHITE, TH)

    # Net line (middle of the court)
    net_y = off_y + int((23.77 / 2) * scale)
    cv2.line(frame, (off_x, net_y), (off_x + mini_w, net_y), WHITE, TH)

    # Service lines and center service line
    for a, b in [(8, 9), (10, 11), (12, 13)]:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), WHITE, TH)

    # Center marks on baselines (each 0.10m long)
    cm = int(0.10 * scale)
    mid_x = off_x + mini_w // 2
    cv2.line(frame, (mid_x, off_y), (mid_x, off_y + cm), WHITE, TH)
    cv2.line(frame, (mid_x, off_y + mini_h), (mid_x, off_y + mini_h - cm), WHITE, TH)

    return frame

def draw_players_on_minimap_centered(frame, coords, scale=10, extra_m=2):
    """
    Draws red circles on the centered minimap to represent player positions.

    Parameters:
        frame (np.ndarray): The image to draw on (usually black background).
        coords (list[tuple[float, float]]): List of player coordinates in meters (x_m, y_m).
        scale (int): Number of pixels per meter. Default is 10.
        extra_m (float): Extra vertical margin in meters added to top and bottom. Default is 2.

    Returns:
        np.ndarray: The frame with red player dots drawn on the minimap.
    """
    # Frame dimensions
    h, w = frame.shape[:2]

    # Minimap size in pixels
    mini_w = int(11 * scale)
    mini_h = int((23.77 + extra_m) * scale)

    # Offsets to center the minimap
    off_x = (w - mini_w) // 2
    off_y = (h - mini_h) // 2

    # Draw a red dot for each player coordinate
    for x_m, y_m in coords:
        x = int(x_m * scale + off_x)
        y = int(y_m * scale + off_y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return frame

def render_player_heatmap_video(global_heatmap, coords_per_frame,
                                size=(720, 1280), fps=30,
                                out_path="/tmp/player_heatmap.mp4"):
    """
    Renders a minimap-style video with a heatmap that accumulates over time and overlays player positions.

    Parameters:
    - writer: cv2.VideoWriter object to write output frames to
    - global_heatmap: (H, W) float32 numpy array being progressively accumulated over time
    - coords_per_frame: list of list-of-(x_m, y_m) positions per frame
    - size: (height, width) of the video
    - dot_radius_px: radius of player dots
    - colormap: OpenCV colormap to apply to the heatmap
    - draw_players: whether to draw player dots
    """
    height, width = size

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    for i, coord in enumerate(coords_per_frame):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        blur = cv2.GaussianBlur(global_heatmap, (101, 101), 0)
        m = blur.max() or 1.0
        hm8 = (blur / m * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(hm8, cv2.COLORMAP_HOT)

        frame = cv2.addWeighted(frame, 0.0, heat_color, 1.0, 0)
        frame = draw_fixed_minimap_centered(frame)
        frame = draw_players_on_minimap_centered(frame, coord)

        writer.write(frame)

    writer.release()
    return out_path


class HeatmapManager:
    """
    Owns all heatmap buffers and rendering for the minimap.

    Parameters
    ----------
    ball_tracker : object
        Must expose `.detections` (list of (x,y) or None).
    get_last_hitter_fn : callable
        Zero-arg function returning 'near' or 'far' at any time (delegates to Game.last_hitter).
    """
    def __init__(self, ball_tracker, get_last_hitter_fn):
        self.ball_tracker = ball_tracker
        self.get_last_hitter = get_last_hitter_fn

        # Heatmap config + buffers (initialized in init_ball_heatmaps)
        self._hm_cfg = None
        self.ball_heat_near = None
        self.ball_heat_far = None
        self.ball_heat_near_svc = None
        self.ball_heat_far_svc = None

        # Optional: last rendered paths
        self._last_heatmap_video_path = None
        self._last_service_heatmap_video_path = None

    # ----------------------- setup + utils -----------------------

    def init_ball_heatmaps(self, frame_shape, scale=10, extra_m=2):
        """
        Initializes four heatmap buffers for visualizing bounce locations on the minimap.

        These heatmaps are aligned to the minimap canvas and scaled to real-world court dimensions.
        The heatmaps are used to track bounce distributions per side (near/far) and per region (entire court or service box).

        Args:
            frame_shape (tuple): Shape of the video frame as (height, width, channels).
            scale (int): Scaling factor to convert meters to pixels for minimap rendering.
            extra_m (float): Additional vertical buffer (in meters) added to court height for extra space.

        Side Effects:
            - Initializes and stores minimap configuration in self._hm_cfg.
            - Creates zero-initialized heatmaps:
                - self.ball_heat_near: all bounces credited to near-side player
                - self.ball_heat_far: all bounces credited to far-side player
                - self.ball_heat_near_svc: near-side bounces within service boxes
                - self.ball_heat_far_svc: far-side bounces within service boxes
        """
        h, w = frame_shape[:2]

        court_w_m = 10.97
        court_h_m = 23.77 + extra_m

        mini_w = int(court_w_m * scale)
        mini_h = int(court_h_m * scale)

        off_x = (w - mini_w) // 2
        off_y = (h - mini_h) // 2

        self._hm_cfg = {
            "scale": scale,
            "extra_m": extra_m,
            "off_x": off_x,
            "off_y": off_y,
            "w": w,
            "h": h,
        }

        # Allocate heatmap canvases
        self.ball_heat_near = np.zeros((h, w), dtype=np.float32)
        self.ball_heat_far = np.zeros((h, w), dtype=np.float32)
        self.ball_heat_near_svc = np.zeros((h, w), dtype=np.float32)
        self.ball_heat_far_svc = np.zeros((h, w), dtype=np.float32)


    def _meters_to_minimap_px(self, x_m, y_m):
        """
        Converts real-world court coordinates (in meters) to pixel coordinates on the minimap.

        Args:
            x_m (float): X position in meters (across court width).
            y_m (float): Y position in meters (along court length).

        Returns:
            tuple[int, int]: Pixel coordinates (x_px, y_px) aligned to minimap canvas.
        """
        scale = self._hm_cfg["scale"] # type: ignore
        offset_x = self._hm_cfg["off_x"] # type: ignore
        offset_y = self._hm_cfg["off_y"] # type: ignore

        x_px = int(x_m * scale + offset_x)
        y_px = int(y_m * scale + offset_y)

        return x_px, y_px


    # ----------------------- accumulation -----------------------

    def accumulate_bounce_heatmap_all(self, frame_idx, court_tracker, spot_radius_px=6, spot_intensity=1.0):
        """
        Add a heat spot to the heatmap for the current bounce frame, regardless of in/out status.

        This function is used to visualize bounce locations on the minimap, accumulating intensity
        at the bounce point. It credits the bounce to the last known hitter ("near" or "far").

        Args:
            frame_idx (int): Index of the frame where the bounce occurred.
            court_tracker: Instance responsible for court homography (provides H matrix).
            spot_radius_px (int): Radius of the heat circle in pixels.
            spot_intensity (float): Intensity value to add to the heatmap at the bounce location.

        Returns:
            None. Updates the corresponding heatmap in-place (self.ball_heat_near or self.ball_heat_far).
        """
        # Determine last hitter (either 'near' or 'far')
        hitter = self.get_last_hitter()
        if hitter not in ("near", "far"):
            return

        # Get ball coordinates in the current frame
        ball_xy = self.ball_tracker.detections[frame_idx]
        if ball_xy is None:
            return

        # Get homography matrix for this frame
        H = court_tracker.get_H_for_frame(frame_idx)
        if H is None:
            return

        # Project ball from image coordinates to court coordinates (in meters)
        court_coord = project_ball_to_court(ball_xy, H)
        if court_coord is None or self._hm_cfg is None:
            return

        # Convert real-world court coordinates to minimap pixel coordinates
        x_px, y_px = self._meters_to_minimap_px(*court_coord)

        # Make sure the coordinates are within frame bounds
        h = self._hm_cfg["h"]
        w = self._hm_cfg["w"]
        if not (0 <= x_px < w and 0 <= y_px < h):
            return

        # Choose the appropriate heatmap based on hitter side
        target = self.ball_heat_far if hitter == "far" else self.ball_heat_near

        # Draw a filled circle at the projected pixel location on the heatmap
        cv2.circle(target, (x_px, y_px), spot_radius_px, spot_intensity, thickness=-1)


    def accumulate_bounce_heatmap_in_service_boxes_svc(self, frame_idx, court_tracker,
                                                   line_margin_m=0.10, spot_radius_px=6, spot_intensity=1.0):
        """
        Add a heat spot to the heatmap only if the bounce lands inside the court and within a service box.
        The spot is added to the hitter-specific service box heatmap.

        This is used for visualizing serve return patterns or bounce distribution specifically within 
        legal service areas.

        Args:
            frame_idx (int): Index of the frame where the bounce is being checked.
            court_tracker: Tracker object that provides homography matrix per frame.
            line_margin_m (float): Margin (in meters) used to relax court boundary checks.
            spot_radius_px (int): Radius of the heat spot in pixels.
            spot_intensity (float): Intensity value added to the heatmap.

        Returns:
            None. Updates the appropriate internal heatmap (ball_heat_near_svc or ball_heat_far_svc).
        """

        # Determine last hitter ("near" or "far")
        hitter = self.get_last_hitter()
        if hitter not in ("near", "far"):
            return

        # Get ball position for the given frame
        ball_xy = self.ball_tracker.detections[frame_idx]
        if ball_xy is None:
            return

        # Get homography matrix for this frame
        H = court_tracker.get_H_for_frame(frame_idx)
        if H is None:
            return

        # Project ball position to court (x_m, y_m in meters)
        court_coord = project_ball_to_court(ball_xy, H)
        if court_coord is None or self._hm_cfg is None:
            return

        x_m, y_m = court_coord

        # Ensure bounce is inside extended court bounds
        in_x = -line_margin_m <= x_m <= 10.97 + line_margin_m
        in_y = -line_margin_m <= y_m <= 23.77 + line_margin_m
        if not (in_x and in_y):
            return

        # Check if bounce falls inside service box region (near or far half)
        NET_Y = 23.77 / 2.0
        SVC_NEAR = 5.48
        SVC_FAR = 18.29
        in_service_box = (SVC_NEAR <= y_m <= NET_Y) or (NET_Y <= y_m <= SVC_FAR)
        if not in_service_box:
            return

        # Convert court meters to minimap pixels
        x_px, y_px = self._meters_to_minimap_px(x_m, y_m)
        h = self._hm_cfg["h"]
        w = self._hm_cfg["w"]

        # Check if the spot is within frame bounds
        if not (0 <= x_px < w and 0 <= y_px < h):
            return

        # Select appropriate service box heatmap for the hitter
        target = self.ball_heat_far_svc if hitter == "far" else self.ball_heat_near_svc

        # Add heat circle at the computed location
        cv2.circle(target, (x_px, y_px), spot_radius_px, spot_intensity, thickness=-1)


    # ----------------------- overlays -----------------------

    def draw_ball_heatmaps_on_minimap(self, canvas_bgr, alpha_near=0.6, alpha_far=0.6,
                                  col_near=(0, 255, 255), col_far=(255, 0, 255)):
        """
        Overlay near-side and far-side bounce heatmaps onto a minimap canvas.

        This function blends two separate heatmap layers—`ball_heat_near` and `ball_heat_far`—
        onto the given canvas image using different colors and transparency values.

        Args:
            canvas_bgr (np.ndarray): The base minimap canvas in BGR format (uint8).
            alpha_near (float): Opacity for blending the near-side heatmap.
            alpha_far (float): Opacity for blending the far-side heatmap.
            col_near (tuple): RGB color tuple used for the near-side heatmap (e.g., yellow).
            col_far (tuple): RGB color tuple used for the far-side heatmap (e.g., purple).

        Returns:
            np.ndarray: The updated minimap canvas with heatmaps overlaid (uint8).
        """

        # Ensure heatmaps are initialized
        if self.ball_heat_near is None or self.ball_heat_far is None:
            return canvas_bgr

        # Normalize a heatmap to range [0.0, 1.0]
        def norm(hm):
            if hm.max() <= 0:
                return np.zeros_like(hm, dtype=np.float32)
            return (hm / hm.max()).astype(np.float32)

        # Normalize both heatmaps
        hn = norm(self.ball_heat_near)  # near-side
        hf = norm(self.ball_heat_far)   # far-side

        # Convert heatmap grayscale to colored layer using given colors
        near_layer = np.dstack([
            hn * col_near[0], hn * col_near[1], hn * col_near[2]
        ]).astype(np.float32)

        far_layer = np.dstack([
            hf * col_far[0], hf * col_far[1], hf * col_far[2]
        ]).astype(np.float32)

        # Blend both layers onto the original canvas
        out = canvas_bgr.astype(np.float32)
        out = cv2.addWeighted(out, 1.0, near_layer, alpha_near, 0.0)
        out = cv2.addWeighted(out, 1.0,  far_layer,  alpha_far, 0.0)

        return np.clip(out, 0, 255).astype(np.uint8)

    def draw_service_heatmaps_on_minimap(self, canvas_bgr, alpha_near=0.6, alpha_far=0.6,
                                     col_near=(0, 200, 0), col_far=(200, 0, 0)):
        """
        Overlay service-box-only bounce heatmaps onto the given minimap canvas.

        This function renders two heatmaps (for near and far players) showing only
        the bounce locations that occurred inside the service boxes. Each heatmap is
        blended onto the canvas using the specified color and opacity.

        Args:
            canvas_bgr (np.ndarray): The base minimap image in BGR format (uint8).
            alpha_near (float): Opacity for blending the near-side service heatmap.
            alpha_far (float): Opacity for blending the far-side service heatmap.
            col_near (tuple): RGB color used for the near-side overlay (e.g., green).
            col_far (tuple): RGB color used for the far-side overlay (e.g., red).

        Returns:
            np.ndarray: The minimap image with overlaid service-box-only heatmaps (uint8).
        """

        # Ensure heatmaps are initialized
        if self.ball_heat_near_svc is None or self.ball_heat_far_svc is None:
            return canvas_bgr

        # Normalize heatmap to 0.0–1.0 range
        def norm(hm):
            if hm.max() <= 0:
                return np.zeros_like(hm, dtype=np.float32)
            return (hm / hm.max()).astype(np.float32)

        hn = norm(self.ball_heat_near_svc)
        hf = norm(self.ball_heat_far_svc)

        # Convert grayscale heatmap to color overlay
        near_layer = np.dstack([
            hn * col_near[0], hn * col_near[1], hn * col_near[2]
        ]).astype(np.float32)

        far_layer = np.dstack([
            hf * col_far[0], hf * col_far[1], hf * col_far[2]
        ]).astype(np.float32)

        # Blend both overlays onto the original canvas
        out = canvas_bgr.astype(np.float32)
        out = cv2.addWeighted(out, 1.0, near_layer, alpha_near, 0.0)
        out = cv2.addWeighted(out, 1.0,  far_layer, alpha_far,  0.0)

        return np.clip(out, 0, 255).astype(np.uint8)


    # ----------------------- video renderers (same logic) -----------------------

    def render_minimap_heatmap_video(self, out_path, court_tracker, ball_xy, size=(720, 1280), fps=30,
                                 scale=10, extra_m=2, line_margin_m=0.10, dot_radius_px=4, show_live_dot=True):
        """
        Render a standalone minimap video showing bounce heatmap accumulation over time.

        The video includes:
        - A fixed minimap in the center of the canvas
        - Bounce heatmap accumulation over frames (using all bounces or self.bounces_array if available)
        - Optionally, a live white dot indicating the ball's current location

        Args:
            out_path (str): Path to save the output video (e.g., "output/minimap.mp4").
            court_tracker: Object used to get homography matrix for each frame.
            ball_xy (list[tuple[float, float] | None]): Ball center positions per frame.
            size (tuple): Size of the output video in pixels (height, width).
            fps (int): Frames per second of the output video.
            scale (int): Scaling factor for converting meters to pixels on the minimap.
            extra_m (int): Extra vertical meters to add beyond the court for heatmap area.
            line_margin_m (float): Unused in this method, included for consistency.
            dot_radius_px (int): Radius of the white dot for the live ball position.
            show_live_dot (bool): Whether to overlay the live ball dot per frame.

        Returns:
            str: Path to the saved video file.
        """

        height, width = size

        # Ensure output path has .mp4 extension
        root, ext = os.path.splitext(out_path)
        if not ext:
            out_path = root + ".mp4"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not vw.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

        total_frames = len(ball_xy)

        # Initialize heatmaps
        self.init_ball_heatmaps(frame_shape=(height, width, 3), scale=scale, extra_m=extra_m)

        # Draw base canvas with court lines
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base = draw_fixed_minimap_centered(base, scale=scale, extra_m=extra_m)

        # Use precomputed bounce array if available
        bounce_set = set(getattr(self, "bounces_array", [])) if hasattr(self, "bounces_array") else set()

        for i in range(total_frames):
            frame = base.copy()

            # Accumulate heatmap if bounce detected at this frame
            if i in bounce_set:
                self.accumulate_bounce_heatmap_all(i, court_tracker)

            # Optionally show the live ball dot
            if show_live_dot:
                pt = ball_xy[i]
                if pt is not None:
                    H = court_tracker.get_H_for_frame(i)
                    if H is not None:
                        court_coord = project_ball_to_court(pt, H)
                        if court_coord is not None:
                            x_px, y_px = self._meters_to_minimap_px(*court_coord)
                            if 0 <= x_px < width and 0 <= y_px < height:
                                cv2.circle(frame, (x_px, y_px), dot_radius_px, (255, 255, 255), -1)

            # Overlay heatmap onto the frame
            frame = self.draw_ball_heatmaps_on_minimap(frame)

            # Write to video
            vw.write(frame)

        vw.release()
        self._last_heatmap_video_path = out_path
        return out_path


    def render_minimap_heatmap_video_service(self, out_path, court_tracker, ball_xy, size=(720, 1280), fps=30,
                                         scale=10, extra_m=2, line_margin_m=0.10, dot_radius_px=4, show_live_dot=True):
        """
        Render a minimap video showing only service-box bounce heatmaps over time.

        This video:
        - Accumulates bounce heatmaps ONLY for bounces inside service boxes
        - Optionally displays a live ball dot
        - Uses a fixed center court as the background
        - Shows heatmap buildup frame-by-frame (bounces appear only after they occur)

        Args:
            out_path (str): Path to save the output video (e.g., "output/service_heatmap.mp4").
            court_tracker: Object used to retrieve homography for each frame.
            ball_xy (list[tuple[float, float] | None]): Ball center coordinates for each frame.
            size (tuple): Dimensions of the video (height, width) in pixels.
            fps (int): Frames per second of the output video.
            scale (int): Scaling factor to convert meters to pixels.
            extra_m (int): Extra vertical meters beyond the court to show in the minimap.
            line_margin_m (float): Margin in meters for determining in-court bounce position.
            dot_radius_px (int): Radius of the white dot representing current ball position.
            show_live_dot (bool): Whether to draw a live dot for the ball in each frame.

        Returns:
            str: Path to the saved video file.
        """

        height, width = size

        # Ensure output filename ends with .mp4
        root, ext = os.path.splitext(out_path)
        if not ext:
            out_path = root + ".mp4"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not vw.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

        total_frames = len(ball_xy)

        # Initialize heatmap buffers
        self.init_ball_heatmaps(frame_shape=(height, width, 3), scale=scale, extra_m=extra_m)

        # Create the base minimap canvas
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base = draw_fixed_minimap_centered(base, scale=scale, extra_m=extra_m)

        # Use precomputed bounces if available
        bounce_set = set(getattr(self, "bounces_array", [])) if hasattr(self, "bounces_array") else set()

        for i in range(total_frames):
            frame = base.copy()

            # Accumulate bounce if it falls within service box
            if i in bounce_set:
                self.accumulate_bounce_heatmap_in_service_boxes_svc(
                    i, court_tracker,
                    line_margin_m=line_margin_m,
                    spot_radius_px=6,
                    spot_intensity=1.0
                )

            # Optionally draw live ball dot
            if show_live_dot:
                pt = ball_xy[i]
                if pt is not None:
                    H = court_tracker.get_H_for_frame(i)
                    if H is not None:
                        court_coord = project_ball_to_court(pt, H)
                        if court_coord is not None:
                            x_px, y_px = self._meters_to_minimap_px(*court_coord)
                            if 0 <= x_px < width and 0 <= y_px < height:
                                cv2.circle(frame, (x_px, y_px), dot_radius_px, (255, 255, 255), -1)

            # Overlay service-box heatmap
            frame = self.draw_service_heatmaps_on_minimap(frame)

            vw.write(frame)

        vw.release()
        self._last_service_heatmap_video_path = out_path
        return out_path
