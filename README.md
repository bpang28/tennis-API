RUNNING

Read both paragraphs first. 

To run this code please go to the following colab and run the cells:
https://colab.research.google.com/drive/1cXcSblwpHn-6Z2ZHQa6tXujZXp3T09vy#scrollTo=uPcyQw1dKxah 

Access to this tennis_main folder will also mean that you do not have to touch any of the code as 
long as the tennis_main folder is under /content/drive/MyDrive/
https://drive.google.com/drive/u/0/folders/1kq210j1m95j7i0X9gkzqY3eVftrL2q-Z 

===========================================================================================================

IMPORTANT VARIABLES

Ball Tracking:
- self.detections (list[tuple[float, float] | [None, None]])
    Stores ball center pixel coordinates per frame. None indicates a missing detection. Used for tracking, 
    bounce detection, and heatmap generation.
    Real ball detections are draw in green and interpolated are drawn in red.

Game Tracking:
- self.bounces_array (list[int])
    List of frame indices where bounces were detected by the model or logic. Used for event annotation 
    and post-processing. Detected using the model. self.bounce is currently being detected using a 
    manual method

- self.hits (set[int])
    Set of frame indices identified as player hits. May be used for visual overlays or scoring logic.

- self.point_events (list[dict])
    Key events per scored point. Each dict includes metadata like: frame_idx, winner, how the point was won, 
    last accepted hit. Useful for generating stats and game timelines.

Player Tracking: 
- top_two_players_dicts (dict[int, dict[int, tuple[int, int, int, int]]])
    Mapping of frame index → player ID → bounding box (x1, y1, x2, y2). Used to determine top two players 
    for each frame. 
    WARNING: The player ID may change mid way through so it is unreliable to consistenly use a player ID.
              To overcome this since there will always be only two players maximum for any given frame, we 
              can determine whether this is the player closer or farther away by using homography. This 
              can then be use to determine the player's position on the minimap or who the player is. 

===========================================================================================================

PIPELINE

Video Input → 
Initial Loop 
    { Frame Filtering → Ball Tracking → Player Tracking → Court Homography → Player Heatmap } → 
Post Processing Loop 
    { Bounce Detection → Hit Detection → Point Segmentation → Data Analysis (includes bounce heatmaps etc) } → 
Output Generation
NOTE: Both the initial loop and the post processing loop are used to loop through all the frames

===========================================================================================================

OUTPUTS

process_input_video.mp4: main video with player's bounding boxes, ball tracking, minimap

heatmap_players.mp4: heatmap of players and minimap over time

minimap_heatmap.mp4: court-based bounce heatmap

minimap_heatmap_weak.mp4: filtered heatmap for bounces that do not pass the service line

===========================================================================================================

SPEED - 8 seconds per 30 frames

Currently the code takes approximately 4 seconds per 30 frames but since it needs to be looped through twice 
(once for the initial loop and once for the post processing loop) it averages to around 8 seconds per 30 frames. 

The intial 4 seconds per 30 fame is due to use of GPU tracking the ball, players, and court. The post processing takes
a shorter time as it only uses GPU to detect bounces but must once again loop through the whole frames again. However 
the time averages to be about the same due to creating heatmaps and completing analysis. The score_single_pass function 
seems to take a lot of time. 

===========================================================================================================