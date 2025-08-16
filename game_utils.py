def _classify_zone_from_xy(x_m: float, y_m: float, y_inc_toward_near: bool) -> str:
    """
    Classify a (x_m, y_m) coordinate into one of four zones on a tennis court:
    'left-back', 'right-back', 'left-front', or 'right-front'.

    Zones are split using:
      - Center line at x = 5.485 (half court width)
      - Service line at y = 5.48 meters from each baseline

    Parameters:
        x_m (float): X position in meters (left-right)
        y_m (float): Y position in meters (baseline to baseline)
        y_inc_toward_near (bool): True if y increases toward near baseline

    Returns:
        str: One of 'left-back', 'right-back', 'left-front', 'right-front'
    """
    COURT_WIDTH = 10.97
    COURT_LENGTH = 23.77
    NET_Y = COURT_LENGTH / 2
    MID_X = COURT_WIDTH / 2
    SERVICE_LINE_DIST = 5.48

    # Determine y-position of service lines based on axis direction
    if y_inc_toward_near:
        svc_far_y = SERVICE_LINE_DIST
        svc_near_y = COURT_LENGTH - SERVICE_LINE_DIST
    else:
        svc_far_y = COURT_LENGTH - SERVICE_LINE_DIST
        svc_near_y = SERVICE_LINE_DIST

    # Determine which half the ball landed in
    in_near_half = y_m >= NET_Y if y_inc_toward_near else y_m <= NET_Y
    svc_y = svc_near_y if in_near_half else svc_far_y

    # Determine front vs back
    if in_near_half:
        is_front = (y_m <= NET_Y and not y_inc_toward_near) or (y_inc_toward_near and y_m <= svc_y)
    else:
        is_front = (y_m >= NET_Y and not y_inc_toward_near) or (y_inc_toward_near and y_m >= svc_y)

    # Determine left vs right (from umpire's view)
    is_left = x_m < MID_X

    if is_left and not is_front:
        return "left-back"
    elif not is_left and not is_front:
        return "right-back"
    elif is_left and is_front:
        return "left-front"
    else:
        return "right-front"

