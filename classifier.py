import math


def classify_shot(player_keypoints):

    # Ensure enough keypoints exist
    if len(player_keypoints) < 11:
        return "unknown"

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6

    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    try:

        rw = player_keypoints[RIGHT_WRIST]
        rs = player_keypoints[RIGHT_SHOULDER]

        lw = player_keypoints[LEFT_WRIST]
        ls = player_keypoints[LEFT_SHOULDER]

        # Ignore invalid points
        if rw[0] == 0 or rw[1] == 0:
            return "unknown"

        # Smash
        if rw[1] < rs[1] - 40:
            return "smash"

        # Forehand
        if rw[0] > rs[0] + 50:
            return "forehand"

        # Backhand
        if rw[0] < rs[0] - 50:
            return "backhand"

    except:
        return "unknown"

    return "unknown"