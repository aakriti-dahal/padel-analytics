import cv2


def draw_ball(frame, balls):

    for ball in balls:

        x1, y1, x2, y2 = map(int, ball["bbox"])

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 255), 2)

        cv2.putText(frame, "Ball",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

    return frame


def draw_shot(frame, shot_type):

    cv2.putText(frame,
                f"Shot: {shot_type}",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    return frame