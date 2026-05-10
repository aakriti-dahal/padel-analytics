import json
import os
import pandas as pd
import collections


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved → {path}  ({os.path.getsize(path) // 1024} KB)")


def save_csv(shot_log, path):
    if not shot_log:
        print("No shots detected — CSV not saved.")
        return

    rows = []
    for s in shot_log:
        bp = s["ball_position"]
        rows.append({
            "shot_id":          s["shot_id"],
            "frame":            s["frame"],
            "timestamp_sec":    s["timestamp"],
            "shot_type":        s["shot_type"],
            "direction":        s["direction"],
            "detection_method": s["detection_method"],
            "ball_x":           round(bp[0], 1) if bp else None,
            "ball_y":           round(bp[1], 1) if bp else None,
            "velocity":         s["velocity"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")


def print_summary(shot_log):
    if not shot_log:
        print("No shots detected.")
        return

    counts  = collections.Counter(s["shot_type"]        for s in shot_log)
    methods = collections.Counter(s["detection_method"] for s in shot_log)

    print("\n╔══════════════════════════╗")
    print("║      Shot Summary        ║")
    print("╠══════════════════════════╣")
    for shot_type, count in counts.most_common():
        bar = "█" * count
        print(f"║  {shot_type:<12} {count:>3}  {bar}")
    print("╠══════════════════════════╣")
    print(f"║  {'TOTAL':<12} {len(shot_log):>3}")
    print("╠══════════════════════════╣")
    print("║  Detection method:")
    for method, count in methods.most_common():
        print(f"║    {method:<22} {count:>3}")
    print("╚══════════════════════════╝\n")


def interpolate_ball_positions(all_results):
    known = []
    for i, result in enumerate(all_results):
        ball = next(
            (d for d in result["detections"] if d["label"] == "padel ball"),
            None
        )
        if ball:
            x1, y1, x2, y2 = ball["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            known.append((i, cx, cy))

    if len(known) < 2:
        print(f"  Only {len(known)} ball detections — interpolation skipped.")
        return all_results

    print(f"  Interpolating between {len(known)} known ball positions...")

    for k in range(len(known) - 1):
        i0, x0, y0 = known[k]
        i1, x1, y1 = known[k + 1]

        gap = i1 - i0
        if gap > 300:
            continue

        for i in range(i0 + 1, i1):
            t  = (i - i0) / gap
            cx = x0 + t * (x1 - x0)
            cy = y0 + t * (y1 - y0)
            r  = 8

            already_has_ball = any(
                d["label"] == "padel ball"
                for d in all_results[i]["detections"]
            )
            if not already_has_ball:
                all_results[i]["detections"].append({
                    "class_id":     0,
                    "label":        "padel ball",
                    "confidence":   0.0,
                    "bbox":         [
                        int(cx - r), int(cy - r),
                        int(cx + r), int(cy + r)
                    ],
                    "keypoints":    None,
                    "interpolated": True,
                })

    ball_total = sum(
        1 for r in all_results
        if any(d["label"] == "padel ball" for d in r["detections"])
    )
    print(f"  Ball coverage after interpolation: "
          f"{ball_total}/{len(all_results)} frames")

    return all_results