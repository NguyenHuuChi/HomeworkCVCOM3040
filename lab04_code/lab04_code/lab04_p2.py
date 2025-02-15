from typing import List
import cv2
import numpy as np
import time
import random

def initialize_tracker(tracker_type, frame, roi):
    """
    Initializes and returns the specified tracker for the given ROI and frame.
    """
    tracker = None
    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    # elif tracker_type == "MOSSE":
    #     tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    else:
        print("Unsupported tracker type:", tracker_type)
        return None

    tracker.init(frame, roi)
    return tracker


def select_object(cap):
    """
    Allows the user to select one bounding box to track in the first frame.
    Returns the selected ROI and the first frame.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video file.")
        return None, None
    roi = cv2.selectROI("Select Object", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Object")
    return roi, frame


def track_object(cap, tracker_types: List[str], roi, frame):
    """
    Tracks the selected object using multiple trackers and compares their performance.
    """
    # Initialize trackers for each tracker type
    trackers = {}
    for t in tracker_types:
        tracker = initialize_tracker(t, frame, roi)
        if tracker is not None:
            trackers[t] = tracker

    # Assign a random color for each tracker for visualization
    colors = {}
    for t in trackers.keys():
        colors[t] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    # Variables to measure performance
    tracker_times = {t: 0.0 for t in trackers.keys()}
    tracker_counts = {t: 0 for t in trackers.keys()}
    total_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # For each tracker, update tracking and draw the bounding box
        for t, tracker in trackers.items():
            t_start = time.time()
            ok, bbox = tracker.update(frame)
            t_end = time.time()
            tracker_times[t] += (t_end - t_start)
            tracker_counts[t] += 1

            if ok:
                # Convert bounding box to integer tuple for drawing
                bbox = tuple(map(int, bbox))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              colors[t], 2)
                cv2.putText(frame, t, (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[t], 2)
            else:
                cv2.putText(frame, t + " lost", (50, 50 + 20 * list(trackers.keys()).index(t)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[t], 2)

        cv2.imshow("Multiple Trackers", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    overall_time = end_time - start_time
    overall_fps = total_frames / overall_time
    print("Overall FPS: {:.2f}".format(overall_fps))
    for t in trackers.keys():
        if tracker_counts[t] > 0:
            avg_time = tracker_times[t] / tracker_counts[t]
            tracker_fps = 1 / avg_time if avg_time > 0 else 0
            print("Tracker: {} - Average update time: {:.4f}s, Approx FPS: {:.2f}".format(t, avg_time, tracker_fps))
    
    cap.release()
    cv2.destroyAllWindows()


def main(vid_path: str, tracker_types: List[str]):
    """
    Main script to load the video, allow object selection, and run multiple trackers.
    """
    cap = cv2.VideoCapture(r"D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab04_code\lab04_code\video\shibuya.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    roi, frame = select_object(cap)
    print("Chiiiiiii")
    if roi is None:
        cap.release()
        return
    print("chiiiiiiiiii")
    track_object(cap, tracker_types, roi, frame)


if __name__ == '__main__':
    # Example usage with four trackers: "KCF", "CSRT", "MIL", "MOSSE"
    trackers = ["KCF", "CSRT", "MIL", "MOSSE"]
    # Replace with your video path as needed.
    main("shibuya.mp4", trackers)
