import os
import numpy as np
import cv2


### Init variables for manual point selection (you can add more if neccessary)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variable to store selected points
points = []

def select_points(event, x, y, flags, param):
    """
    Mouse callback function to select points.
    """
    global points
    # TODO: Add coordinates to 'points' when the left mouse button is double-clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        points.append((x, y))


def main(path_vid: str):
    """
    Main script
    """
    ## TODO: Load the video and show the initial frame.
    # Open the video file
    cap = cv2.VideoCapture(r"D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab04_code\lab04_code\video\shibuya.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        returnsssssss
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video file.")
        cap.release()
        return
    
    # Display the initial frame in a window
    cv2.imshow("Initial Frame", frame)
    # print("Press any key to close the window...")
    # cv2.waitKey(0)
    
    ## Set up mouse callback on the same window ("Initial Frame")
    cv2.namedWindow("Initial Frame")
    cv2.setMouseCallback("Initial Frame", select_points)
    
    ## TODO: Create a standby loop, wait to manually select point on the video frame.
    ## Setup condition to start tracking when 's' is pressed
    print("Select a point by double-clicking on the frame, then press 's' to start tracking.")
    # Standby loop: display the frame and wait for point selection and 's' key press
    while True:
        # Copy the frame to draw the selected points without modifying the original frame
        display_frame = frame.copy()
        # Draw all selected points (if any)
        for pt in points:
            cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
    
        cv2.imshow("Initial Frame", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # If 's' is pressed and at least one point has been selected, start tracking
        if key == ord('s') and len(points) > 0:
            break
        
        # Optionally, allow exit with 'q'
        if key == ord('q'):
            print("Exiting without starting tracking.")
            cap.release()
            cv2.destroyAllWindows()
            return

    ## Reshape array after selection (the variable 'edges' is computed but not used further)
    edges = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    
    ## TODO: Initialize some tracing variables, parameters, etc.
    # Convert selected points into a numpy array of shape (N, 1, 2)
    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    # Create a trajectory list for each point. Each trajectory is a list of (x, y) tuples.
    trajectories = [[pt] for pt in points]
    
    # Set Lucas-Kanade optical flow parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Convert the first frame to grayscale (this will be our reference frame)
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ## MAIN TRACKING LOOP
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if no more frames are available
        
        # Convert current frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ## TODO: Apply Lucas-Kanade to compute optical flow (sparse) for the selected points
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        ## TODO: Filter out points that couldn't be tracked and update the trajectory list.
        if p1 is None:
            break  # Exit if tracking fails
        # st is a status array where 1 means the corresponding point was found
        good_new = p1[st == 1].reshape(-1, 2)
        good_old = p0[st == 1].reshape(-1, 2)
        
        # Update the trajectory for each successfully tracked point
        for i, new_pt in enumerate(good_new):
            trajectories[i].append(tuple(new_pt))
        
        ## TODO: Draw lines and circles to visualize the tracking
        for tr in trajectories:
            # Draw the trajectory lines
            for j in range(1, len(tr)):
                cv2.line(frame, tuple(map(int, tr[j - 1])), tuple(map(int, tr[j])), (0, 255, 0), 2)
            # Draw the current location as a circle
            cv2.circle(frame, tuple(map(int, tr[-1])), 5, (0, 0, 255), -1)
        
        ## TODO: Display current frame, & update the reference frame and the points for the next iteration
        cv2.imshow("Tracking", frame)
        old_gray = frame_gray.copy()         # Update the reference frame
        p0 = good_new.reshape(-1, 1, 2)        # Set the good new points as the next starting points
        
        ## Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    ## TODO: exit video program
    cap.release()
    cv2.destroyAllWindows()
    
    ## TODO: Plot the trajectories into matplotlib figures.
    plt.figure(figsize=(8, 6))
    for tr in trajectories:
        # Convert trajectory list into a NumPy array for easy plotting
        tr_array = np.array(tr)
        plt.plot(tr_array[:, 0], tr_array[:, 1], marker='o', linestyle='-', label="Trajectory")
    plt.title("Tracked Trajectories")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main("D:\Computer-Vision-2025-Jan-25_19-39-22-456\Computer-Vision-2025-Jan-25_19-39-22-456\viewer\files\lab04_code\lab04_code\video\shibuya.mp4")


