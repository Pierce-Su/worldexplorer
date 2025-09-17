import cv2
import numpy as np

# should be 120
def find_first_bright_frame(video_path, threshold=120):
    """
    Returns the ID of the first frame where any of the following is above the threshold:
    - The total frame mean
    - Middle 250x250 region mean 
    - Middle 100x100 region mean
    
    If no frame meets the condition, returns the total frame count.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return -1
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate coordinates for the middle 250x250 region
    start_x_250 = (width - 250) // 2
    start_y_250 = (height - 250) // 2
    end_x_250 = start_x_250 + 250
    end_y_250 = start_y_250 + 250
    
    # Calculate coordinates for the middle 100x100 region
    start_x_100 = (width - 100) // 2
    start_y_100 = (height - 100) // 2
    end_x_100 = start_x_100 + 100
    end_y_100 = start_y_100 + 100
    
    frame_count = 0
    
    # Loop through all frames in the video
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # If frame was not read successfully, we've reached the end of the video
        if not ret:
            break
        
        # Ensure the frame is in grayscale format
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # Extract the regions
        middle_region_250 = gray_frame[start_y_250:end_y_250, start_x_250:end_x_250]
        middle_region_100 = gray_frame[start_y_100:end_y_100, start_x_100:end_x_100]
        
        # Calculate mean values
        total_mean = np.mean(gray_frame)
        middle_mean_250 = np.mean(middle_region_250)
        middle_mean_100 = np.mean(middle_region_100)
        
        # Check if any mean is above threshold
        # if (total_mean > threshold or 
        #     middle_mean_250 > threshold or 
        #     middle_mean_100 > threshold):
        if (middle_mean_100 > threshold):
            cap.release()  # Release the video capture object
            return frame_count  # Return the current frame ID
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    
    # If no frame was found meeting the condition, return the last ID + 1
    return frame_count
