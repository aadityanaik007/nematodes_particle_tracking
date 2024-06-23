import cv2
import numpy as np
import os

# Preprocessing function with enhanced filtering
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Remove small and large objects
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, -1]
    
    min_size = 60  # Lowered min_size to detect smaller particles
    max_size = 500  # Increased max_size to include slightly larger particles
    filtered = np.zeros(labels.shape, dtype=np.uint8)
    
    for i in range(num_labels - 1):
        if min_size <= sizes[i] <= max_size:
            filtered[labels == i + 1] = 255
            
    return filtered

# Feature detection function with ORB parameters tweaked
def detect_features(binary):
    orb = cv2.ORB_create(nfeatures=1000,  # Increased number of features
                         scaleFactor=1.2,  # Scale factor between levels in the scale pyramid
                         nlevels=8,  # Number of levels in the scale pyramid
                         edgeThreshold=15,  # Size of the border where features are not detected
                         firstLevel=0,  # Level of the pyramid to put the source image
                         WTA_K=2,  # Number of points that produce each element of the oriented BRIEF descriptor
                         scoreType=cv2.ORB_HARRIS_SCORE,  # Score used to rank the features
                         patchSize=31,  # Size of the patch used by the oriented BRIEF descriptor
                         fastThreshold=20)  # Threshold for the FAST corner detector
    keypoints, descriptors = orb.detectAndCompute(binary, None)
    return keypoints, descriptors

# Tracking function with BFMatcher
def track_particles(descriptors1, descriptors2, kp1, kp2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    matched_points = [(kp1[m.queryIdx].pt, kp2[m.trainIdx].pt) for m in matches]
    return matched_points

# Main function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    
    if not ret:
        print("Failed to open video")
        return
    
    binary1 = preprocess_frame(frame1)
    kp1, des1 = detect_features(binary1)
    
    while cap.isOpened():
        ret, frame2 = cap.read()
        
        if not ret:
            break
        
        binary2 = preprocess_frame(frame2)
        kp2, des2 = detect_features(binary2)
        
        if des1 is not None and des2 is not None:
            matched_points = track_particles(des1, des2, kp1, kp2)
            
            for pt1, pt2 in matched_points:
                cv2.circle(frame2, (int(pt2[0]), int(pt2[1])), 5, (0, 255, 0), -1)
            
            kp1, des1 = kp2, des2
        
        cv2.imshow("Tracked Frame", frame2)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Construct the relative path to the video file
relative_video_path = os.path.join("..", "videos", "nematodeTestVideo8.mp4")

# Run the processing function on the provided video file
process_video(relative_video_path)
