import cv2
import numpy as np
import os

# Preprocessing function to detect white-colored moving nematodes
def preprocess_frame(frame, prev_frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to handle varying lighting conditions and better isolate white nematodes
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Motion detection: Frame differencing to detect moving objects
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # Lowered threshold for more sensitivity
        binary = cv2.bitwise_and(binary, motion_mask)

    # Morphological operations to reduce noise and close gaps
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove small noise

    # Find contours (worm-shaped objects)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape (elongated objects)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Adjust these values based on the expected size of your nematodes
        if 500 < area < 5000:  # Lowered the minimum area to capture smaller nematodes
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))  # Circularity: 1 for circles
            # Reduced reliance on circularity to capture elongated shapes better
            if circularity < 0.6:  # Adjusted to allow more elongated shapes
                filtered_contours.append(contour)

    # Return up to 3 contours (assuming there are three nematodes in the video)
    return filtered_contours[:3], gray

# Main function to process video with motion and contour tracking
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()

    if not ret:
        print("Failed to open video")
        return

    prev_frame = None

    while cap.isOpened():
        ret, frame2 = cap.read()

        if not ret:
            break

        # Preprocess the frame and detect contours of moving, white nematodes
        contours, gray_frame = preprocess_frame(frame2, prev_frame)
        prev_frame = gray_frame  # Store the current frame for motion detection in the next loop

        # Track contours by drawing bounding boxes around each nematode
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

        # Resize frame for display
        scale_percent = 20  # Zoom out by 50%
        width = int(frame2.shape[1] * scale_percent / 100)
        height = int(frame2.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

        # Display the frame with tracked contours
        cv2.imshow("Tracked Frame", resized_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Path to the video file

# Construct the relative path to the video file
relative_video_path = os.path.join("videos/shorten-clip-nematode-single.mp4")
print(relative_video_path)

# Run the processing function on the provided video file
process_video(relative_video_path)
