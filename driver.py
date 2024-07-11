import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a video file 'driving_video.mp4'
video_path = 'driving_video.mp4'

# Load video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Parameters for lane detection (simplified for example)
lane_detection_threshold = 200  # Example threshold for detecting lane changes
frame_count = 0
lane_change_detected = []

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process frame for lane detection (simplified example)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Assume that if the sum of edges exceeds a threshold, a lane change is detected
    if np.sum(edges) > lane_detection_threshold:
        lane_change_detected.append(True)
    else:
        lane_change_detected.append(False)
    
    frame_count += 1

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Simulated speed and acceleration data as in previous example
np.random.seed(0)
time = np.arange(0, frame_count, 0.5)
speed = np.random.normal(loc=50, scale=10, size=len(time))
acceleration = np.diff(speed, prepend=speed[0]) / np.diff(time, prepend=1)

# Create DataFrame for analysis
data = pd.DataFrame({'Time': time, 'Speed': speed, 'Acceleration': acceleration})
data['Lane_Change'] = lane_change_detected[:len(data)]  # Ensure the lengths match

# Analyze data for risky behavior
speed_threshold = 80
acceleration_threshold = 10

data['Speeding'] = data['Speed'] > speed_threshold
data['Sudden_Acceleration'] = np.abs(data['Acceleration']) > acceleration_threshold

# Plotting the data
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(data['Time'], data['Speed'], label='Speed')
plt.axhline(speed_threshold, color='r', linestyle='--', label='Speed Threshold')
plt.scatter(data['Time'][data['Speeding']], data['Speed'][data['Speeding']], color='r', label='Speeding')
plt.title('Driver Speed Analysis')
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data['Time'], data['Acceleration'], label='Acceleration', color='g')
plt.axhline(acceleration_threshold, color='r', linestyle='--', label='Acceleration Threshold')
plt.axhline(-acceleration_threshold, color='r', linestyle='--')
plt.scatter(data['Time'][data['Sudden_Acceleration']], data['Acceleration'][data['Sudden_Acceleration']], color='r', label='Sudden Acceleration')
plt.title('Driver Acceleration Analysis')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data['Time'], data['Lane_Change'], label='Lane Change', color='b')
plt.title('Lane Change Detection')
plt.xlabel('Time (s)')
plt.ylabel('Lane Change (Boolean)')
plt.legend()

plt.tight_layout()
plt.show()

# Display periods of risky behavior
risky_behavior = data[(data['Speeding']) | (data['Sudden_Acceleration']) | (data['Lane_Change'])]
print("Periods of Risky Behavior:")
print(risky_behavior[['Time', 'Speed', 'Acceleration', 'Speeding', 'Sudden_Acceleration', 'Lane_Change']])
