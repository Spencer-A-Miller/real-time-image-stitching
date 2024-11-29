import cv2
import threading
import time
import numpy as np
from src.images import Image
from src.matching import (
    MultiImageMatches,
    PairMatch,
    build_homographies,
    find_connected_components,
)

from src.rendering import multi_band_blending, set_gain_compensations, simple_blending


# Replace with the actual IP address and port shown in DroidCam app on your Android device
android_ip = ''
port = ''

cap1 = cv2.VideoCapture(f'http://{android_ip}:{port}/video')
cap2 = cv2.VideoCapture(0)

# Shared variables
frame1 = None
frame2 = None
stop_thread = False

# Lock for thread synchronization
lock = threading.Lock()

# Function to read frames from Android device
def read_android_frame():
    global frame1, stop_thread
    while not stop_thread:
        ret1, frame1_new = cap1.read()
        if not ret1:
            break
        with lock:
            frame1 = frame1_new

# Function to read frames from laptop webcam
def read_laptop_frame():
    global frame2, stop_thread
    while not stop_thread:
        ret2, frame2_new = cap2.read()
        if not ret2:
            break
        with lock:
            frame2 = frame2_new

# Function to stitch frames and display the result
def stitch_frames():
    global stop_thread
    prev_time = time.time()
    frame_count = 0
    while not stop_thread:
        with lock:
            if frame1 is not None and frame2 is not None:
                image1 = Image(frame1)
                image2 = Image(frame2)

                image1.compute_features()
                image2.compute_features()

                images = [image1, image2]
                
                matcher = MultiImageMatches(images)
                pair_matches: list[PairMatch] = matcher.get_pair_matches()
                pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)
                connected_components = find_connected_components(pair_matches)

                build_homographies(connected_components, pair_matches)

                time.sleep(0.1)

                for connected_component in connected_components:
                    component_matches = [
                        pair_match
                        for pair_match in pair_matches
                        if pair_match.image_a in connected_component
                    ]

                    set_gain_compensations(
                        connected_component,
                        component_matches,
                        sigma_n=10,
                        sigma_g=0.1,
                    )

                time.sleep(0.1)

                for image in images:
                    image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)

                results = []
                
                results = [
                    simple_blending(connected_component)
                    for connected_component in connected_components
                ]

                # Stitch the two frames side by side
                stitched_frame = cv2.hconcat([frame1, frame2])
                # Display the stitched video
                cv2.imshow('Stitched Video', stitched_frame)
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - prev_time
                if elapsed_time >= 1:
                    fps = frame_count / elapsed_time
                    print(f"Combined FPS: {fps:.2f}")
                    frame_count = 0
                    prev_time = current_time
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_thread = True
            break

# Start the threads
thread1 = threading.Thread(target=read_android_frame)
thread2 = threading.Thread(target=read_laptop_frame)
thread3 = threading.Thread(target=stitch_frames)

thread1.start()
thread2.start()
thread3.start()

# Wait for threads to finish
thread1.join()
thread2.join()
thread3.join()

cap1.release()
cap2.release()
cv2.destroyAllWindows()
