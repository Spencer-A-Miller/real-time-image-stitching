import cv2
import threading
import time

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
