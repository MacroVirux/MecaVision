import cv2
from pyzbar.pyzbar import decode
from pylibdmtx.pylibdmtx import decode as decode_datamatrix
import numpy as np
import threading
from queue import Queue

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or adjust for your webcam index

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

# Queue to handle frames for decoding
frame_queue = Queue(maxsize=1)  # Keep only the most recent frame
decoded_results = []

# Worker thread for decoding
def decode_worker():
    global decoded_results
    while True:
        frame = frame_queue.get()  # Wait for a frame
        if frame is None:
            break  # Stop the worker
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Decode QR codes
        qr_codes = decode(frame)
        for code in qr_codes:
            decoded_results.append(("QR", code.data.decode('utf-8')))

        # Decode Data Matrix codes
        resized_gray = cv2.resize(gray, (640, 480))  # Downscale for performance
        data_matrix_codes = decode_datamatrix(resized_gray, max_count=1)
        for dm_code in data_matrix_codes:
            decoded_results.append(("DM", dm_code.data.decode('utf-8')))

# Start decoding thread
decode_thread = threading.Thread(target=decode_worker, daemon=True)
decode_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Send the latest frame to the decoding thread
    if not frame_queue.full():
        frame_queue.put(frame.copy())

    # Display decoded results
    for code_type, data in decoded_results:
        print(f"{code_type} Code Content: {data}")
    decoded_results.clear()  # Clear results after displaying

    # Show the video feed
    cv2.imshow('frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the worker thread
frame_queue.put(None)
decode_thread.join()

cap.release()
cv2.destroyAllWindows()
