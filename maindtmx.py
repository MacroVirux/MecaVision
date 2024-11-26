import cv2
from pyzbar.pyzbar import decode
from pylibdmtx.pylibdmtx import decode as decode_datamatrix
import numpy as np
import threading
from queue import Queue
import streamlit as st

# Streamlit configuration
st.set_page_config(layout="wide")

# Display Title
st.title("QR and Data Matrix Code Scanner")

# Output areas
video_placeholder = st.empty()  # For displaying video
decoded_placeholder = st.empty()  # For displaying decoded results

# Initialize webcam
cap = cv2.VideoCapture(2)  # Adjust index for your webcam

if not cap.isOpened():
    st.error("Error: Could not access the webcam.")
    st.stop()

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

# Streamlit app loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        # Send the latest frame to the decoding thread
        if not frame_queue.full():
            frame_queue.put(frame.copy())

        # Convert frame to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display video feed
        video_placeholder.image(rgb_frame, channels="RGB") 

        # Display decoded results
        if decoded_results:
            decoded_placeholder.write("Decoded Results:")
            for code_type, data in decoded_results:
                st.write(f"**{code_type} Code:** {data}")
            decoded_results.clear()

except Exception as e:
    st.error(f"An error occurred: {e}")

finally:
    # Stop the worker thread
    frame_queue.put(None)
    decode_thread.join()

    cap.release()
