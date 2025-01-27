import streamlit as st
import cv2

def main():
    st.title("Facial Recognition App")
    run = st.checkbox("Enable Camera", key="enable_camera")
    FRAME_WINDOW = st.image([], use_container_width=True)
    left_column, right_column = st.columns(2)
    left_column.button("Upload", use_container_width=True, key="save_btn")
    right_column.button("Recognize", use_container_width=True, key="recognize_btn")
    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            FRAME_WINDOW.image(frame)
        else:
            st.write("Camera Stopped")
    camera.release()