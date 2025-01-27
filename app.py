import streamlit as st
import cv2
import os

from detector import Detector
from augmenter import Augmenter
from derainer import Derainer
from recognizer import Recognizer

class App:
    
    def __init__(self, device):
        self.augmenter = Augmenter()
        self.detector = Detector(device)
        self.derainer = Derainer(device)
        self.recognizer = Recognizer(device)
        
    def preprocess(self, frame):
        og_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = self.detector(frame)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print("Confidence: ", box.conf, "Boxes: ", box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'Face: {box.conf}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return og_frame, frame
    
    def augment(self, frame):
        cv2.resize(frame, (800, 600))
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.augmenter(frame)
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def recognize(self, frame):
        rainy_img = self.augment(frame)
        st.write("Step 1: Augmenting image with rain...")
        left_column, right_column = st.columns(2)
        left_column.image(frame, caption="Original Image")
        right_column.image(rainy_img, caption="Rainy Image")
        
        st.write("Step 2: Deraining image...")
        derained_img = self.derainer(rainy_img)
        left_column, right_column = st.columns(2)
        left_column.image(rainy_img, caption="Rainy Image")
        right_column.image(derained_img, caption="Derained Image")
        
        st.write("Step 3: Recognizing face...")
        cropped_images = []
        boxes = self.detector(derained_img)
        if boxes:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = derained_img[y1:y2, x1:x2]
                cropped_images.append(cropped)
                st.image(cropped, caption=f"Cropped Face {i + 1}", channels="RGB", use_column_width=True)
            st.write(f"{len(cropped_images)} bounding box(es) detected.")
        else:
            st.write("No bounding boxes detected. Passing the image directly to the recognizer...")
            cropped_images.append(derained_img)
        
        st.write("Step 4: Recognizing...")
        for i, cropped in enumerate(cropped_images):
            result = self.recognizer(cropped, cv2.imread("image.jpg"))
            st.write(f"Recognition Result for Face {i + 1}: {result}")
    
    def __call__(self):
        st.title("Facial Recognition App")
        run = st.checkbox("Enable Camera", key="enable_camera")
        FRAME_WINDOW = st.image([], use_container_width=True)
        left_column, right_column = st.columns(2)
        save_btn = left_column.button("Upload", use_container_width=True, key="save_btn")
        recognize_btn = right_column.button("Recognize", use_container_width=True, key="recognize_btn")
        camera = cv2.VideoCapture(0)
        while run:
            ret, frame = camera.read()
            if ret:
                og_frame, frame = self.preprocess(frame)
                frame = cv2.resize(frame, (800, 600))
                FRAME_WINDOW.image(frame)
                if save_btn:
                    cv2.imwrite("image.jpg", frame)
                    st.write("Image Saved")
                if recognize_btn:
                    if not os.path.exists("image.jpg"):
                        st.write("No image to recognize. Please save an image first.")
                        continue
                    st.write("Recognizing...")
                    st.session_state.enable_camera = False
                    self.recognize(og_frame)
                    st.session_state.enable_camera = True
            else:
                st.write("Camera Stopped")
        camera.release()