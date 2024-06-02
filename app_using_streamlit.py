import streamlit as st
import mediapipe as mp
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import threading
import time
lock = threading.Lock()
text_container = ""
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'L', 1: 'A', 2: 'B', 3: 'c', 4: 'Hello'}
text_output = st.empty()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
placeholder = st.empty()
class signdetection(VideoTransformerBase):

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.text = None

    def transform(self, frame):
        frame_input = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)

        data_aux = []
        # frame = cv2.imread(input)
        frame_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        results = hands.process(frame_rgb)
        with self.frame_lock:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    frame_input,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                self.text = labels_dict[int(prediction[0])]
            else:
                self.text = "No hand landmarks detected in the current frame."
        return frame_input

def main():

    st.title("Sign language translation app")
    activities = ["Home", "Webcam sign detection"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    if choice == "Home":
        st.write("""
                    This is sign language translation app

                     """)
    elif choice == "Webcam sign detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your sign")

        webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=signdetection, media_stream_constraints={"video": True, "audio": False})
        text_output = st.empty()

        while webrtc_ctx.video_transformer:
            with webrtc_ctx.video_transformer.frame_lock:
                output = webrtc_ctx.video_transformer.text
                text_output.markdown(f"**Text:** {output}")
                time.sleep(0.0001)

if __name__ == "__main__":
    main()



