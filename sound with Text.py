import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
import io
import tempfile

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'L', 1: 'A', 2: 'B', 3: 'C', 4: 'V', 5: 'W', 6: 'Y'}

# Initialize Pygame mixer
pygame.mixer.init()

# Track the current and previous predictions
previous_prediction = None
audio_playing = False

while True:
    data_aux = []

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.extend([x, y])

    if data_aux:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_value = int(prediction[0])
        if predicted_value in labels_dict:
            predicted_character = labels_dict[predicted_value]

            # Check if the prediction has changed
            if predicted_character != previous_prediction:
                print(f"The recognized sign is: {predicted_character}")
                previous_prediction = predicted_character

                # Generate and play audio for the predicted sign
                #text = f"The recognized sign is: {predicted_character}"
                text = predicted_character
                tts = gTTS(text=text, lang='en')

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_filename = fp.name

                tts.save(temp_filename)
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                audio_playing = True

            # Display the recognized character on the frame
            cv2.putText(frame, f'Sign: {predicted_character}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
            #cv2.putText(frame, predicted_character, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,cv2.LINE_AA)

        else:
            print(f"Unknown prediction: {predicted_value}")
    else:
        print("No hand landmarks detected in the current frame.")
        previous_prediction = None  # Reset if no landmarks are detected

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()