import cv2
import mediapipe as mp
import numpy as np
import pickle

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        with open('F:/ISL/models/gesture_classifier.pickle', 'rb') as f:
            self.model = pickle.load(f)
        with open('F:/ISL/models/label_map.pickle', 'rb') as f:
            self.label_map = pickle.load(f)

        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.predicted_label = ""
        self.frame_counter = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        self.frame_counter += 1

        if results.multi_hand_landmarks and self.frame_counter % 5 == 0:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0.0] * 42)

            if len(data_aux) == 84:
                prediction = self.model.predict([np.array(data_aux)])
                self.predicted_label = self.inv_label_map[int(prediction[0])]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Show prediction
        cv2.putText(frame, self.predicted_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 3, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_prediction(self):
        return self.predicted_label
