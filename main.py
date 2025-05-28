import cv2
import mediapipe as mp
import numpy as np
import time
import pygame as pg

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # Adjust device index accordingly

prev_time = 0  # For FPS calculation

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        exit(1)
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)  # MediaPipe requires contiguous array
        results = hands.process(rgb)

        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        finger_tips = [4, 8, 12, 16, 20]

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label  # 'Left' or 'Right'

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            fingers_up = []

            # Thumb logic depends on hand side
            if handedness == 'Right':
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            else:
                if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Other fingers
            for tip in finger_tips[1:]:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            count = sum(fingers_up)
            cv2.putText(frame, f'Fingers: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display FPS at top left
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.imshow('Hand Finger Count', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
