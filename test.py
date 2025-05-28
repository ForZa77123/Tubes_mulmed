import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow('Video Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()