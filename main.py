import cv2
import mediapipe as mp
import numpy as np
import time
import pygame as pg
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # Adjust device index accordingly

prev_time = 0  # For FPS calculation

def draw_start_screen(frame):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, 'TEBAK ANOMALI', (w//2-220, h//2-60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
    cv2.putText(frame, 'Tekan [Spasi] untuk mulai', (w//2-270, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
    return frame

def draw_question_screen(frame, choices, timer=None):
    h, w, _ = frame.shape
    # Area untuk gambar (5 gambar, horizontal), lebih kecil dan di bagian atas
    img_w, img_h = 120, 120  # Ukuran gambar diperkecil
    margin = 30
    total_width = img_w*5 + margin*4
    # Jika frame terlalu kecil, kecilkan gambar dan margin
    if total_width > w:
        scale = w / (img_w*5 + margin*4 + 1)
        img_w = int(img_w * scale)
        img_h = int(img_h * scale)
        margin = int(margin * scale)
        total_width = img_w*5 + margin*4
    start_x = max(0, (w - total_width) // 2)
    y = 30  # Gambar di bagian atas
    for i, cid in enumerate(choices):
        img_path = f"aset/img/{cid}.png"
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            img = cv2.resize(img, (img_w, img_h))
            x = start_x + i*(img_w+margin)
            # Pastikan tidak keluar frame
            if x+img_w > w:
                continue
            # Jika PNG transparan
            if img.shape[2] == 4:
                alpha_s = img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    frame[y:y+img_h, x:x+img_w, c] = (alpha_s * img[:, :, c] + alpha_l * frame[y:y+img_h, x:x+img_w, c])
            else:
                frame[y:y+img_h, x:x+img_w] = img
            # Nomor urut
            cv2.rectangle(frame, (x, y+img_h+5), (x+img_w, y+img_h+40), (0,0,0), -1)
            cv2.putText(frame, str(i+1), (x+img_w//2-15, y+img_h+35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
    if timer is not None:
        cv2.putText(frame, f"Timer: {timer}", (w//2-80, y+img_h+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
    return frame

pg.mixer.init()

# Game state
state = 'start'  # 'start', 'question', 'countdown', 'result', 'end'
current_question = 0
score = 0
max_questions = 5
question_data = []
user_answer = None
countdown_start = None

# Helper: generate question set
all_ids = list(range(1, 25))
def generate_questions():
    questions = []
    used = set()
    for _ in range(max_questions):
        answer = random.choice([i for i in all_ids if i not in used])
        used.add(answer)
        choices = [answer]
        while len(choices) < 5:
            c = random.choice(all_ids)
            if c not in choices:
                choices.append(c)
        random.shuffle(choices)
        questions.append({'answer': answer, 'choices': choices})
    return questions

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        exit(1)
    
    question_data = generate_questions()
    state = 'start'
    current_question = 0
    score = 0
    user_answer = None
    countdown_start = None
    last_gesture = 1
    gesture_history = []  # Untuk stabilisasi gesture
    gesture_stable = 1
    gesture_stable_count = 0
    gesture_stable_required = 3  # butuh 3 frame berturut-turut
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        frame = cv2.flip(frame, 1)
        key = cv2.waitKey(1) & 0xFF

        if state == 'start':
            frame = draw_start_screen(frame)
            cv2.imshow('Hand Finger Count', frame)
            if key == ord(' '):
                state = 'question'
                current_question = 0
                score = 0
                question_data = generate_questions()
                user_answer = None
                countdown_start = None
                last_gesture = 1
                gesture_history = []
                gesture_stable = 1
                gesture_stable_count = 0
            continue

        if state == 'question':
            q = question_data[current_question]
            # Play sound only once per question
            if countdown_start is None:
                sound_path = f"aset/sound/{q['answer']}.wav"
                pg.mixer.music.load(sound_path)
                pg.mixer.music.play()
                countdown_start = time.time()
            timer = 10 - int(time.time() - countdown_start)  # Ubah countdown jadi 10 detik
            frame = draw_question_screen(frame, q['choices'], timer=timer)
            # Hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            results = hands.process(rgb)
            finger_tips = [4, 8, 12, 16, 20]
            gesture_count = None
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                landmarks = hand_landmarks.landmark
                fingers_up = []
                if handedness == 'Right':
                    fingers_up.append(1 if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x else 0)
                else:
                    fingers_up.append(1 if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x else 0)
                for tip in finger_tips[1:]:
                    fingers_up.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)
                count = sum(fingers_up)
                if 1 <= count <= 5:
                    gesture_count = count
                    gesture_history.append(count)
                    if len(gesture_history) > gesture_stable_required:
                        gesture_history.pop(0)
                    # Cek stabilisasi
                    if len(gesture_history) == gesture_stable_required and all(g == gesture_history[0] for g in gesture_history):
                        gesture_stable = gesture_history[0]
                        gesture_stable_count = gesture_stable_required
                    else:
                        gesture_stable_count = 0
                # Visualisasi landmark tangan
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Tampilkan gesture yang terbaca di layar
            if gesture_count is not None:
                cv2.putText(frame, f'Gesture: {gesture_count}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            else:
                cv2.putText(frame, 'Gesture: -', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            # Tampilkan gesture stabil
            cv2.putText(frame, f'Stable: {gesture_stable if gesture_stable_count else "-"}', (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
            # Countdown
            if timer <= 0:
                user_answer = gesture_stable if gesture_stable_count else last_gesture
                state = 'result'
                countdown_start = None
                pg.mixer.music.stop()
            cv2.imshow('Hand Finger Count', frame)
            continue

        if state == 'result':
            q = question_data[current_question]
            idx = q['choices'].index(q['answer'])
            correct = (user_answer == idx+1)
            if correct:
                score += 20
                feedback = 'BENAR!'
                color = (0,255,0)
            else:
                feedback = 'SALAH!'
                color = (0,0,255)
            frame = draw_question_screen(frame, q['choices'])
            h, w, _ = frame.shape
            cv2.putText(frame, feedback, (w//2-120, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            cv2.putText(frame, f'Jawaban: {idx+1}', (w//2-120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
            cv2.putText(frame, f'Skor: {score}', (w//2-120, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
            cv2.imshow('Hand Finger Count', frame)
            cv2.waitKey(1200)
            current_question += 1
            if current_question >= max_questions:
                state = 'end'
            else:
                state = 'question'
            continue

        if state == 'end':
            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, 'SELESAI!', (w//2-180, h//2-60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
            cv2.putText(frame, f'Skor Akhir: {score}', (w//2-220, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
            cv2.putText(frame, 'Tekan [Spasi] untuk main lagi', (w//2-270, h//2+80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            cv2.imshow('Hand Finger Count', frame)
            if key == ord(' '):
                state = 'question'
                current_question = 0
                score = 0
                question_data = generate_questions()
                user_answer = None
                countdown_start = None
                last_gesture = 1
                gesture_history = []
                gesture_stable = 1
                gesture_stable_count = 0
            continue

        # fallback: tampilkan frame
        cv2.imshow('Hand Finger Count', frame)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
