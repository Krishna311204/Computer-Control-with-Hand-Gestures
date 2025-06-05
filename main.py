import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
dragging = False
def distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_h, image_w, _ = frame.shape
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            idx_finger = (int(lm[8].x * image_w), int(lm[8].y * image_h))
            mid_finger = (int(lm[12].x * image_w), int(lm[12].y * image_h))
            thumb = (int(lm[4].x * image_w), int(lm[4].y * image_h))
            screen_x = np.interp(idx_finger[0], [0, image_w], [0, screen_w])
            screen_y = np.interp(idx_finger[1], [0, image_h], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)
            dist_thumb_index = distance(idx_finger, thumb)
            dist_thumb_middle = distance(mid_finger, thumb)
            dist_index_middle = distance(idx_finger, mid_finger)
            if dist_thumb_index < 30:
                pyautogui.click()
                cv2.putText(frame, 'Left Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                time.sleep(0.3)
            elif dist_thumb_middle < 30:
                pyautogui.rightClick()
                cv2.putText(frame, 'Right Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                time.sleep(0.3)
            elif dist_thumb_index < 40 and not dragging:
                pyautogui.mouseDown()
                dragging = True
                cv2.putText(frame, 'Dragging...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif dragging and dist_thumb_index > 50:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(frame, 'Drop', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time.sleep(0.2)
            elif dist_index_middle < 40:
                scroll_y = idx_finger[1] - mid_finger[1]
                pyautogui.scroll(int(-scroll_y * 5))
                cv2.putText(frame, 'Scrolling...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Mouse Control', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
