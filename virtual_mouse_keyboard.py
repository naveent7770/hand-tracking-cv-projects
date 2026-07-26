"""
Virtual Mouse & Keyboard using Hand Tracking
---------------------------------------------
Move your INDEX finger to move the mouse cursor.
Pinch your THUMB + INDEX finger together to CLICK.
Hover + pinch over a key on the on-screen virtual keyboard to TYPE.

Install requirements first (run in your terminal, not here):
    pip install opencv-python mediapipe pyautogui numpy

Run:
    python virtual_mouse_keyboard.py

Press 'q' to quit, 'm' to toggle between Mouse mode and Keyboard mode.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ---------------------- CONFIG ----------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
SMOOTHING = 5          # higher = smoother but more lag
CLICK_DISTANCE = 35    # pixel distance between thumb & index to count as a "pinch"
CLICK_COOLDOWN = 0.4   # seconds between clicks/key presses

pyautogui.FAILSAFE = False  # prevents pyautogui from throwing errors at screen corners

# ---------------------- HAND TRACKER ----------------------
class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_landmark_positions(self, frame):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            for idx, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((idx, cx, cy))
        return landmark_list


# ---------------------- VIRTUAL KEYBOARD ----------------------
KEYS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "BACK"],
]

def draw_keyboard(frame, key_boxes):
    overlay = frame.copy()
    for key, (x, y, w, h) in key_boxes.items():
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 1)
        font_scale = 0.5 if len(key) > 2 else 0.8
        cv2.putText(overlay, key, (x + 8, y + int(h * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

def build_key_boxes(start_x=50, start_y=250, key_w=45, key_h=45, gap=8):
    boxes = {}
    for row_idx, row in enumerate(KEYS):
        x = start_x
        y = start_y + row_idx * (key_h + gap)
        for key in row:
            w = key_w * 3 if key == "SPACE" else (key_w * 2 if key == "BACK" else key_w)
            boxes[key] = (x, y, w, key_h)
            x += w + gap
    return boxes


# ---------------------- MAIN LOOP ----------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW avoids Windows camera hang issues
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Try changing the index (0 -> 1) or closing other apps using the camera.")
        return

    tracker = HandTracker(max_hands=1)
    key_boxes = build_key_boxes()

    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0
    mode = "mouse"   # "mouse" or "keyboard"
    typed_text = ""
    last_action_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera not found / could not read frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror view feels natural
        frame = tracker.find_hands(frame)
        landmarks = tracker.get_landmark_positions(frame)

        cv2.putText(frame, f"Mode: {mode.upper()}  (press 'm' to switch, 'q' to quit)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if landmarks:
            index_x, index_y = landmarks[8][1], landmarks[8][2]
            thumb_x, thumb_y = landmarks[4][1], landmarks[4][2]
            pinch_dist = np.hypot(index_x - thumb_x, index_y - thumb_y)
            pinching = pinch_dist < CLICK_DISTANCE

            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
            if pinching:
                cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 255), 3)

            if mode == "mouse":
                screen_x = np.interp(index_x, (100, CAM_WIDTH - 100), (0, SCREEN_WIDTH))
                screen_y = np.interp(index_y, (100, CAM_HEIGHT - 100), (0, SCREEN_HEIGHT))

                curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
                curr_y = prev_y + (screen_y - prev_y) / SMOOTHING
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                if pinching and (time.time() - last_action_time) > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_action_time = time.time()
                    cv2.putText(frame, "CLICK", (index_x + 15, index_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif mode == "keyboard":
                frame = draw_keyboard(frame, key_boxes)
                for key, (x, y, w, h) in key_boxes.items():
                    if x < index_x < x + w and y < index_y < y + h:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        font_scale = 0.5 if len(key) > 2 else 0.8
                        cv2.putText(frame, key, (x + 8, y + int(h * 0.65)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
                        if pinching and (time.time() - last_action_time) > CLICK_COOLDOWN:
                            last_action_time = time.time()
                            if key == "SPACE":
                                typed_text += " "
                            elif key == "BACK":
                                typed_text = typed_text[:-1]
                            else:
                                typed_text += key
                            pyautogui.typewrite(key if len(key) == 1 else "")

                cv2.rectangle(frame, (50, 150), (590, 210), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, typed_text[-30:], (60, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Virtual Mouse & Keyboard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "keyboard" if mode == "mouse" else "mouse"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()