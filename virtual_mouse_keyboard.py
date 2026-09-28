import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time


# =========================================================
# SETTINGS
# =========================================================

CAM_WIDTH = 640
CAM_HEIGHT = 480

SMOOTHING = 5
CLICK_DISTANCE = 35
CLICK_COOLDOWN = 0.5

pyautogui.FAILSAFE = False


# =========================================================
# MEDIAPIPE HAND TRACKER
# =========================================================

class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.results = None


    def find_hands(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:

            for hand_landmarks in self.results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame


    def get_landmarks(self, frame):

        landmarks = []

        if self.results is not None:
            if self.results.multi_hand_landmarks:

                hand = self.results.multi_hand_landmarks[0]

                h, w, _ = frame.shape

                for i, lm in enumerate(hand.landmark):

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    landmarks.append((i, x, y))

        return landmarks


    def close(self):

        self.hands.close()


# =========================================================
# VIRTUAL KEYBOARD
# =========================================================

KEYS = [

    list("QWERTYUIOP"),

    list("ASDFGHJKL"),

    list("ZXCVBNM"),

    ["SPACE", "BACKSPACE", "CLEAR"]

]


def build_keyboard():

    boxes = {}

    start_x = 35
    start_y = 250

    key_w = 45
    key_h = 45

    gap = 8

    for row_number, row in enumerate(KEYS):

        x = start_x

        y = start_y + row_number * (key_h + gap)

        for key in row:

            if key == "SPACE":
                width = key_w * 3

            elif key == "BACKSPACE":
                width = key_w * 2

            elif key == "CLEAR":
                width = key_w * 2

            else:
                width = key_w

            boxes[key] = (x, y, width, key_h)

            x += width + gap

    return boxes


def draw_keyboard(frame, keyboard):

    for key, (x, y, w, h) in keyboard.items():

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (45, 45, 45),
            -1
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

        if len(key) == 1:
            font_size = 0.8
        else:
            font_size = 0.5

        text_x = x + 8
        text_y = y + 29

        cv2.putText(
            frame,
            key,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            2
        )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("-----------------------------------------")
    print(" AI VIRTUAL MOUSE & KEYBOARD")
    print("-----------------------------------------")
    print("M = Change Mouse/Keyboard Mode")
    print("Q = Quit")
    print("-----------------------------------------")


    # Screen size

    screen_width, screen_height = pyautogui.size()

    print("Screen:", screen_width, "x", screen_height)


    # Camera

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("\nERROR: Camera open nahi ho raha.")
        print("Camera permission/check camera.")
        return


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


    # Hand tracker

    tracker = HandTracker()

    keyboard = build_keyboard()


    # Variables

    mode = "mouse"

    prev_x = 0
    prev_y = 0

    last_click_time = 0

    typed_text = ""

    previous_pinch = False

    previous_time = time.time()


    # =====================================================
    # LOOP
    # =====================================================

    while True:

        success, frame = cap.read()

        if not success:

            print("Camera frame read nahi ho raha.")
            break


        # Mirror camera

        frame = cv2.flip(frame, 1)


        # Hand detection

        frame = tracker.find_hands(frame)

        landmarks = tracker.get_landmarks(frame)


        # =================================================
        # FPS
        # =================================================

        current_time = time.time()

        fps = 1 / max(current_time - previous_time, 0.001)

        previous_time = current_time


        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (520, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # =================================================
        # MODE DISPLAY
        # =================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (250, 45),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            "MODE: " + mode.upper(),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # =================================================
        # HAND FOUND
        # =================================================

        if landmarks:

            # Index finger = landmark 8
            index_x = landmarks[8][1]
            index_y = landmarks[8][2]

            # Thumb = landmark 4
            thumb_x = landmarks[4][1]
            thumb_y = landmarks[4][2]


            # Draw index finger

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (0, 255, 0),
                -1
            )


            # Draw thumb

            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                -1
            )


            # Distance between thumb and index

            distance = np.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )


            pinching = distance < CLICK_DISTANCE


            # Draw line when pinching

            if pinching:

                cv2.line(
                    frame,
                    (index_x, index_y),
                    (thumb_x, thumb_y),
                    (0, 255, 255),
                    3
                )


            # =================================================
            # MOUSE MODE
            # =================================================

            if mode == "mouse":

                # Convert camera position to screen position

                screen_x = np.interp(
                    index_x,
                    (80, CAM_WIDTH - 80),
                    (0, screen_width)
                )

                screen_y = np.interp(
                    index_y,
                    (80, CAM_HEIGHT - 80),
                    (0, screen_height)
                )


                # Smoothing

                curr_x = prev_x + (
                    screen_x - prev_x
                ) / SMOOTHING


                curr_y = prev_y + (
                    screen_y - prev_y
                ) / SMOOTHING


                # Move mouse

                try:

                    pyautogui.moveTo(
                        int(curr_x),
                        int(curr_y),
                        duration=0
                    )

                except Exception:

                    pass


                prev_x = curr_x
                prev_y = curr_y


                # =================================================
                # CLICK
                # =================================================

                current_time = time.time()


                if (
                    pinching
                    and not previous_pinch
                    and current_time - last_click_time > CLICK_COOLDOWN
                ):

                    try:

                        pyautogui.click()

                        print("Mouse Click")

                    except Exception:

                        pass


                    last_click_time = current_time


                    cv2.putText(
                        frame,
                        "CLICK!",
                        (index_x + 15, index_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )


            # =================================================
            # KEYBOARD MODE
            # =================================================

            elif mode == "keyboard":

                draw_keyboard(
                    frame,
                    keyboard
                )


                # Check every key

                for key_name, box in keyboard.items():

                    x, y, w, h = box


                    # Finger is inside key

                    if (
                        x < index_x < x + w
                        and
                        y < index_y < y + h
                    ):


                        # Highlight key

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            -1
                        )


                        font_size = 0.5 if len(key_name) > 1 else 0.8


                        cv2.putText(
                            frame,
                            key_name,
                            (x + 8, y + 29),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size,
                            (0, 0, 0),
                            2
                        )


                        # Key press only once per pinch

                        current_time = time.time()


                        if (
                            pinching
                            and not previous_pinch
                            and current_time - last_click_time > CLICK_COOLDOWN
                        ):

                            last_click_time = current_time


                            # SPACE

                            if key_name == "SPACE":

                                typed_text += " "

                                pyautogui.press("space")


                            # BACKSPACE

                            elif key_name == "BACKSPACE":

                                typed_text = typed_text[:-1]

                                pyautogui.press("backspace")


                            # CLEAR

                            elif key_name == "CLEAR":

                                typed_text = ""


                            # Normal alphabet

                            else:

                                typed_text += key_name

                                pyautogui.press(
                                    key_name.lower()
                                )


                        break


                # =================================================
                # TYPED TEXT DISPLAY
                # =================================================

                cv2.rectangle(
                    frame,
                    (10, 140),
                    (630, 210),
                    (0, 0, 0),
                    -1
                )


                cv2.rectangle(
                    frame,
                    (10, 140),
                    (630, 210),
                    (0, 255, 0),
                    2
                )


                display_text = typed_text[-35:]


                cv2.putText(
                    frame,
                    "TYPED: " + display_text,
                    (20, 185),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            previous_pinch = pinching


        else:

            # No hand

            previous_pinch = False


        # =================================================
        # INSTRUCTIONS
        # =================================================

        cv2.putText(
            frame,
            "M = Mode | Q = Quit",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # =================================================
        # SHOW WINDOW
        # =================================================

        cv2.imshow(
            "AI Virtual Mouse & Keyboard",
            frame
        )


        # =================================================
        # KEYBOARD INPUT
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        # Q = Quit

        if key == ord("q"):

            break


        # M = Mode change

        elif key == ord("m"):

            if mode == "mouse":

                mode = "keyboard"

                print("Mode: KEYBOARD")

            else:

                mode = "mouse"

                print("Mode: MOUSE")


    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    tracker.close()

    cv2.destroyAllWindows()

    print("\nProgram closed successfully.")


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
