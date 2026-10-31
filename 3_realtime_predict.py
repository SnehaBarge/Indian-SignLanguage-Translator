"""
STEP 3 — Real-Time ISL Recognition
Opens webcam, detects hand with MediaPipe, feeds 30-frame window
to trained LSTM, displays predicted letter live.

Controls:
  SPACE     = add space
  BACKSPACE = delete last character
  C         = clear sentence
  Q / ESC   = quit
"""

import cv2
import numpy as np
import os
import collections
import time
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as mpImage, ImageFormat

MODEL_PATH   = os.path.join('model', 'isl_lstm_model.keras')
ENCODER_PATH = os.path.join('model', 'label_encoder.pkl')
LANDMARKER   = os.path.join('model', 'hand_landmarker.task')

SEQ_LEN      = 30
FEAT         = 63
CONF_THRESH  = 0.80   # minimum confidence to accept a prediction
STAB_FRAMES  = 10     # consecutive frames needed before appending a letter
APPEND_DELAY = 1.2    # seconds between auto-appends (prevents duplicates)


def extract_landmarks(frame, lm):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = lm.detect(mpImage(image_format=ImageFormat.SRGB, data=rgb))
    if res.hand_landmarks:
        c = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
        c -= c[0]
        return c.flatten(), res
    return np.zeros(FEAT, dtype=np.float32), res


def draw_landmarks(frame, res):
    if not res.hand_landmarks:
        return
    h, w = frame.shape[:2]
    lm = res.hand_landmarks[0]
    for conn in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(
            frame,
            (int(lm[conn.start].x * w), int(lm[conn.start].y * h)),
            (int(lm[conn.end].x * w),   int(lm[conn.end].y * h)),
            (50, 205, 50), 2,
        )
    for p in lm:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, (255, 255, 255), -1)



def draw_ui(frame, sentence, letter, conf, hand_detected, stab):
    h, w = frame.shape[:2]

    # ---------------- TOP SECTION ----------------

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 100), (59, 41, 30), -1)

    # Project title
    cv2.putText(
        frame,
        "ISL TRANSLATOR",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Accumulated sentence
    cv2.putText(
        frame,
        sentence or '(start signing...)',
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 200, 80),
        2
    )

    # ---------------- BOTTOM SECTION ----------------

    py = h - 140

    # Bottom panel background
    cv2.rectangle(frame, (0, py), (w, h), (56, 50, 38), -1)

    # Separator line
    cv2.line(frame, (0, py), (w, py), (90,90,90), 2)

    # Keyboard instructions
    controls_text = 'SPACE=space  BACK=delete  C=clear  Q=quit'

    text_width = cv2.getTextSize(
        controls_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )[0][0]

    cv2.putText(
        frame,
        controls_text,
        (w - text_width - 20, py + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # ---------------- HAND DETECTION ----------------

    if hand_detected:

        cv2.putText(
            frame,
            f'Hand Detected | Letter: {letter}',
            (20, py + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 205, 50),
            2
        )

        # Confidence bar
        bar_w = int(max(0, min(conf, 1)) * 300)

        bar_colour = (
            (50, 205, 50)
            if conf >= CONF_THRESH
            else (0, 60, 220)
        )

        cv2.putText(
            frame,
            'Confidence',
            (20, py + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1
        )

        cv2.rectangle(
            frame,
            (20, py + 65),
            (320, py + 80),
            (80, 80, 80),
            -1
        )

        cv2.rectangle(
            frame,
            (20, py + 65),
            (20 + bar_w, py + 80),
            bar_colour,
            -1
        )

        cv2.putText(
            frame,
            f'{conf * 100:.0f}%',
            (330, py + 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        # Stability indicators
        cv2.putText(
            frame,
            'Prediction Stability',
            (20, py + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1
        )

        for i in range(STAB_FRAMES):
            colour = (
                (50, 205, 50)
                if i < stab
                else (80, 80, 80)
            )

            cv2.circle(
                frame,
                (200 + i * 22, py + 100),
                7,
                colour,
                -1
            )

    else:

        cv2.putText(
            frame,
            'No Hand Detected',
            (20, py + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 80, 255),
            2
        )

        cv2.putText(
            frame,
            'Show your hand to start recognition',
            (20, py + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )

    
    # ---------------- MOUSE CLICK BUTTONS ----------------

    global BUTTON_RECTS

    buttons = ["SPACE", "BACK", "CLEAR", "QUIT"]

    btn_w = 145
    btn_h = 42
    gap = 12

    start_x = w - (4 * btn_w + 3 * gap) - 20
    btn_y = h - 55

    BUTTON_RECTS = {}

    for i, name in enumerate(buttons):

        x1 = start_x + i * (btn_w + gap)
        y1 = btn_y
        x2 = x1 + btn_w
        y2 = y1 + btn_h

        BUTTON_RECTS[name] = (x1, y1, x2, y2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (100, 100, 100),
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        text_size = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )[0]

        text_x = x1 + (btn_w - text_size[0]) // 2
        text_y = y1 + (btn_h + text_size[1]) // 2

        cv2.putText(
            frame,
            name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )



# Stores the button rectangles and clicked button
BUTTON_RECTS = {}
mouse_action = None


def handle_mouse(event, x, y, flags, param):
    global mouse_action

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse clicked:", x, y)

        for name, (x1, y1, x2, y2) in BUTTON_RECTS.items():

            if x1 <= x <= x2 and y1 <= y <= y2:
                print("Clicked button:", name)
                mouse_action = name
                break



def main():
    global mouse_action
    # Validate required files exist
    for path, name in [(MODEL_PATH, 'model'), (ENCODER_PATH, 'encoder'), (LANDMARKER, 'landmarker')]:
        if not os.path.exists(path):
            print(f"Missing required file: {path}")
            print("Run 1_collect_data.py → 2_train_model.py first.")
            return

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    enc   = joblib.load(ENCODER_PATH)
    print(f"Ready — {len(enc.classes_)} classes: {list(enc.classes_)}")

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
    )
    lm = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    seq        = collections.deque(maxlen=SEQ_LEN)
    pred_buf   = collections.deque(maxlen=STAB_FRAMES)
    sentence   = ''
    cur_letter = ''
    cur_conf   = 0.0
    last_t     = 0.0
    last_appended = None

    print("Running — show your hand.  SPACE / BACK / C=clear / Q=quit")
    cv2.namedWindow('ISL Real-Time')
    cv2.setMouseCallback('ISL Real-Time', handle_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        lmk, res = extract_landmarks(frame, lm)
        draw_landmarks(frame, res)
        hand_present = bool(res.hand_landmarks)
        stab = 0
        if hand_present:
            seq.append(lmk)

            if len(seq) == SEQ_LEN:
                probs      = model.predict(np.expand_dims(np.array(seq), 0), verbose=0)[0]
                idx        = np.argmax(probs)
                conf       = float(probs[idx])
                label      = enc.classes_[idx]
                cur_letter = label
                cur_conf   = conf

                pred_buf.append(label if conf >= CONF_THRESH else '')
                stab = sum(1 for p in pred_buf if p == cur_letter)

                now = time.time()
                if (stab >= STAB_FRAMES and conf >= CONF_THRESH
                        and label != last_appended and now - last_t > APPEND_DELAY):
                    sentence += label
                    last_t = now
                    last_appended = label
                    print(f"  → {label}  |  {sentence}")
        else:
            seq.clear()
            pred_buf.clear()
            cur_letter = ''
            cur_conf = 0.0
            last_t = 0.0
            last_appended = None

        draw_ui(frame, sentence, cur_letter, cur_conf,
                hand_present, stab)
        cv2.imshow('ISL Real-Time', frame)

        k = cv2.waitKey(1) & 0xFF

        # Get mouse button action
        action = mouse_action
        mouse_action = None

        # Keyboard controls
        if k in (ord('q'), 27):
            break

        elif k == ord(' '):
            action = 'SPACE'

        elif k in (8, 127):
            action = 'BACK'

        elif k == ord('c'):
            action = 'CLEAR'

        # Execute actions
        if action == 'SPACE':
            sentence += ' '

        elif action == 'BACK':
            sentence = sentence[:-1]

        elif action == 'CLEAR':
            sentence = ''

        elif action == 'QUIT':
            break

    cap.release()
    cv2.destroyAllWindows()
    lm.close()
    print(f"\nFinal sentence: {sentence}")


if __name__ == '__main__':
    main()
