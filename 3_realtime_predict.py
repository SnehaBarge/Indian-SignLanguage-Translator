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

    # Top bar — current sentence
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        sentence or '(start signing...)',
        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2,
    )

    # Bottom panel — prediction info
    py = h - 110
    cv2.rectangle(frame, (0, py), (w, h), (60, 60, 60), -1)

    if hand_detected:
        cv2.putText(frame, f'Detected: {letter}', (10, py + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 205, 50), 2)

        # Confidence bar
        bar_w = int(conf * 300)
        bar_colour = (50, 205, 50) if conf >= CONF_THRESH else (0, 60, 220)
        cv2.rectangle(frame, (10, py + 50), (310, py + 70), (80, 80, 80), -1)
        cv2.rectangle(frame, (10, py + 50), (10 + bar_w, py + 70), bar_colour, -1)
        cv2.putText(frame, f'{conf * 100:.0f}%', (320, py + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Stability dots
        for i in range(STAB_FRAMES):
            colour = (50, 205, 50) if i < stab else (80, 80, 80)
            cv2.circle(frame, (10 + i * 22, py + 95), 8, colour, -1)
    else:
        cv2.putText(frame, 'No hand detected', (10, py + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 60, 220), 2)

    cv2.putText(
        frame,
        'SPACE=space  BACK=delete  C=clear  Q=quit',
        (w - 510, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )


def main():
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
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            sentence += ' '
        elif k == 8:
            sentence = sentence[:-1]
        elif k == ord('c'):
            sentence = ''

    cap.release()
    cv2.destroyAllWindows()
    lm.close()
    print(f"\nFinal sentence: {sentence}")


if __name__ == '__main__':
    main()
