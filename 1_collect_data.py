"""
STEP 1 — Data Collection
Run to collect ISL gesture data from webcam.
  SPACE = record sample  |  Q = skip letter

Output: data/raw/<LETTER>/<N>.npy  (shape: 30 x 63)
"""

import cv2
import numpy as np
import os
import argparse
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as mpImage, ImageFormat

LABELS        = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
SEQUENCE_LEN  = 30
FEATURE_SIZE  = 63   # 21 landmarks × 3 (x, y, z)
DATA_DIR      = os.path.join('data', 'raw')
MODEL_PATH    = os.path.join('model', 'hand_landmarker.task')


def download_model():
    os.makedirs('model', exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
        return True
    print("Downloading MediaPipe hand landmarker (~9 MB)...")
    import urllib.request
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    try:
        urllib.request.urlretrieve(url, MODEL_PATH)
        return True
    except Exception as e:
        print(f"Download failed: {e}\nSave model manually to {MODEL_PATH}")
        return False


def make_landmarker():
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def extract_landmarks(frame_bgr, lm):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = lm.detect(mpImage(image_format=ImageFormat.SRGB, data=rgb))
    if res.hand_landmarks:
        c = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
        c -= c[0]          # normalise relative to wrist
        return c.flatten(), res
    return np.zeros(FEATURE_SIZE, dtype=np.float32), res


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
            (0, 255, 0), 2,
        )
    for p in lm:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (255, 255, 255), -1)


def collect(cap, lm, label, label_idx, seqs, seq_len):
    save_dir = os.path.join(DATA_DIR, label)
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.endswith('.npy')])
    print(f"\n[{label_idx + 1}/26] {label}  (existing: {existing})")

    for i in range(existing, existing + seqs):
        # Wait for SPACE key to start recording
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            _, res = extract_landmarks(frame, lm)
            draw_landmarks(frame, res)
            cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Letter: {label}  Sample {i + 1}/{existing + seqs}  SPACE=record  Q=skip",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2,
            )
            cv2.imshow('ISL Collection', frame)
            k = cv2.waitKey(10) & 0xFF
            if k == ord(' '):
                break
            if k == ord('q'):
                return

        # Record one sequence of seq_len frames
        seq = []
        for _ in range(seq_len):
            ret, frame = cap.read()
            if not ret:
                seq.append(np.zeros(FEATURE_SIZE, dtype=np.float32))
                continue
            frame = cv2.flip(frame, 1)
            lmk, res = extract_landmarks(frame, lm)
            seq.append(lmk)
            draw_landmarks(frame, res)
            cv2.putText(frame, f"RECORDING {label}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('ISL Collection', frame)
            cv2.waitKey(1)

        np.save(os.path.join(save_dir, f'{i}.npy'), np.array(seq))
        print(f"  Saved sample {i + 1}")
        time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description="ISL gesture data collector")
    parser.add_argument('--sequences', type=int, default=30,
                        help='Number of samples to record per letter (default: 30)')
    parser.add_argument('--letters', type=str, default='',
                        help='Specific letters to collect, e.g. --letters ABC (default: all A-Z)')
    args = parser.parse_args()

    labels = list(args.letters.upper()) if args.letters else LABELS

    if not download_model():
        return

    lm = make_landmarker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        for i, label in enumerate(labels):
            collect(cap, lm, label, i, args.sequences, SEQUENCE_LEN)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        lm.close()

    print("\nDone! Run: python 2_train_model.py")


if __name__ == '__main__':
    main()
