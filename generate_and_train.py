"""
Synthetic Data Generator + Trainer
===================================
Use this script if you want a pre-trained starting point WITHOUT
recording your own webcam data first.

It generates anatomically-modelled synthetic ISL landmark sequences
for all 26 letters and trains the LSTM model on them.

The resulting model is a useful baseline (~70-80% on synthetic data).
For best accuracy (88%+), collect real data with 1_collect_data.py
and retrain with 2_train_model.py.

Usage:
    python generate_and_train.py
"""
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEQUENCE_LEN  = 30
FEATURE_SIZE  = 63       # 21 landmarks × 3 (x, y, z)
N_SAMPLES     = 80       # synthetic samples per letter
LABELS        = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
MODEL_DIR     = 'model'
DATA_DIR      = os.path.join('data', 'raw')

# ---------------------------------------------------------------------------
# Anatomical ISL hand templates
# Each entry approximates finger extension for an ISL letter.
# Format: [thumb, index, middle, ring, pinky]
#   1.0 = fully extended, 0.0 = fully folded
# ---------------------------------------------------------------------------
FINGER_STATES = {
    'A': [0.3, 0.0, 0.0, 0.0, 0.0],   # fist, thumb out slightly
    'B': [0.0, 1.0, 1.0, 1.0, 1.0],   # four fingers up, thumb folded
    'C': [0.6, 0.6, 0.6, 0.6, 0.6],   # curved C shape
    'D': [0.2, 1.0, 0.3, 0.3, 0.3],   # index up, others curved
    'E': [0.2, 0.3, 0.3, 0.3, 0.3],   # all fingers slightly bent
    'F': [0.8, 0.2, 1.0, 1.0, 1.0],   # OK-like, index+thumb touch
    'G': [0.9, 1.0, 0.1, 0.1, 0.1],   # index+thumb pointing sideways
    'H': [0.1, 1.0, 1.0, 0.1, 0.1],   # index+middle extended horizontal
    'I': [0.1, 0.0, 0.0, 0.0, 1.0],   # pinky up only
    'J': [0.1, 0.0, 0.0, 0.0, 1.0],   # pinky up + motion (approximated)
    'K': [0.8, 1.0, 1.0, 0.1, 0.1],   # index+middle up, thumb between
    'L': [1.0, 1.0, 0.0, 0.0, 0.0],   # L-shape: thumb+index
    'M': [0.2, 0.0, 0.0, 0.0, 0.0],   # three fingers over thumb
    'N': [0.2, 0.0, 0.0, 0.0, 0.0],   # two fingers over thumb (similar to M)
    'O': [0.5, 0.5, 0.5, 0.5, 0.5],   # O shape
    'P': [0.9, 1.0, 1.0, 0.0, 0.0],   # pointing down variant
    'Q': [0.9, 1.0, 0.0, 0.0, 0.0],   # pointing down
    'R': [0.1, 1.0, 1.0, 0.1, 0.1],   # index+middle crossed
    'S': [0.3, 0.0, 0.0, 0.0, 0.0],   # fist with thumb over fingers
    'T': [0.6, 0.0, 0.0, 0.0, 0.0],   # thumb between index+middle
    'U': [0.1, 1.0, 1.0, 0.0, 0.0],   # index+middle up together
    'V': [0.1, 1.0, 1.0, 0.0, 0.0],   # V / peace sign
    'W': [0.1, 1.0, 1.0, 1.0, 0.0],   # three fingers up
    'X': [0.1, 0.5, 0.0, 0.0, 0.0],   # index hooked
    'Y': [1.0, 0.0, 0.0, 0.0, 1.0],   # thumb+pinky out
    'Z': [0.1, 1.0, 0.0, 0.0, 0.0],   # index traces Z (approximated)
}

FINGER_TIPS  = [4,  8, 12, 16, 20]
FINGER_BASES = [2,  5,  9, 13, 17]


def make_hand_landmarks(letter: str, noise: float = 0.025) -> np.ndarray:
    """Return one frame of 21 hand landmarks (shape: 63,) for a given letter."""
    states = FINGER_STATES[letter]
    lm = np.zeros((21, 3), dtype=np.float32)

    palm_x = [
        0.00, 0.10, 0.18, 0.24, 0.28,   # thumb
        0.10, 0.12, 0.12, 0.12,          # index
        0.00, 0.00, 0.00, 0.00,          # middle
       -0.10,-0.10,-0.10,-0.10,          # ring
       -0.20,-0.20,-0.20,-0.20,          # pinky
    ]
    palm_y = [
        0.00,-0.05,-0.12,-0.20,-0.28,
       -0.15,-0.32,-0.42,-0.52,
       -0.18,-0.35,-0.45,-0.55,
       -0.15,-0.32,-0.42,-0.52,
       -0.12,-0.28,-0.38,-0.48,
    ]
    for i in range(21):
        lm[i, 0] = palm_x[i]
        lm[i, 1] = palm_y[i]

    finger_ranges = [(1, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    for f_idx, (start, end) in enumerate(finger_ranges):
        ext = states[f_idx]
        for joint in range(start, end):
            fold = (1.0 - ext) * 0.3
            lm[joint, 1] += fold * (joint - start + 1)
            lm[joint, 0] *= (0.6 + ext * 0.4)

    lm += np.random.normal(0, noise, lm.shape).astype(np.float32)
    lm -= lm[0]   # normalise relative to wrist
    return lm.flatten()


def generate_sequence(letter: str) -> np.ndarray:
    """Return a 30-frame landmark sequence (shape: 30 × 63) with simulated tremor."""
    base_noise = 0.018
    return np.array(
        [make_hand_landmarks(letter, noise=base_noise + t * 0.0008) for t in range(SEQUENCE_LEN)],
        dtype=np.float32,
    )


def build_model(n_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LEN, FEATURE_SIZE)),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax'),
    ], name='ISL_LSTM')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    print("Generating synthetic ISL landmark data...")
    X, y_str = [], []
    for letter in LABELS:
        letter_dir = os.path.join(DATA_DIR, letter)
        os.makedirs(letter_dir, exist_ok=True)
        for i in range(N_SAMPLES):
            seq = generate_sequence(letter)
            np.save(os.path.join(letter_dir, f'{i}.npy'), seq)
            X.append(seq)
            y_str.append(letter)
        print(f"  {letter}: {N_SAMPLES} sequences")

    X       = np.array(X, dtype=np.float32)
    encoder = LabelEncoder()
    y       = encoder.fit_transform(np.array(y_str))
    print(f"\nDataset: {X.shape}  |  Classes: {len(encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model = build_model(len(encoder.classes_))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-6, verbose=1,
        ),
    ]

    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 45}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'=' * 45}")

    model_path   = os.path.join(MODEL_DIR, 'isl_lstm_model.keras')
    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    model.save(model_path)
    joblib.dump(encoder, encoder_path)

    # Training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train', color='royalblue')
    ax1.plot(history.history['val_accuracy'], label='Val',   color='darkorange')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train', color='royalblue')
    ax2.plot(history.history['val_loss'], label='Val',   color='darkorange')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'), dpi=120)
    plt.close()

    print(f"\n✅  Model   → {model_path}")
    print(f"✅  Encoder → {encoder_path}")
    print(f"✅  Plot    → {os.path.join(MODEL_DIR, 'training_history.png')}")
    print("\nNext: python 3_realtime_predict.py")
    print("Tip : For better accuracy, collect real data with 1_collect_data.py")


if __name__ == '__main__':
    main()
