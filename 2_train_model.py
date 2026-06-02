"""
STEP 2 — Train LSTM Model
Loads landmark sequences from data/raw/, trains a 2-layer LSTM,
saves model + label encoder + training plot.

Usage: python 2_train_model.py [--epochs 60] [--batch_size 32]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR     = os.path.join('data', 'raw')
MODEL_DIR    = 'model'
MODEL_PATH   = os.path.join(MODEL_DIR, 'isl_lstm_model.keras')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
PLOT_PATH    = os.path.join(MODEL_DIR, 'training_history.png')
SEQ_LEN      = 30
FEAT         = 63


def load_dataset():
    X, y = [], []
    for label in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, label)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.npy')]
        for f in sorted(files):
            seq = np.load(os.path.join(d, f))
            if seq.shape == (SEQ_LEN, FEAT):
                X.append(seq)
                y.append(label)
        print(f"  {label}: {len(files)} samples")

    if not X:
        raise ValueError(
            f"No data found in '{DATA_DIR}'.\n"
            "Run 1_collect_data.py first, or use generate_and_train.py "
            "to bootstrap with synthetic data."
        )
    return np.array(X, dtype=np.float32), np.array(y)


def build_model(n_classes):
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, FEAT)),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64),
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
    parser = argparse.ArgumentParser(description="Train the ISL LSTM model")
    parser.add_argument('--epochs',     type=int,   default=60,   help='Max training epochs (default: 60)')
    parser.add_argument('--batch_size', type=int,   default=32,   help='Batch size (default: 32)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split fraction (default: 0.15)')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\nLoading data from '{DATA_DIR}'...")
    X, y_str = load_dataset()
    enc = LabelEncoder()
    y = enc.fit_transform(y_str)
    print(f"Total samples: {len(X)}  |  Classes: {len(enc.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )

    model = build_model(len(enc.classes_))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=6,
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=0,
        ),
    ]

    print(f"\nTraining for up to {args.epochs} epochs...")
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n{'=' * 45}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'=' * 45}")

    # Per-letter accuracy breakdown
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print("\nPer-letter accuracy:")
    for i, cls in enumerate(enc.classes_):
        mask = y_te == i
        if mask.sum() > 0:
            a = (preds[mask] == i).mean() * 100
            bar = '█' * int(a / 5)
            print(f"  {cls}: {a:5.1f}%  {bar}")

    # Save model and encoder
    model.save(MODEL_PATH)
    joblib.dump(enc, ENCODER_PATH)

    # Training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist.history['accuracy'],     label='Train', color='royalblue')
    ax1.plot(hist.history['val_accuracy'], label='Val',   color='darkorange')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(hist.history['loss'],     label='Train', color='royalblue')
    ax2.plot(hist.history['val_loss'], label='Val',   color='darkorange')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close()

    print(f"\n✅  Model   → {MODEL_PATH}")
    print(f"✅  Encoder → {ENCODER_PATH}")
    print(f"✅  Plot    → {PLOT_PATH}")
    print(f"\nNext: python 3_realtime_predict.py")


if __name__ == '__main__':
    main()
