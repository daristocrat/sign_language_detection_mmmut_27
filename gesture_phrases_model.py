"""
ASL Common Phrases / Word Sign Recognition
==========================================
Recognizes 15 common ASL word signs using MediaPipe hand landmarks + Random Forest.

Signs recognized:
  hello, thank_you, please, yes, no, i_love_you,
  good, bad, help, more, eat, stop, sorry, wait, name

Model: Random Forest on 21 hand landmarks (x,y,z) = 63 features + 10 angle features
Dataset: Synthetic landmark data per sign template

Run to train and save the model:
    python gesture_phrases_model.py
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
#  1. PHRASE LANDMARK TEMPLATES
# ─────────────────────────────────────────

def generate_phrases_dataset(samples_per_class=250, noise=0.018):
    """
    Synthetic ASL word-sign landmark dataset.
    Each sample = 21 landmarks × 3 coords (x,y,z) = 63 features.
    Landmarks: WRIST(0), THUMB(1-4), INDEX(5-8), MIDDLE(9-12), RING(13-16), PINKY(17-20)
    """
    np.random.seed(99)

    phrase_templates = {

        'hello': [  # Open hand — all 5 fingers spread wide (wave)
            [0.50, 0.80, 0],
            [0.33, 0.64, 0], [0.23, 0.57, 0], [0.15, 0.50, 0], [0.08, 0.43, 0],
            [0.42, 0.52, 0], [0.37, 0.38, 0], [0.33, 0.26, 0], [0.30, 0.16, 0],
            [0.50, 0.50, 0], [0.50, 0.36, 0], [0.50, 0.24, 0], [0.50, 0.14, 0],
            [0.58, 0.51, 0], [0.62, 0.37, 0], [0.65, 0.26, 0], [0.67, 0.16, 0],
            [0.64, 0.55, 0], [0.70, 0.43, 0], [0.74, 0.34, 0], [0.77, 0.26, 0],
        ],

        'thank_you': [  # B-hand from chin outward — all 4 fingers flat together, thumb lightly out
            [0.50, 0.80, 0],
            [0.38, 0.62, 0], [0.32, 0.55, 0], [0.30, 0.61, 0], [0.29, 0.66, 0],
            [0.44, 0.52, 0], [0.44, 0.39, 0], [0.44, 0.28, 0], [0.44, 0.20, 0],
            [0.50, 0.51, 0], [0.50, 0.38, 0], [0.50, 0.27, 0], [0.50, 0.19, 0],
            [0.56, 0.52, 0], [0.56, 0.39, 0], [0.56, 0.28, 0], [0.56, 0.20, 0],
            [0.62, 0.55, 0], [0.62, 0.43, 0], [0.62, 0.34, 0], [0.62, 0.27, 0],
        ],

        'please': [  # Flat hand on chest, palm inward — B-hand but mirrored thumb side
            [0.50, 0.80, 0],
            [0.62, 0.62, 0], [0.68, 0.55, 0], [0.70, 0.61, 0], [0.71, 0.66, 0],
            [0.57, 0.52, 0], [0.57, 0.39, 0], [0.57, 0.28, 0], [0.57, 0.20, 0],
            [0.51, 0.51, 0], [0.51, 0.38, 0], [0.51, 0.27, 0], [0.51, 0.19, 0],
            [0.45, 0.52, 0], [0.45, 0.39, 0], [0.45, 0.28, 0], [0.45, 0.20, 0],
            [0.39, 0.55, 0], [0.39, 0.43, 0], [0.39, 0.34, 0], [0.39, 0.27, 0],
        ],

        'yes': [  # Closed fist, wrist nod — tight fist, thumb beside index knuckle
            [0.50, 0.80, 0],
            [0.40, 0.60, 0], [0.36, 0.53, 0], [0.39, 0.59, 0], [0.41, 0.63, 0],
            [0.44, 0.52, 0], [0.44, 0.46, 0], [0.44, 0.52, 0], [0.44, 0.56, 0],
            [0.50, 0.51, 0], [0.50, 0.45, 0], [0.50, 0.51, 0], [0.50, 0.55, 0],
            [0.56, 0.52, 0], [0.56, 0.46, 0], [0.56, 0.52, 0], [0.56, 0.56, 0],
            [0.62, 0.54, 0], [0.62, 0.48, 0], [0.62, 0.53, 0], [0.62, 0.57, 0],
        ],

        'no': [  # Index + middle extended, spread (peace/scissors)
            [0.50, 0.80, 0],
            [0.40, 0.63, 0], [0.36, 0.57, 0], [0.38, 0.62, 0], [0.40, 0.66, 0],
            [0.43, 0.50, 0], [0.39, 0.37, 0], [0.36, 0.26, 0], [0.34, 0.17, 0],
            [0.51, 0.50, 0], [0.54, 0.37, 0], [0.56, 0.26, 0], [0.57, 0.17, 0],
            [0.56, 0.52, 0], [0.56, 0.46, 0], [0.56, 0.52, 0], [0.56, 0.56, 0],
            [0.62, 0.54, 0], [0.62, 0.48, 0], [0.62, 0.53, 0], [0.62, 0.57, 0],
        ],

        'i_love_you': [  # ILY: thumb + index + pinky extended, middle + ring curled
            [0.50, 0.80, 0],
            [0.33, 0.62, 0], [0.24, 0.56, 0], [0.16, 0.51, 0], [0.10, 0.46, 0],
            [0.44, 0.50, 0], [0.44, 0.37, 0], [0.44, 0.26, 0], [0.44, 0.17, 0],
            [0.50, 0.51, 0], [0.50, 0.45, 0], [0.50, 0.51, 0], [0.50, 0.55, 0],
            [0.56, 0.52, 0], [0.56, 0.46, 0], [0.56, 0.52, 0], [0.56, 0.56, 0],
            [0.62, 0.53, 0], [0.64, 0.42, 0], [0.66, 0.33, 0], [0.68, 0.26, 0],
        ],

        'good': [  # B-hand from chin — same 4 fingers up but thumb more open/angled
            [0.50, 0.80, 0],
            [0.36, 0.61, 0], [0.27, 0.53, 0], [0.24, 0.59, 0], [0.22, 0.64, 0],
            [0.43, 0.50, 0], [0.43, 0.37, 0], [0.43, 0.26, 0], [0.43, 0.18, 0],
            [0.49, 0.50, 0], [0.49, 0.37, 0], [0.49, 0.26, 0], [0.49, 0.18, 0],
            [0.55, 0.51, 0], [0.55, 0.38, 0], [0.55, 0.27, 0], [0.55, 0.19, 0],
            [0.61, 0.54, 0], [0.61, 0.42, 0], [0.61, 0.33, 0], [0.61, 0.26, 0],
        ],

        'bad': [  # Fingers from chin dropping down — B-hand flipped downward
            [0.50, 0.80, 0],
            [0.38, 0.63, 0], [0.32, 0.70, 0], [0.30, 0.75, 0], [0.29, 0.79, 0],
            [0.44, 0.62, 0], [0.44, 0.73, 0], [0.44, 0.82, 0], [0.44, 0.90, 0],
            [0.50, 0.61, 0], [0.50, 0.72, 0], [0.50, 0.81, 0], [0.50, 0.89, 0],
            [0.56, 0.62, 0], [0.56, 0.73, 0], [0.56, 0.82, 0], [0.56, 0.90, 0],
            [0.62, 0.65, 0], [0.62, 0.74, 0], [0.62, 0.82, 0], [0.62, 0.89, 0],
        ],

        'help': [  # Thumbs up — only thumb extended up, all fingers tightly curled
            [0.50, 0.80, 0],
            [0.46, 0.62, 0], [0.44, 0.51, 0], [0.43, 0.40, 0], [0.42, 0.30, 0],
            [0.47, 0.54, 0], [0.47, 0.48, 0], [0.47, 0.54, 0], [0.47, 0.58, 0],
            [0.52, 0.53, 0], [0.52, 0.47, 0], [0.52, 0.53, 0], [0.52, 0.57, 0],
            [0.57, 0.54, 0], [0.57, 0.48, 0], [0.57, 0.54, 0], [0.57, 0.58, 0],
            [0.62, 0.56, 0], [0.62, 0.50, 0], [0.62, 0.55, 0], [0.62, 0.59, 0],
        ],

        'more': [  # All fingertips pinched together — flattened O / bunch
            [0.50, 0.80, 0],
            [0.40, 0.56, 0], [0.36, 0.46, 0], [0.40, 0.38, 0], [0.44, 0.32, 0],
            [0.46, 0.50, 0], [0.44, 0.41, 0], [0.44, 0.35, 0], [0.44, 0.30, 0],
            [0.50, 0.49, 0], [0.50, 0.40, 0], [0.50, 0.34, 0], [0.50, 0.30, 0],
            [0.54, 0.50, 0], [0.55, 0.41, 0], [0.55, 0.35, 0], [0.55, 0.30, 0],
            [0.60, 0.52, 0], [0.61, 0.44, 0], [0.62, 0.38, 0], [0.63, 0.34, 0],
        ],

        'eat': [  # All fingertips together, pointing toward mouth — tighter pinch than 'more'
            [0.50, 0.80, 0],
            [0.42, 0.57, 0], [0.38, 0.47, 0], [0.41, 0.39, 0], [0.44, 0.32, 0],
            [0.47, 0.51, 0], [0.45, 0.41, 0], [0.44, 0.34, 0], [0.44, 0.28, 0],
            [0.51, 0.50, 0], [0.50, 0.40, 0], [0.50, 0.33, 0], [0.50, 0.27, 0],
            [0.55, 0.51, 0], [0.55, 0.41, 0], [0.55, 0.34, 0], [0.55, 0.28, 0],
            [0.60, 0.53, 0], [0.61, 0.44, 0], [0.62, 0.37, 0], [0.62, 0.31, 0],
        ],

        'stop': [  # Palm flat, held up — all fingers together, slightly spread from 'thank_you'
            [0.50, 0.80, 0],
            [0.36, 0.66, 0], [0.30, 0.60, 0], [0.33, 0.66, 0], [0.36, 0.71, 0],
            [0.42, 0.52, 0], [0.42, 0.39, 0], [0.42, 0.28, 0], [0.42, 0.20, 0],
            [0.48, 0.51, 0], [0.48, 0.38, 0], [0.48, 0.27, 0], [0.48, 0.19, 0],
            [0.54, 0.52, 0], [0.54, 0.39, 0], [0.54, 0.28, 0], [0.54, 0.20, 0],
            [0.60, 0.56, 0], [0.60, 0.44, 0], [0.60, 0.35, 0], [0.60, 0.28, 0],
        ],

        'sorry': [  # Fist with thumb draped over top of fingers (S-shape)
            [0.50, 0.80, 0],
            [0.40, 0.57, 0], [0.35, 0.49, 0], [0.38, 0.55, 0], [0.42, 0.57, 0],
            [0.45, 0.53, 0], [0.45, 0.47, 0], [0.45, 0.53, 0], [0.45, 0.57, 0],
            [0.50, 0.52, 0], [0.50, 0.46, 0], [0.50, 0.52, 0], [0.50, 0.56, 0],
            [0.55, 0.53, 0], [0.55, 0.47, 0], [0.55, 0.53, 0], [0.55, 0.57, 0],
            [0.61, 0.55, 0], [0.61, 0.49, 0], [0.61, 0.54, 0], [0.61, 0.58, 0],
        ],

        'wait': [  # All fingers spread, pointing downward
            [0.50, 0.80, 0],
            [0.36, 0.73, 0], [0.27, 0.77, 0], [0.22, 0.83, 0], [0.18, 0.88, 0],
            [0.43, 0.69, 0], [0.38, 0.79, 0], [0.36, 0.88, 0], [0.34, 0.96, 0],
            [0.49, 0.68, 0], [0.49, 0.79, 0], [0.49, 0.88, 0], [0.49, 0.96, 0],
            [0.55, 0.69, 0], [0.58, 0.79, 0], [0.59, 0.88, 0], [0.60, 0.96, 0],
            [0.61, 0.70, 0], [0.65, 0.79, 0], [0.67, 0.87, 0], [0.69, 0.94, 0],
        ],

        'name': [  # H-shape — index + middle extended sideways (pointing left)
            [0.50, 0.80, 0],
            [0.42, 0.64, 0], [0.42, 0.71, 0], [0.42, 0.76, 0], [0.42, 0.80, 0],
            [0.45, 0.54, 0], [0.35, 0.52, 0], [0.26, 0.51, 0], [0.18, 0.50, 0],
            [0.51, 0.54, 0], [0.41, 0.52, 0], [0.32, 0.51, 0], [0.24, 0.50, 0],
            [0.57, 0.54, 0], [0.57, 0.47, 0], [0.57, 0.53, 0], [0.57, 0.57, 0],
            [0.62, 0.56, 0], [0.62, 0.50, 0], [0.62, 0.55, 0], [0.62, 0.59, 0],
        ],
    }

    X_data, y_data = [], []
    labels = sorted(phrase_templates.keys())

    for label in labels:
        template = np.array(phrase_templates[label])   # (21, 3)
        for _ in range(samples_per_class):
            scale = np.random.uniform(0.85, 1.15)
            tx    = np.random.uniform(-0.05, 0.05)
            ty    = np.random.uniform(-0.05, 0.05)
            sample = template.copy()
            sample[:, 0] = sample[:, 0] * scale + tx
            sample[:, 1] = sample[:, 1] * scale + ty
            sample += np.random.normal(0, noise, sample.shape)
            X_data.append(sample.flatten())
            y_data.append(label)

    return np.array(X_data), np.array(y_data), labels


# ─────────────────────────────────────────
#  2. FEATURE ENGINEERING  (shared logic)
# ─────────────────────────────────────────

def normalize_landmarks(landmarks_flat):
    lm = landmarks_flat.reshape(21, 3)
    wrist = lm[0]
    lm_centered = lm - wrist
    scale = np.linalg.norm(lm_centered[9]) + 1e-6
    return (lm_centered / scale).flatten()


def extract_angle_features(landmarks_flat):
    lm = landmarks_flat.reshape(21, 3)
    angles = []
    finger_joints = [
        [1, 2, 3], [2, 3, 4],
        [5, 6, 7], [6, 7, 8],
        [9, 10, 11], [10, 11, 12],
        [13, 14, 15], [14, 15, 16],
        [17, 18, 19], [18, 19, 20],
    ]
    for a, b, c in finger_joints:
        v1 = lm[a] - lm[b]
        v2 = lm[c] - lm[b]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.clip(cos_a, -1, 1))
    return np.array(angles)


def prepare_features(X_raw):
    features = []
    for sample in X_raw:
        norm   = normalize_landmarks(sample)
        angles = extract_angle_features(sample)
        features.append(np.concatenate([norm, angles]))
    return np.array(features)


# ─────────────────────────────────────────
#  3. TRAIN
# ─────────────────────────────────────────

def train_model():
    print("=" * 60)
    print("  ASL Phrase Recognition - Training")
    print("=" * 60)

    print("\n[*] Generating phrase dataset (15 signs x 250 samples)...")
    X_raw, y, labels = generate_phrases_dataset(samples_per_class=250, noise=0.018)
    print(f"   Dataset: {X_raw.shape}  |  Classes: {len(labels)}")
    print(f"   Signs: {', '.join(labels)}")

    print("\n[*] Engineering features...")
    X = prepare_features(X_raw)
    print(f"   Feature vector: {X.shape[1]} (63 coords + 10 angles)")

    print("\n[*] Splitting 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[*] Training Random Forest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)

    print("\n[*] Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   [OK] Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_data = {
        'model':    model,
        'labels':   labels,
        'accuracy': acc,
    }
    with open("models/phrase_rf_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("   [OK] Saved: models/phrase_rf_model.pkl")
    return model, labels, acc


# ─────────────────────────────────────────
#  4. INFERENCE
# ─────────────────────────────────────────

def predict_phrase(landmarks_flat, model_data):
    """
    Predict common ASL phrase/word from 21 hand landmarks.

    Args:
        landmarks_flat : np.array (63,) — 21 × (x,y,z)
        model_data     : dict with 'model' and 'labels'

    Returns:
        (predicted_phrase, confidence, top3_list)
    """
    features = prepare_features([landmarks_flat])[0].reshape(1, -1)
    model    = model_data['model']
    labels   = model_data['labels']

    probs    = model.predict_proba(features)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    predicted  = labels[np.argmax(probs)]
    confidence = probs[np.argmax(probs)]
    top3 = [(labels[i], round(probs[i] * 100, 1)) for i in top3_idx]
    return predicted, confidence, top3


if __name__ == "__main__":
    train_model()
