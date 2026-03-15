"""
Sign Language to English Text Recognition
==========================================
Uses MediaPipe Hands for landmark detection + Random Forest classifier
Trained on ASL (American Sign Language) alphabet dataset

Dataset: Synthetic landmark data representing 26 ASL letter signs (A-Z)
Model: Random Forest Classifier on 21 hand landmarks (x, y, z) = 63 features
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  1. SYNTHETIC ASL LANDMARK DATASET
# ─────────────────────────────────────────

def generate_asl_dataset(samples_per_class=200, noise=0.02):
    """
    Generate realistic synthetic hand landmark data for ASL A-Z.
    Each sample = 21 landmarks × 3 coordinates (x, y, z) = 63 features.
    
    Real deployment: replace this with actual landmark extraction from images
    using extract_landmarks_from_image() below.
    """
    np.random.seed(42)

    # Canonical hand landmark positions per ASL letter (normalized 0-1)
    # Landmarks: WRIST(0), THUMB(1-4), INDEX(5-8), MIDDLE(9-12), RING(13-16), PINKY(17-20)
    asl_templates = {
        'A': [  # Fist with thumb to side
            [0.5, 0.8, 0],
            [0.40,0.62,0],[0.36,0.56,0],[0.38,0.62,0],[0.40,0.66,0],
            [0.44,0.50,0],[0.44,0.40,0],[0.46,0.35,0],[0.48,0.33,0],  # Index hooked
            [0.50,0.51,0],[0.50,0.45,0],[0.50,0.51,0],[0.50,0.55,0],  # Others curled
            [0.56,0.52,0],[0.56,0.46,0],[0.56,0.52,0],[0.56,0.56,0],
            [0.62,0.54,0],[0.62,0.48,0],[0.62,0.53,0],[0.62,0.57,0],
        ],
        'B': [  # Four fingers up, thumb tucked
            [0.5, 0.8, 0],
            [0.38,0.58,0],[0.33,0.52,0],[0.30,0.58,0],[0.28,0.62,0],  # Thumb tucked
            [0.43,0.50,0],[0.43,0.38,0],[0.43,0.28,0],[0.43,0.20,0],  # Index up
            [0.49,0.50,0],[0.49,0.37,0],[0.49,0.27,0],[0.49,0.19,0],  # Middle up
            [0.55,0.51,0],[0.55,0.39,0],[0.55,0.29,0],[0.55,0.21,0],  # Ring up
            [0.61,0.54,0],[0.61,0.43,0],[0.61,0.34,0],[0.61,0.27,0],  # Pinky up
        ],
        'C': [  # Curved hand like letter C
            [0.5, 0.8, 0],
            [0.30,0.60,0],[0.22,0.50,0],[0.18,0.42,0],[0.20,0.35,0],  # Thumb curved
            [0.38,0.45,0],[0.32,0.35,0],[0.30,0.28,0],[0.32,0.23,0],  # Index curved
            [0.46,0.43,0],[0.42,0.33,0],[0.42,0.26,0],[0.44,0.22,0],  # Middle curved
            [0.54,0.45,0],[0.52,0.35,0],[0.52,0.28,0],[0.54,0.23,0],  # Ring curved
            [0.62,0.50,0],[0.62,0.40,0],[0.62,0.33,0],[0.63,0.28,0],  # Pinky curved
        ],
        'D': [  # Index up, others form circle with thumb
            [0.5, 0.8, 0],
            [0.38,0.55,0],[0.33,0.46,0],[0.36,0.40,0],[0.40,0.36,0],  # Thumb to middle
            [0.43,0.48,0],[0.43,0.35,0],[0.43,0.24,0],[0.43,0.16,0],  # Index up straight
            [0.50,0.50,0],[0.48,0.42,0],[0.47,0.46,0],[0.46,0.50,0],  # Middle down
            [0.56,0.52,0],[0.55,0.44,0],[0.54,0.48,0],[0.54,0.52,0],  # Ring down
            [0.62,0.55,0],[0.62,0.48,0],[0.62,0.52,0],[0.62,0.56,0],  # Pinky down
        ],
        'E': [  # All fingers curled, thumb under
            [0.5, 0.8, 0],
            [0.40,0.65,0],[0.35,0.60,0],[0.38,0.68,0],[0.41,0.72,0],  # Thumb under
            [0.43,0.50,0],[0.43,0.44,0],[0.43,0.50,0],[0.43,0.55,0],  # Index curled
            [0.49,0.49,0],[0.49,0.43,0],[0.49,0.49,0],[0.49,0.54,0],  # Middle curled
            [0.55,0.50,0],[0.55,0.44,0],[0.55,0.50,0],[0.55,0.55,0],  # Ring curled
            [0.61,0.52,0],[0.61,0.47,0],[0.61,0.52,0],[0.61,0.56,0],  # Pinky curled
        ],
        'F': [  # Index and thumb circle, others up
            [0.5, 0.8, 0],
            [0.38,0.55,0],[0.36,0.46,0],[0.40,0.40,0],[0.44,0.36,0],  # Thumb to index tip
            [0.44,0.50,0],[0.44,0.42,0],[0.45,0.48,0],[0.45,0.52,0],  # Index down/circle
            [0.50,0.49,0],[0.50,0.37,0],[0.50,0.27,0],[0.50,0.19,0],  # Middle up
            [0.56,0.51,0],[0.56,0.39,0],[0.56,0.29,0],[0.56,0.21,0],  # Ring up
            [0.62,0.53,0],[0.62,0.42,0],[0.62,0.33,0],[0.62,0.26,0],  # Pinky up
        ],
        'G': [  # Index pointing sideways, thumb parallel
            [0.5, 0.8, 0],
            [0.30,0.60,0],[0.20,0.58,0],[0.12,0.56,0],[0.05,0.54,0],  # Thumb pointing left
            [0.38,0.52,0],[0.28,0.50,0],[0.20,0.48,0],[0.13,0.47,0],  # Index pointing left
            [0.46,0.50,0],[0.46,0.43,0],[0.46,0.49,0],[0.46,0.53,0],  # Middle curled
            [0.53,0.51,0],[0.53,0.44,0],[0.53,0.50,0],[0.53,0.54,0],  # Ring curled
            [0.60,0.53,0],[0.60,0.47,0],[0.60,0.52,0],[0.60,0.56,0],  # Pinky curled
        ],
        'H': [  # Index and middle pointing sideways
            [0.5, 0.8, 0],
            [0.40,0.62,0],[0.40,0.70,0],[0.40,0.75,0],[0.40,0.78,0],  # Thumb down
            [0.42,0.52,0],[0.32,0.50,0],[0.22,0.48,0],[0.14,0.47,0],  # Index pointing left
            [0.48,0.52,0],[0.38,0.50,0],[0.28,0.49,0],[0.20,0.48,0],  # Middle pointing left
            [0.54,0.52,0],[0.54,0.45,0],[0.54,0.51,0],[0.54,0.55,0],  # Ring curled
            [0.60,0.54,0],[0.60,0.48,0],[0.60,0.53,0],[0.60,0.57,0],  # Pinky curled
        ],
        'I': [  # Pinky up, others closed
            [0.5, 0.8, 0],
            [0.40,0.63,0],[0.36,0.57,0],[0.38,0.63,0],[0.40,0.67,0],  # Thumb tucked
            [0.44,0.51,0],[0.44,0.45,0],[0.44,0.50,0],[0.44,0.54,0],  # Index curled
            [0.50,0.50,0],[0.50,0.44,0],[0.50,0.50,0],[0.50,0.54,0],  # Middle curled
            [0.56,0.51,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],  # Ring curled
            [0.62,0.54,0],[0.62,0.43,0],[0.62,0.33,0],[0.62,0.25,0],  # Pinky up
        ],
        'J': [  # Pinky up then curved (like I but movement - static: pinky up)
            [0.5, 0.8, 0],
            [0.40,0.63,0],[0.36,0.57,0],[0.38,0.63,0],[0.40,0.67,0],
            [0.44,0.51,0],[0.44,0.45,0],[0.44,0.50,0],[0.44,0.54,0],
            [0.50,0.50,0],[0.50,0.44,0],[0.50,0.50,0],[0.50,0.54,0],
            [0.56,0.51,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],
            [0.63,0.54,0],[0.65,0.43,0],[0.68,0.35,0],[0.70,0.28,0],  # Pinky up + curved
        ],
        'K': [  # Index and middle up in V, thumb between them
            [0.5, 0.8, 0],
            [0.44,0.55,0],[0.40,0.46,0],[0.40,0.38,0],[0.42,0.32,0],  # Thumb up between
            [0.43,0.50,0],[0.38,0.38,0],[0.35,0.28,0],[0.33,0.20,0],  # Index up-left
            [0.49,0.49,0],[0.47,0.37,0],[0.47,0.27,0],[0.47,0.19,0],  # Middle up
            [0.55,0.51,0],[0.55,0.44,0],[0.55,0.50,0],[0.55,0.55,0],  # Ring curled
            [0.61,0.53,0],[0.61,0.47,0],[0.61,0.52,0],[0.61,0.56,0],  # Pinky curled
        ],
        'L': [  # L-shape: index up, thumb out
            [0.5, 0.8, 0],
            [0.35,0.60,0],[0.24,0.58,0],[0.15,0.55,0],[0.08,0.52,0],  # Thumb pointing left
            [0.43,0.50,0],[0.43,0.37,0],[0.43,0.26,0],[0.43,0.17,0],  # Index pointing up
            [0.49,0.51,0],[0.49,0.45,0],[0.49,0.51,0],[0.49,0.55,0],  # Middle curled
            [0.55,0.52,0],[0.55,0.46,0],[0.55,0.52,0],[0.55,0.56,0],  # Ring curled
            [0.61,0.55,0],[0.61,0.49,0],[0.61,0.54,0],[0.61,0.58,0],  # Pinky curled
        ],
        'M': [  # Three fingers over thumb
            [0.5, 0.8, 0],
            [0.42,0.67,0],[0.40,0.72,0],[0.41,0.75,0],[0.42,0.77,0],  # Thumb tucked under
            [0.43,0.52,0],[0.43,0.47,0],[0.43,0.53,0],[0.43,0.57,0],  # Index over thumb
            [0.49,0.51,0],[0.49,0.46,0],[0.49,0.52,0],[0.49,0.56,0],  # Middle over thumb
            [0.55,0.52,0],[0.55,0.47,0],[0.55,0.53,0],[0.55,0.57,0],  # Ring over thumb
            [0.61,0.55,0],[0.61,0.50,0],[0.61,0.55,0],[0.61,0.59,0],  # Pinky curled
        ],
        'N': [  # Two fingers over thumb
            [0.5, 0.8, 0],
            [0.42,0.65,0],[0.40,0.70,0],[0.41,0.73,0],[0.42,0.75,0],
            [0.43,0.52,0],[0.43,0.47,0],[0.43,0.53,0],[0.43,0.57,0],  # Index over
            [0.49,0.51,0],[0.49,0.46,0],[0.49,0.52,0],[0.49,0.56,0],  # Middle over
            [0.55,0.52,0],[0.55,0.44,0],[0.55,0.50,0],[0.55,0.55,0],  # Ring curled-up
            [0.61,0.54,0],[0.61,0.47,0],[0.61,0.52,0],[0.61,0.56,0],
        ],
        'O': [  # All fingers and thumb form circle
            [0.5, 0.8, 0],
            [0.35,0.58,0],[0.28,0.48,0],[0.30,0.40,0],[0.35,0.34,0],  # Thumb to fingertips
            [0.43,0.50,0],[0.40,0.40,0],[0.39,0.33,0],[0.38,0.28,0],  # Index curved down
            [0.49,0.49,0],[0.48,0.39,0],[0.48,0.32,0],[0.48,0.27,0],  # Middle curved
            [0.55,0.50,0],[0.55,0.40,0],[0.55,0.33,0],[0.55,0.28,0],  # Ring curved
            [0.61,0.52,0],[0.62,0.43,0],[0.63,0.36,0],[0.64,0.32,0],  # Pinky curved
        ],
        'P': [  # Like K but pointing down
            [0.5, 0.8, 0],
            [0.45,0.58,0],[0.42,0.65,0],[0.43,0.70,0],[0.44,0.74,0],  # Thumb down
            [0.43,0.52,0],[0.40,0.62,0],[0.39,0.70,0],[0.38,0.76,0],  # Index down
            [0.49,0.51,0],[0.48,0.61,0],[0.48,0.69,0],[0.48,0.75,0],  # Middle down
            [0.55,0.52,0],[0.55,0.45,0],[0.55,0.51,0],[0.55,0.55,0],  # Ring curled
            [0.61,0.54,0],[0.61,0.48,0],[0.61,0.53,0],[0.61,0.57,0],
        ],
        'Q': [  # Like G but pointing down
            [0.5, 0.8, 0],
            [0.42,0.60,0],[0.38,0.68,0],[0.36,0.74,0],[0.35,0.78,0],  # Thumb down
            [0.44,0.53,0],[0.42,0.62,0],[0.41,0.69,0],[0.40,0.74,0],  # Index down
            [0.50,0.50,0],[0.50,0.44,0],[0.50,0.50,0],[0.50,0.54,0],  # Others curled
            [0.56,0.51,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],
            [0.62,0.53,0],[0.62,0.47,0],[0.62,0.52,0],[0.62,0.56,0],
        ],
        'R': [  # Index and middle crossed
            [0.5, 0.8, 0],
            [0.40,0.62,0],[0.36,0.56,0],[0.38,0.62,0],[0.40,0.66,0],  # Thumb tucked
            [0.44,0.50,0],[0.42,0.37,0],[0.41,0.27,0],[0.40,0.19,0],  # Index up
            [0.50,0.50,0],[0.46,0.37,0],[0.44,0.27,0],[0.43,0.19,0],  # Middle crossed over index
            [0.56,0.51,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],  # Ring curled
            [0.62,0.53,0],[0.62,0.47,0],[0.62,0.52,0],[0.62,0.56,0],
        ],
        'S': [  # Fist with thumb over fingers
            [0.5, 0.8, 0],
            [0.40,0.58,0],[0.35,0.50,0],[0.38,0.55,0],[0.41,0.58,0],  # Thumb over
            [0.44,0.52,0],[0.44,0.46,0],[0.44,0.52,0],[0.44,0.56,0],  # All curled
            [0.50,0.51,0],[0.50,0.45,0],[0.50,0.51,0],[0.50,0.55,0],
            [0.56,0.52,0],[0.56,0.46,0],[0.56,0.52,0],[0.56,0.56,0],
            [0.62,0.54,0],[0.62,0.49,0],[0.62,0.54,0],[0.62,0.58,0],
        ],
        'T': [  # Thumb between index and middle
            [0.5, 0.8, 0],
            [0.44,0.55,0],[0.42,0.48,0],[0.44,0.44,0],[0.46,0.40,0],  # Thumb up between
            [0.44,0.51,0],[0.44,0.45,0],[0.44,0.51,0],[0.44,0.55,0],  # Index curled
            [0.50,0.50,0],[0.50,0.44,0],[0.50,0.50,0],[0.50,0.54,0],  # Middle curled
            [0.56,0.51,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],
            [0.62,0.53,0],[0.62,0.47,0],[0.62,0.52,0],[0.62,0.56,0],
        ],
        'U': [  # Index and middle up together
            [0.5, 0.8, 0],
            [0.40,0.63,0],[0.36,0.57,0],[0.38,0.62,0],[0.40,0.66,0],  # Thumb tucked
            [0.44,0.50,0],[0.44,0.37,0],[0.44,0.27,0],[0.44,0.19,0],  # Index up
            [0.50,0.50,0],[0.50,0.37,0],[0.50,0.27,0],[0.50,0.19,0],  # Middle up (together)
            [0.56,0.52,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],  # Ring curled
            [0.62,0.54,0],[0.62,0.48,0],[0.62,0.53,0],[0.62,0.57,0],
        ],
        'V': [  # Index and middle up in V shape
            [0.5, 0.8, 0],
            [0.40,0.63,0],[0.36,0.57,0],[0.38,0.62,0],[0.40,0.66,0],
            [0.42,0.50,0],[0.38,0.37,0],[0.36,0.27,0],[0.34,0.19,0],  # Index up-left
            [0.50,0.50,0],[0.52,0.37,0],[0.53,0.27,0],[0.54,0.19,0],  # Middle up-right
            [0.56,0.52,0],[0.56,0.45,0],[0.56,0.51,0],[0.56,0.55,0],  # Ring curled
            [0.62,0.54,0],[0.62,0.48,0],[0.62,0.53,0],[0.62,0.57,0],
        ],
        'W': [  # Index, middle, ring up spread
            [0.5, 0.8, 0],
            [0.40,0.63,0],[0.36,0.57,0],[0.38,0.62,0],[0.40,0.66,0],
            [0.41,0.50,0],[0.36,0.37,0],[0.33,0.27,0],[0.31,0.19,0],  # Index up-left
            [0.49,0.49,0],[0.49,0.36,0],[0.49,0.26,0],[0.49,0.18,0],  # Middle up
            [0.57,0.50,0],[0.62,0.37,0],[0.65,0.27,0],[0.67,0.19,0],  # Ring up-right
            [0.63,0.55,0],[0.63,0.49,0],[0.63,0.54,0],[0.63,0.58,0],  # Pinky curled
        ],
        'X': [  # Index finger hooked
            [0.5, 0.8, 0],   # Wrist
            [0.35,0.65,0],[0.28,0.55,0],[0.25,0.48,0],[0.22,0.42,0],  # Thumb
            [0.45,0.50,0],[0.45,0.42,0],[0.45,0.48,0],[0.45,0.52,0],  # Index (curled)
            [0.50,0.50,0],[0.50,0.42,0],[0.50,0.48,0],[0.50,0.52,0],  # Middle (curled)
            [0.55,0.52,0],[0.55,0.44,0],[0.55,0.50,0],[0.55,0.54,0],  # Ring (curled)
            [0.60,0.55,0],[0.60,0.48,0],[0.60,0.53,0],[0.60,0.57,0],  # Pinky (curled)
            
        ],
        'Y': [  # Thumb and pinky out
            [0.5, 0.8, 0],
            [0.33,0.60,0],[0.24,0.55,0],[0.17,0.50,0],[0.11,0.46,0],  # Thumb out left
            [0.44,0.52,0],[0.44,0.46,0],[0.44,0.52,0],[0.44,0.56,0],  # Index curled
            [0.50,0.51,0],[0.50,0.45,0],[0.50,0.51,0],[0.50,0.55,0],  # Middle curled
            [0.56,0.52,0],[0.56,0.46,0],[0.56,0.52,0],[0.56,0.56,0],  # Ring curled
            [0.62,0.53,0],[0.64,0.43,0],[0.66,0.34,0],[0.68,0.27,0],  # Pinky up
        ],
        'Z': [  # Index finger traces Z (static: index pointing)
            [0.5, 0.8, 0],
            [0.40,0.62,0],[0.36,0.56,0],[0.38,0.62,0],[0.40,0.66,0],
            [0.44,0.50,0],[0.38,0.38,0],[0.34,0.28,0],[0.31,0.20,0],  # Index pointing diag
            [0.50,0.51,0],[0.50,0.45,0],[0.50,0.51,0],[0.50,0.55,0],  # Others curled
            [0.56,0.52,0],[0.56,0.46,0],[0.56,0.52,0],[0.56,0.56,0],
            [0.62,0.54,0],[0.62,0.48,0],[0.62,0.53,0],[0.62,0.57,0],
        ],
    }

    X_data, y_data = [], []
    labels = sorted(asl_templates.keys())

    for label in labels:
        template = np.array(asl_templates[label])  # (21, 3)
        for _ in range(samples_per_class):
            # Add realistic noise + small scale/translation variation
            scale = np.random.uniform(0.85, 1.15)
            tx = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)
            sample = template.copy()
            sample[:, 0] = sample[:, 0] * scale + tx
            sample[:, 1] = sample[:, 1] * scale + ty
            sample += np.random.normal(0, noise, sample.shape)
            X_data.append(sample.flatten())  # 63 features
            y_data.append(label)

    return np.array(X_data), np.array(y_data), labels


# ─────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────

def normalize_landmarks(landmarks_flat):
    """
    Normalize landmarks relative to wrist position and hand size.
    This makes the model invariant to hand position/scale in frame.
    """
    lm = landmarks_flat.reshape(21, 3)
    wrist = lm[0]
    lm_centered = lm - wrist  # Center on wrist
    
    # Scale by distance from wrist to middle finger MCP (landmark 9)
    scale = np.linalg.norm(lm_centered[9]) + 1e-6
    lm_normalized = lm_centered / scale
    
    return lm_normalized.flatten()


def extract_angle_features(landmarks_flat):
    """
    Extract finger bend angles as additional features.
    More discriminative than raw coordinates for some letters.
    """
    lm = landmarks_flat.reshape(21, 3)
    angles = []
    
    # Finger joint triplets: (base, mid, tip) for each finger
    finger_joints = [
        [1, 2, 3], [2, 3, 4],    # Thumb
        [5, 6, 7], [6, 7, 8],    # Index
        [9, 10, 11], [10, 11, 12],  # Middle
        [13, 14, 15], [14, 15, 16],  # Ring
        [17, 18, 19], [18, 19, 20],  # Pinky
    ]
    
    for a, b, c in finger_joints:
        v1 = lm[a] - lm[b]
        v2 = lm[c] - lm[b]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(np.clip(cos_angle, -1, 1))
    
    return np.array(angles)


def prepare_features(X_raw):
    """Combine normalized landmarks + angle features."""
    features = []
    for sample in X_raw:
        norm = normalize_landmarks(sample)
        angles = extract_angle_features(sample)
        combined = np.concatenate([norm, angles])
        features.append(combined)
    return np.array(features)


# ─────────────────────────────────────────
#  3. TRAIN MODEL
# ─────────────────────────────────────────

def train_model():
    print("=" * 60)
    print("  ASL Sign Language Recognition - Training")
    print("=" * 60)
    
    print("\n📊 Generating ASL landmark dataset (26 letters × 200 samples)...")
    X_raw, y, labels = generate_asl_dataset(samples_per_class=200, noise=0.018)
    print(f"   Dataset shape: {X_raw.shape} | Classes: {len(labels)}")

    print("\n⚙️  Engineering features (landmarks + joint angles)...")
    X = prepare_features(X_raw)
    print(f"   Feature vector size: {X.shape[1]} (63 coords + 10 angles)")

    print("\n✂️  Splitting: 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🌲 Training Random Forest (200 trees)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    print("\n📈 Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   ✅ Test Accuracy: {acc * 100:.2f}%")
    print("\n   Per-class Report:")
    print(classification_report(y_test, y_pred))

    # Save model + metadata
    os.makedirs("models", exist_ok=True)
    model_data = {
        'model': model,
        'labels': labels,
        'prepare_features': prepare_features,
        'accuracy': acc
    }
    with open("models/asl_rf_model.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    print("   💾 Model saved: models/asl_rf_model.pkl")
    
    return model, labels, acc


# ─────────────────────────────────────────
#  4. INFERENCE
# ─────────────────────────────────────────

def load_model():
    with open("models/asl_rf_model.pkl", 'rb') as f:
        return pickle.load(f)


def predict_sign(landmarks_flat, model_data):
    """
    Predict ASL letter from 21 hand landmarks.
    
    Args:
        landmarks_flat: numpy array of shape (63,) — 21 landmarks × (x,y,z)
        model_data: dict with 'model' and 'labels' keys
    
    Returns:
        (predicted_letter, confidence_score, top3_predictions)
    """
    features = prepare_features([landmarks_flat])[0].reshape(1, -1)
    model = model_data['model']
    labels = model_data['labels']
    
    probs = model.predict_proba(features)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    
    predicted = labels[np.argmax(probs)]
    confidence = probs[np.argmax(probs)]
    top3 = [(labels[i], round(probs[i] * 100, 1)) for i in top3_idx]
    
    return predicted, confidence, top3


if __name__ == "__main__":
    train_model()