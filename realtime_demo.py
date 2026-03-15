"""
Real-Time ASL Sign Language Recognition
========================================
Works with MediaPipe 0.10.30+ (Tasks API)

FIRST-TIME SETUP — run this once to download the hand model:
    python realtime_demo.py --download

Then run normally:
    python realtime_demo.py

Controls:
  SPACE     - Add predicted letter to sentence
  BACKSPACE - Delete last character
  ENTER     - Clear sentence
  Q / ESC   - Quit
"""

import cv2
import numpy as np
import pickle
import time
import os
import sys
import argparse
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sign_language_model import prepare_features, predict_sign

# ── Model file path ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL   = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ── CONFIG ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.50
PREDICTION_SMOOTHING = 7
FONT = cv2.FONT_HERSHEY_SIMPLEX

CLR_GREEN   = (50,  220, 100)
CLR_BLUE    = (220, 150, 50)
CLR_WHITE   = (240, 240, 240)
CLR_GRAY    = (100, 100, 110)
CLR_YELLOW  = (30,  220, 230)
CLR_RED     = (50,  50,  220)
CLR_OVERLAY = (25,  25,  40)

# Landmark connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
FINGER_COLORS = [
    (80, 80, 255), (80, 200, 255), (80, 255, 160),
    (255, 200, 80), (255, 80, 255),
]
CONN_COLOR_MAP = {
    **{c: FINGER_COLORS[0] for c in [(0,1),(1,2),(2,3),(3,4)]},
    **{c: FINGER_COLORS[1] for c in [(0,5),(5,6),(6,7),(7,8)]},
    **{c: FINGER_COLORS[2] for c in [(0,9),(9,10),(10,11),(11,12)]},
    **{c: FINGER_COLORS[3] for c in [(0,13),(13,14),(14,15),(15,16)]},
    **{c: FINGER_COLORS[4] for c in [(0,17),(17,18),(18,19),(19,20)]},
    **{c: (60, 70, 100)    for c in [(5,9),(9,13),(13,17)]},
}


# ── Download helper ─────────────────────────────────────────────
def download_model():
    os.makedirs(os.path.dirname(HAND_MODEL), exist_ok=True)
    if os.path.exists(HAND_MODEL) and os.path.getsize(HAND_MODEL) > 1_000_000:
        print(f"✅ Model already exists: {HAND_MODEL}")
        return True
    print(f"📥 Downloading hand landmarker model (~26MB)...")
    print(f"   From: {HAND_MODEL_URL}")
    try:
        def _progress(count, block, total):
            pct = min(count * block / total * 100, 100)
            print(f"\r   {pct:.1f}%", end="", flush=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL, _progress)
        print(f"\n✅ Saved to: {HAND_MODEL}")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("   Try manually downloading from:")
        print(f"   {HAND_MODEL_URL}")
        print(f"   and save it to: {HAND_MODEL}")
        return False


# ── Hand drawing ────────────────────────────────────────────────
def draw_hand_landmarks(frame, landmarks_px):
    """Draw hand skeleton on frame. landmarks_px = list of (x,y) pixel coords."""
    for (a, b) in HAND_CONNECTIONS:
        color = CONN_COLOR_MAP.get((a, b), (150, 150, 150))
        cv2.line(frame, landmarks_px[a], landmarks_px[b], color, 2)
    for i, (px, py) in enumerate(landmarks_px):
        is_tip = i in {4, 8, 12, 16, 20}
        r = 6 if is_tip else 3
        cv2.circle(frame, (px, py), r + 2, (255, 255, 255), -1)
        cv2.circle(frame, (px, py), r, (50, 50, 80), -1)
        if is_tip:
            cv2.circle(frame, (px, py), r - 2, (200, 220, 255), -1)


# ── UI ──────────────────────────────────────────────────────────
def draw_ui(frame, predicted_letter, confidence, top3, sentence, fps):
    h, w = frame.shape[:2]

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 70), CLR_OVERLAY, -1)
    cv2.addWeighted(ov, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "ASL Sign Language Recognition",
                (15, 30), FONT, 0.7, CLR_BLUE, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 90, 25), FONT, 0.55, CLR_GRAY, 1)

    # Right panel
    px = w - 235
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px - 10, 75), (w - 5, 345), CLR_OVERLAY, -1)
    cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)

    color = CLR_GREEN if confidence >= CONFIDENCE_THRESHOLD else CLR_RED
    ctxt  = f"{confidence*100:.0f}%" if confidence >= CONFIDENCE_THRESHOLD \
            else f"{confidence*100:.0f}% (low)"

    cv2.putText(frame, "Prediction:", (px, 105), FONT, 0.5, CLR_GRAY, 1)
    cv2.putText(frame, predicted_letter or "?",
                (px + 5, 178), FONT, 3.5, color, 5)
    cv2.putText(frame, ctxt, (px, 205), FONT, 0.5, color, 1)

    cv2.putText(frame, "Top 3:", (px, 235), FONT, 0.45, CLR_GRAY, 1)
    for i, (ltr, prob) in enumerate(top3):
        bw  = int((prob / 100) * 155)
        by  = 245 + i * 30
        cv2.rectangle(frame, (px, by), (px + 155, by + 18), (35, 35, 52), -1)
        cv2.rectangle(frame, (px, by), (px + bw,  by + 18),
                      CLR_BLUE if i == 0 else CLR_GRAY, -1)
        cv2.putText(frame, f"{ltr}: {prob:.0f}%",
                    (px + 3, by + 13), FONT, 0.42, CLR_WHITE, 1)

    # Bottom sentence bar
    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, h - 115), (w, h), CLR_OVERLAY, -1)
    cv2.addWeighted(ov3, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, "Sentence:", (15, h - 88), FONT, 0.5, CLR_GRAY, 1)
    disp = (sentence if sentence else "_")[-35:]
    cv2.putText(frame, disp, (15, h - 52), FONT, 1.0, CLR_YELLOW, 2)
    cv2.putText(frame,
                "[SPACE] Add  [BKSP] Delete  [ENTER] Clear  [Q] Quit",
                (15, h - 15), FONT, 0.4, CLR_GRAY, 1)
    return frame


# ── Prediction smoother ─────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window=7):
        self.window  = window
        self.history = []

    def update(self, letter, conf):
        self.history.append((letter, conf))
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_stable(self):
        if not self.history:
            return None, 0.0
        best = Counter(l for l, _ in self.history).most_common(1)[0][0]
        avg  = float(np.mean([c for l, c in self.history if l == best]))
        return best, avg


# ── Main ────────────────────────────────────────────────────────
def run_realtime():
    # Load classifier
    clf_path = os.path.join(SCRIPT_DIR, "models", "asl_rf_model.pkl")
    print("📦 Loading ASL classifier...")
    if not os.path.exists(clf_path):
        print(f"❌ Not found: {clf_path}")
        print("   Run sign_language_model.py first.")
        return
    with open(clf_path, "rb") as f:
        model_data = pickle.load(f)
    print(f"   Accuracy: {model_data['accuracy']*100:.1f}%")

    # Load MediaPipe Tasks hand landmarker
    if not os.path.exists(HAND_MODEL) or os.path.getsize(HAND_MODEL) < 1_000_000:
        print("\n⚠️  Hand landmarker model not found.")
        print("   Run:  python realtime_demo.py --download")
        return

    print("🤚 Loading MediaPipe hand landmarker...")
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("✅ Hand landmarker ready")
    except Exception as e:
        print(f"❌ Failed to load MediaPipe: {e}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try index 1 if 0 fails
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    smoother  = PredictionSmoother(window=PREDICTION_SMOOTHING)
    sentence  = ""
    prev_time = time.time()
    frame_idx = 0

    print("\n🎥 Camera running! Show ASL hand signs.")
    print("   SPACE=add  BKSP=delete  ENTER=clear  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        frame_idx += 1

        predicted_letter = "?"
        confidence       = 0.0
        top3             = [("?", 0)] * 3
        hand_detected    = False

        # Run detection (Tasks API needs timestamps in ms)
        try:
            rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp  = int(time.time() * 1000)
            result     = detector.detect_for_video(mp_image, timestamp)

            if result.hand_landmarks:
                hand_detected = True
                lm_list = result.hand_landmarks[0]  # first hand

                # Convert to pixel coords for drawing
                lm_px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
                draw_hand_landmarks(frame, lm_px)

                # Build flat array (x, y, z) × 21 for classifier
                lm_flat = np.array([[lm.x, lm.y, lm.z] for lm in lm_list]).flatten()
                predicted_letter, confidence, top3 = predict_sign(lm_flat, model_data)
                smoother.update(predicted_letter, confidence)
                predicted_letter, confidence = smoother.get_stable()

        except Exception as e:
            cv2.putText(frame, f"Detection error: {e}", (15, h // 2 - 30),
                        FONT, 0.5, CLR_RED, 1)

        # FPS
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if not hand_detected:
            predicted_letter = "–"
            top3 = [("–", 0)] * 3

        frame = draw_ui(frame, predicted_letter, confidence, top3, sentence, fps)

        if not hand_detected:
            cv2.putText(frame,
                        "No hand detected — show your hand clearly!",
                        (15, h // 2), FONT, 0.75, CLR_RED, 2)

        cv2.imshow("ASL Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            if predicted_letter not in ("?", "–", None):
                sentence += predicted_letter
                print(f"   Added '{predicted_letter}' → '{sentence}'")
        elif key == 8:
            sentence = sentence[:-1]
        elif key == 13:
            print(f"   Sentence: '{sentence}'")
            sentence = ""

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done. Final sentence: '{sentence}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download the MediaPipe hand landmarker model")
    args = parser.parse_args()

    if args.download:
        download_model()
    else:
        run_realtime()