"""
Real-Time ASL Common Phrases Recognition
=========================================
Recognizes 15 common ASL word signs:
  hello, thank_you, please, yes, no, i_love_you,
  good, bad, help, more, eat, stop, sorry, wait, name

SETUP (one-time):
  1. Train the phrase model:
         python gesture_phrases_model.py
  2. Download the hand-landmarker if not already done:
         python realtime_demo.py --download

Then run:
    python realtime_phrases_demo.py

Controls:
  SPACE     - Confirm / log current prediction
  BACKSPACE - Delete last word
  ENTER     - Clear log
  Q / ESC   - Quit
"""

import cv2
import numpy as np
import pickle
import time
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gesture_phrases_model import prepare_features, predict_phrase

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL   = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
PHRASE_MODEL = os.path.join(SCRIPT_DIR, "models", "phrase_rf_model.pkl")

# ── CONFIG ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
PREDICTION_SMOOTHING = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Teal/dark colour scheme to distinguish from the alphabet demo
CLR_TEAL    = ( 180, 200,  50)
CLR_ORANGE  = (  50, 160, 230)
CLR_WHITE   = ( 240, 240, 240)
CLR_GRAY    = ( 100, 100, 110)
CLR_YELLOW  = (  30, 220, 230)
CLR_RED     = (  50,  50, 220)
CLR_OVERLAY = (  20,  30,  35)
CLR_GREEN   = (  50, 210, 120)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
FINGER_COLORS = [
    (80,  80, 255), (80, 200, 255), (80, 255, 160),
    (255,200,  80), (255,  80, 255),
]
CONN_COLOR_MAP = {
    **{c: FINGER_COLORS[0] for c in [(0,1),(1,2),(2,3),(3,4)]},
    **{c: FINGER_COLORS[1] for c in [(0,5),(5,6),(6,7),(7,8)]},
    **{c: FINGER_COLORS[2] for c in [(0,9),(9,10),(10,11),(11,12)]},
    **{c: FINGER_COLORS[3] for c in [(0,13),(13,14),(14,15),(15,16)]},
    **{c: FINGER_COLORS[4] for c in [(0,17),(17,18),(18,19),(19,20)]},
    **{c: (60, 70, 100)    for c in [(5,9),(9,13),(13,17)]},
}

# Human-readable display names
DISPLAY_NAMES = {
    'hello':      'Hello',
    'thank_you':  'Thank You',
    'please':     'Please',
    'yes':        'Yes',
    'no':         'No',
    'i_love_you': 'I Love You',
    'good':       'Good',
    'bad':        'Bad',
    'help':       'Help',
    'more':       'More',
    'eat':        'Eat / Food',
    'stop':       'Stop',
    'sorry':      'Sorry',
    'wait':       'Wait',
    'name':       'Name',
}


# ── Hand drawing ─────────────────────────────────────────────────
def draw_hand_landmarks(frame, landmarks_px):
    for (a, b) in HAND_CONNECTIONS:
        color = CONN_COLOR_MAP.get((a, b), (150, 150, 150))
        cv2.line(frame, landmarks_px[a], landmarks_px[b], color, 2)
    for i, (px, py) in enumerate(landmarks_px):
        is_tip = i in {4, 8, 12, 16, 20}
        r = 6 if is_tip else 3
        cv2.circle(frame, (px, py), r + 2, (255, 255, 255), -1)
        cv2.circle(frame, (px, py), r,     ( 30,  50,  60), -1)
        if is_tip:
            cv2.circle(frame, (px, py), r - 2, (180, 230, 255), -1)


# ── Prediction smoother ──────────────────────────────────────────
class PredictionSmoother:
    def __init__(self, window=8):
        self.window  = window
        self.history = []

    def update(self, phrase, conf):
        self.history.append((phrase, conf))
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_stable(self):
        if not self.history:
            return None, 0.0
        best = Counter(p for p, _ in self.history).most_common(1)[0][0]
        avg  = float(np.mean([c for p, c in self.history if p == best]))
        return best, avg


# ── UI ───────────────────────────────────────────────────────────
def draw_ui(frame, phrase, confidence, top3, log, fps):
    h, w = frame.shape[:2]

    # ── Top bar ──
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 70), CLR_OVERLAY, -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, "ASL Phrase Recognition",
                (15, 30), FONT, 0.7, CLR_TEAL, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 90, 25), FONT, 0.55, CLR_GRAY, 1)

    # ── Right panel ──
    px = w - 260
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px - 10, 75), (w - 5, 360), CLR_OVERLAY, -1)
    cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)

    color = CLR_GREEN if confidence >= CONFIDENCE_THRESHOLD else CLR_RED
    ctxt  = f"{confidence * 100:.0f}%" if confidence >= CONFIDENCE_THRESHOLD \
            else f"{confidence * 100:.0f}% (low)"

    display = DISPLAY_NAMES.get(phrase, phrase or "?") if phrase else "?"
    # Split long display names
    if len(display) > 10:
        parts = display.split()
        line1 = parts[0]
        line2 = " ".join(parts[1:]) if len(parts) > 1 else ""
    else:
        line1 = display
        line2 = ""

    cv2.putText(frame, "Sign:", (px, 105), FONT, 0.5, CLR_GRAY, 1)
    cv2.putText(frame, line1, (px + 5, 160), FONT, 1.5, color, 3)
    if line2:
        cv2.putText(frame, line2, (px + 5, 195), FONT, 1.5, color, 3)
    cv2.putText(frame, ctxt,
                (px, 215 if not line2 else 230), FONT, 0.5, color, 1)

    cv2.putText(frame, "Top 3:", (px, 255), FONT, 0.45, CLR_GRAY, 1)
    for i, (lbl, prob) in enumerate(top3):
        bw  = int((prob / 100) * 170)
        by  = 265 + i * 30
        dn  = DISPLAY_NAMES.get(lbl, lbl)[:14]
        cv2.rectangle(frame, (px, by), (px + 170, by + 18), (30, 40, 45), -1)
        cv2.rectangle(frame, (px, by), (px + bw,   by + 18),
                      CLR_TEAL if i == 0 else CLR_GRAY, -1)
        cv2.putText(frame, f"{dn}: {prob:.0f}%",
                    (px + 3, by + 13), FONT, 0.38, CLR_WHITE, 1)

    # ── Bottom log bar ──
    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, h - 115), (w, h), CLR_OVERLAY, -1)
    cv2.addWeighted(ov3, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, "Log:", (15, h - 88), FONT, 0.5, CLR_GRAY, 1)
    disp = (log if log else "_")
    # Wrap if too long
    words = disp.split()
    line = " ".join(words[-6:]) if len(words) > 6 else disp
    cv2.putText(frame, line, (15, h - 52), FONT, 0.9, CLR_YELLOW, 2)
    cv2.putText(frame,
                "[SPACE] Log  [BKSP] Del Word  [ENTER] Clear  [Q] Quit",
                (15, h - 15), FONT, 0.4, CLR_GRAY, 1)
    return frame


# ── Main ─────────────────────────────────────────────────────────
def run_realtime():
    # Load phrase classifier
    print("[*] Loading phrase classifier...")
    if not os.path.exists(PHRASE_MODEL):
        print(f"[!] Not found: {PHRASE_MODEL}")
        print("   Run:  python gesture_phrases_model.py")
        return
    with open(PHRASE_MODEL, "rb") as f:
        model_data = pickle.load(f)
    print(f"   Accuracy: {model_data['accuracy']*100:.1f}%")
    print(f"   Signs: {', '.join(model_data['labels'])}")

    # Load MediaPipe hand landmarker
    if not os.path.exists(HAND_MODEL) or os.path.getsize(HAND_MODEL) < 1_000_000:
        print("\n[!] Hand landmarker model not found.")
        print("   Run:  python realtime_demo.py --download")
        return

    print("[*] Loading MediaPipe hand landmarker...")
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
        print("[OK] Hand landmarker ready")
    except Exception as e:
        print(f"[!] Failed to load MediaPipe: {e}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[!] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoother  = PredictionSmoother(window=PREDICTION_SMOOTHING)
    log       = ""
    prev_time = time.time()

    print("\n[*] Camera running! Show ASL word signs.")
    print("   SPACE=log  BKSP=del word  ENTER=clear  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        phrase        = "?"
        confidence    = 0.0
        top3          = [("?", 0)] * 3
        hand_detected = False

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int(time.time() * 1000)
            result    = detector.detect_for_video(mp_image, timestamp)

            if result.hand_landmarks:
                hand_detected = True
                lm_list = result.hand_landmarks[0]

                lm_px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
                draw_hand_landmarks(frame, lm_px)

                lm_flat = np.array([[lm.x, lm.y, lm.z] for lm in lm_list]).flatten()
                phrase, confidence, top3 = predict_phrase(lm_flat, model_data)
                smoother.update(phrase, confidence)
                phrase, confidence = smoother.get_stable()

        except Exception as e:
            cv2.putText(frame, f"Detection error: {e}",
                        (15, h // 2 - 30), FONT, 0.5, CLR_RED, 1)

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if not hand_detected:
            phrase = "–"
            top3   = [("–", 0)] * 3

        frame = draw_ui(frame, phrase, confidence, top3, log, fps)

        if not hand_detected:
            cv2.putText(frame,
                        "No hand detected — show your hand clearly!",
                        (15, h // 2), FONT, 0.75, CLR_RED, 2)

        cv2.imshow("ASL Phrase Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            if phrase and phrase not in ("?", "–", None):
                display = DISPLAY_NAMES.get(phrase, phrase)
                log += (" " if log else "") + display
                print(f"   Logged '{display}' → '{log}'")
        elif key == 8:   # BACKSPACE — delete last word
            words = log.rsplit(' ', 1)
            log   = words[0] if len(words) > 1 else ""
        elif key == 13:  # ENTER — clear
            print(f"   Log: '{log}'")
            log = ""

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done. Final log: '{log}'")


if __name__ == "__main__":
    run_realtime()
