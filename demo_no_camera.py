"""
Demo without Camera — Test the model on all 26 ASL letters
Run: python demo_no_camera.py
"""

import numpy as np
import pickle
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sign_language_model import generate_asl_dataset, prepare_features, predict_sign


def run_demo():
    print("=" * 60)
    print("  ASL Sign Language — Model Demo (No Camera)")
    print("=" * 60)

    # Load or train
    if not os.path.exists("models/asl_rf_model.pkl"):
        print("\n⚠️  Model not found. Training first...")
        from sign_language_model import train_model
        train_model()

    with open("models/asl_rf_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    print(f"\n✅ Model loaded — Training accuracy: {model_data['accuracy']*100:.1f}%\n")

    # Generate one fresh test sample per letter
    X_raw, y, labels = generate_asl_dataset(samples_per_class=5, noise=0.015)

    print(f"{'Letter':<8}{'Predicted':<12}{'Confidence':<14}{'Top 3':<35}{'Correct?'}")
    print("-" * 75)

    correct = 0
    total = 0
    # One sample per letter (first occurrence)
    seen = set()
    for i, label in enumerate(y):
        if label in seen:
            continue
        seen.add(label)

        lm_flat = X_raw[i]
        pred, conf, top3 = predict_sign(lm_flat, model_data)
        is_correct = pred == label
        correct += is_correct
        total += 1

        top3_str = " | ".join(f"{l}:{p:.0f}%" for l, p in top3)
        mark = "✅" if is_correct else "❌"
        print(f"  {label:<8}{pred:<12}{conf*100:.1f}%{'':>8}{top3_str:<35}  {mark}")

    print("-" * 75)
    print(f"\n📊 Demo Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    
    # Spell a word
    print("\n" + "=" * 60)
    print("  Spelling 'HELLO' using model predictions")
    print("=" * 60)
    
    word = "HELLO"
    result = ""
    for ch in word:
        idx = y.tolist().index(ch)
        lm_flat = X_raw[idx]
        pred, conf, _ = predict_sign(lm_flat, model_data)
        result += pred
        print(f"  Sign '{ch}' → Predicted: '{pred}' ({conf*100:.0f}% confidence)")
    
    print(f"\n  Output text: '{result}'")
    print(f"  Target:      '{word}'")
    print(f"  Match: {'✅ Yes!' if result == word else '❌ No'}")


if __name__ == "__main__":
    run_demo()