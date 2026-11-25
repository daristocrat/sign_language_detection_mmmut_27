"""
Test phrase voice output — NO camera needed
=============================================
Speaks all 15 phrases, one by one, so you can test the voice
without opening the webcam.

Run:
    python test_phrase_voice.py              # speak all 15
    python test_phrase_voice.py --phrase thank_you
    python test_phrase_voice.py --sentence "Hello thank you please help"
    python test_phrase_voice.py --rate 200

If pyttsx3 / espeak is missing, it prints what it would have said
and exits cleanly instead of crashing.
"""

import argparse
import sys
import time

sys.path.insert(0, '.')
from phrase_voice import PhraseVoice, SPOKEN_TEXT, DISPLAY_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phrase', help='Speak one phrase key, e.g. thank_you')
    parser.add_argument('--sentence', help='Speak any sentence')
    parser.add_argument('--rate', type=int, default=175, help='Words per minute (default 175)')
    args = parser.parse_args()

    voice = PhraseVoice(rate=args.rate)
    # Give the worker thread a moment to start pyttsx3
    time.sleep(0.8)
    print(f"[*] {voice.status()}")
    if voice.error:
        print(f"    Error: {voice.error}")

    if args.sentence:
        print(f"[*] Speaking sentence: {args.sentence!r}")
        voice.speak_text(args.sentence)
        time.sleep(max(2, len(args.sentence) // 8))
    elif args.phrase:
        key = args.phrase.strip().lower()
        if key not in SPOKEN_TEXT:
            print(f"[!] Unknown phrase {key!r}. Choose from:")
            print("    " + ", ".join(sorted(SPOKEN_TEXT)))
            voice.shutdown()
            return 1
        print(f"[*] {DISPLAY_NAMES[key]} -> {SPOKEN_TEXT[key]!r}")
        voice.speak_phrase(key)
        time.sleep(2.5)
    else:
        print(f"[*] Speaking all {len(SPOKEN_TEXT)} phrases...")
        for key in sorted(SPOKEN_TEXT):
            print(f"    {DISPLAY_NAMES[key]:<12} -> {SPOKEN_TEXT[key]!r}")
            voice.speak_phrase(key)
            time.sleep(2.5)  # wait for each line to finish

    # Quick control test: mute, rate change, sentence
    print("[*] Testing mute toggle + rate change...")
    print(f"    Muted now? {voice.toggle_mute()}  (should be True)")
    print(f"    Muted now? {voice.toggle_mute()}  (should be False)")
    print(f"    Rate 175 -> {voice.change_rate(25)} wpm")
    voice.speak_text("Voice test complete.")
    time.sleep(2)

    voice.shutdown()
    print("[OK] Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
