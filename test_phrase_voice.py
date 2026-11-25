"""
Test phrase voice output v2 — NO camera needed
=============================================
Speaks all 15 phrases, one by one, so you can test the voice
without opening the webcam.

Run:
    python test_phrase_voice.py              # speak all 15
    python test_phrase_voice.py --phrase thank_you
    python test_phrase_voice.py --sentence "Hello thank you please help"
    python test_phrase_voice.py --all "hello thank_you please"   # V key test

v2: every speak is instant + interruptible. Pressing SPACE twice fast
speaks the LATEST phrase, it never queues up old ones behind it.

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
    parser.add_argument('--sentence', help='Speak any sentence instantly')
    parser.add_argument('--all', help='V-key test: space-separated phrase keys, e.g. "hello thank_you please"')
    parser.add_argument('--rate', type=int, default=190, help='Words per minute (default 190)')
    args = parser.parse_args()

    voice = PhraseVoice(rate=args.rate)
    print(f"[*] {voice.status()}  (warmup done, ready={voice.available})")
    if voice.error:
        print(f"    Error: {voice.error}")

    if args.sentence:
        t0 = time.time()
        print(f"[*] Speaking sentence instantly: {args.sentence!r}")
        voice.speak_text(args.sentence)
        print(f"    handed to speaker in {time.time()-t0:.3f}s")
        time.sleep(max(2, len(args.sentence) // 8))
    elif args.all:
        keys = args.all.strip().lower().split()
        bad = [k for k in keys if k not in SPOKEN_TEXT]
        if bad:
            print(f"[!] Unknown phrases: {bad}. Choose from:")
            print("    " + ", ".join(sorted(SPOKEN_TEXT)))
            voice.shutdown()
            return 1
        texts = [SPOKEN_TEXT[k] for k in keys]
        print(f"[*] V key: speaking all {len(keys)} phrases as one sentence:")
        print(f"    {' '.join(texts)!r}")
        voice.speak_all(texts)
        time.sleep(max(2, len(' '.join(texts)) // 8))
    elif args.phrase:
        key = args.phrase.strip().lower()
        if key not in SPOKEN_TEXT:
            print(f"[!] Unknown phrase {key!r}. Choose from:")
            print("    " + ", ".join(sorted(SPOKEN_TEXT)))
            voice.shutdown()
            return 1
        t0 = time.time()
        print(f"[*] {DISPLAY_NAMES[key]} -> {SPOKEN_TEXT[key]!r}")
        voice.speak_phrase(key)
        print(f"    handed to speaker in {time.time()-t0:.3f}s (want <0.2s)")
        time.sleep(2.5)
    else:
        print(f"[*] Speaking all {len(SPOKEN_TEXT)} phrases...")
        for key in sorted(SPOKEN_TEXT):
            print(f"    {DISPLAY_NAMES[key]:<12} -> {SPOKEN_TEXT[key]!r}")
            voice.speak_phrase(key)
            time.sleep(2.5)

    print("[*] Testing interrupt: speak Hello, then immediately Thank You - latest must win")
    voice.speak_phrase('hello')
    voice.speak_phrase('thank_you')
    time.sleep(2)
    print("[*] Testing mute toggle + rate change...")
    print(f"    Muted now? {voice.toggle_mute()}  (should be True)")
    print(f"    Muted now? {voice.toggle_mute()}  (should be False)")
    print(f"    Rate {args.rate} -> {voice.change_rate(20)} wpm")
    voice.speak_text("Voice test complete.")
    time.sleep(2)

    voice.shutdown()
    print("[OK] Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
