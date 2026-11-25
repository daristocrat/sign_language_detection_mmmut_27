# Phrase Voice Output — for the Sign Language project

Voice output for the **15 phrases only** (not the A–Z alphabet demo).

Your original `realtime_phrases_demo.py` had **no voice code at all** — the README
promised TTS, but nothing spoke. These files add it.

## Files — copy all 3 into your project folder

| File | What it is |
|---|---|
| `phrase_voice.py` | The voice engine. Threaded pyttsx3, non-blocking. Import it, don't edit it. |
| `realtime_phrases_voice_demo.py` | Your original `realtime_phrases_demo.py` + voice only. Detection / smoothing / UI unchanged. |
| `test_phrase_voice.py` | Test the voice with **no camera**. Speaks all 15 phrases. |

Also add to `requirements.txt`:
```
pyttsx3
```

## Install

```bash
pip install pyttsx3
```

Linux also needs a speech backend (Windows / macOS work out of the box):
```bash
sudo apt install espeak-ng    # Ubuntu / Debian
```

If pyttsx3 / espeak is missing, the demo **still runs** — it prints
`Voice: UNAVAILABLE` and shows what it would have said, instead of crashing.
That is intentional, so a viva demo never dies on a missing driver.

## Run

```bash
# 1. Test voice first, no camera
python test_phrase_voice.py
python test_phrase_voice.py --phrase thank_you

# 2. Live demo — SPACE logs a phrase AND speaks it
python realtime_phrases_voice_demo.py

# 3. Hands-free — speak every stable phrase automatically
python realtime_phrases_voice_demo.py --auto-speak

# 4. Silent, or faster voice
python realtime_phrases_voice_demo.py --no-voice
python realtime_phrases_voice_demo.py --rate 200
```

## Controls

| Key | Action |
|---|---|
| `SPACE` | Log current phrase **+ speak it** |
| `V` | Speak the whole logged sentence |
| `A` | Toggle auto-speak on / off |
| `M` | Mute / unmute |
| `+` / `-` | Faster / slower voice (100–250 wpm) |
| `BACKSPACE` | Delete last word |
| `ENTER` | Clear log + stop speaking |
| `Q` / `ESC` | Quit |

## What it says — all 15 phrases

| Sign | Spoken |
|---|---|
| Hello | "Hello there!" |
| Thank You | "Thank you." |
| Please | "Please." |
| Yes | "Yes." |
| No | "No." |
| I Love You | "I love you." |
| Good | "Good." |
| Bad | "Bad." |
| Help | "Please help me." |
| More | "More, please." |
| Eat / Food | "I want to eat." |
| Stop | "Stop." |
| Sorry | "I'm sorry." |
| Wait | "Please wait." |
| Name | "My name is." |

Change any line in `phrase_voice.py` → `SPOKEN_TEXT` to change what is spoken.
`name` says "My name is." on purpose — the sign alone has no name in it, so
the full sentence needs your name added after it. Change it to e.g.
`"My name is Prince."` if you want it complete.

## How it works

1. Webcam → MediaPipe (21 landmarks) → Random Forest → stable phrase
   (same as before, 8-frame smoothing, 0.45 confidence).
2. On `SPACE` (or auto-speak after 1.2 s stable), the phrase key is sent
   to `PhraseVoice.speak_phrase()`.
3. `PhraseVoice` puts the sentence on a queue. A background thread runs
   pyttsx3, so the video never freezes while it talks.
4. Same phrase is never queued twice in a row, and the queue holds max
   5 lines — if you're signing fast, old lines are dropped instead of
   lagging behind.

Auto-speak rule: a phrase must be stable for **1.2 seconds** and different
from the last spoken phrase before it speaks. That stops it repeating
"Hello hello hello" while you hold your hand up.

## Tested

- `python -m py_compile` — all 3 files pass.
- `test_phrase_voice.py --phrase thank_you` — passes, prints
  `Voice: UNAVAILABLE` on a machine with no espeak, and exits cleanly
  instead of crashing. On a machine with espeak / Windows / macOS it
  speaks out loud.
- Mute toggle, rate change (175 → 195 wpm), bad phrase key, and
  shutdown were all tested in the same run.
- Live webcam demo was **not** run here (no camera in this environment) —
  run step 2 above on your laptop to hear it live.
