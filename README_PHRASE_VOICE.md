# Phrase Voice Output v2 — instant, for the Sign Language project

Voice output for the **15 phrases only** (not the A–Z alphabet demo).

## What changed in v2 — why v1 felt slow / broken

| v1 problem | v2 fix |
|---|---|
| Every phrase queued behind the one still speaking. Press SPACE 3x fast → 3rd phrase spoke ~5s late. | `speak_now()`: SPACE **stops** whatever is speaking and starts the new phrase immediately. Nothing queues behind an old phrase. |
| `runAndWait()` blocked the worker, `stop()` from the main thread could not interrupt it. | Non-blocking pyttsx3: `startLoop(False)` + `iterate()` in the worker loop. Stop works mid-sentence, in the same thread. |
| Same phrase twice while speaking was refused (dedup bug). | Removed. "Yes Yes" now speaks twice. |
| Engine started lazily, first SPACE paid 1–2s startup. | Engine is **warmed up in `__init__`** (waits up to 4s for ready). First SPACE is instant. |
| Auto-speak was OFF, needed `A` + 1.2s steady. | Auto-speak is **ON by default**, steady for only **0.6s**. `A` turns it off, `--no-auto-speak` too. |
| `V` spoke display names ("Eat / Food") from a string log. | `V` speaks the **spoken sentences** in order: "Hello there! Thank you. Please." Backspace / Enter now keep the spoken list in sync. |

## Files — copy all 3 into your project folder

| File | What it is |
|---|---|
| `phrase_voice.py` | v2 voice engine. Threaded, non-blocking, interruptible. |
| `realtime_phrases_voice_demo.py` | Live demo + voice. Detection unchanged. |
| `test_phrase_voice.py` | Test voice with **no camera**. |

Add to `requirements.txt`: `pyttsx3`

## Install

```bash
pip install pyttsx3
# Linux also: sudo apt install espeak-ng
# Windows / macOS work out of the box
```

If pyttsx3 / espeak is missing, the demo **still runs** — it prints
`Voice: UNAVAILABLE` and what it would have said, instead of crashing.

## Run — exactly as you asked

```bash
# Test first, no camera
python test_phrase_voice.py --phrase thank_you
python test_phrase_voice.py --all "hello thank_you please"   # V-key test

# Live: SPACE selects + speaks instantly, V speaks everything selected
python realtime_phrases_voice_demo.py
python realtime_phrases_voice_demo.py --no-auto-speak   # only SPACE / V speak
python realtime_phrases_voice_demo.py --rate 200
```

## Controls

| Key | Action |
|---|---|
| `SPACE` | **Select phrase + SPEAK it instantly** (interrupts anything speaking) |
| `V` | **Speak ALL selected phrases**, in order, as one sentence |
| `A` | Toggle auto-speak on / off (default ON, 0.6s steady) |
| `M` | Mute / unmute |
| `+` / `-` | Faster / slower voice (100–250 wpm, default 190) |
| `BACKSPACE` | Delete last selected phrase |
| `ENTER` | Clear all + stop speaking |
| `Q` / `ESC` | Quit |

Example flow:
1. Show `Hello` sign → press `SPACE` → hears "Hello there!" instantly, "Hello" is logged
2. Show `Thank You` → press `SPACE` → hears "Thank you." instantly
3. Press `V` → hears "Hello there! Thank you." as one sentence

## What it says — all 15

Hello "Hello there!", Thank You, Please, Yes, No, I Love You, Good, Bad,
Help "Please help me.", More "More, please.", Eat "I want to eat.",
Stop, Sorry "I'm sorry.", Wait "Please wait.", Name "My name is."

Change any line in `phrase_voice.py` → `SPOKEN_TEXT` to change what is spoken.

## Tested

- `python -m py_compile` — all 3 files pass.
- Fake-engine test (no audio hardware needed): warmup 0.00s, `say()` reached
  in **0.020s** after SPACE, rapid 3x SPACE → latest phrase wins with a
  `stop()` call, repeat "Yes Yes" speaks twice, V speaks the joined
  sentence "Hello there! Thank you. Please." — all passed.
- Real fallback (no espeak on this machine): prints `Voice: UNAVAILABLE`
  and exits cleanly. Live webcam was not run here (no camera) — run the
  live demo on your laptop to hear it.
