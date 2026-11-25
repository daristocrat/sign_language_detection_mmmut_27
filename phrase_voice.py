"""
Phrase Voice Output — for the sign-language phrase project
=============================================================
Speaks the 15 ASL phrases from gesture_phrases_model.py out loud.

Why this is a separate file:
  * pyttsx3 blocks while it speaks. If you call it directly from the
    webcam loop, the video freezes on every word.
  * This class runs the speaker in its own background thread with a
    queue, so the camera keeps running while it talks.
  * If pyttsx3 / espeak is not installed, it fails gracefully:
    voice is disabled, the demo still runs, and it prints what it
    *would* have said.

Usage:
    from phrase_voice import PhraseVoice

    voice = PhraseVoice(rate=175)          # 175 words/min, default
    voice.speak_phrase("thank_you")        # -> speaks "Thank you."
    voice.speak_text("Hello, thank you.")  # speak any sentence
    voice.toggle_mute()                    # M key
    voice.change_rate(+20)                 # + key
    voice.stop()                           # stop current speech
    voice.shutdown()                       # call on quit

Install:
    pip install pyttsx3
    # Linux also needs a speech backend:
    sudo apt install espeak-ng     # Ubuntu / Debian
    # Windows / macOS: pyttsx3 works out of the box (SAPI5 / NSSpeech)
"""

import queue
import threading

# Phrase key (as in gesture_phrases_model.py) -> natural spoken sentence.
# Keep these short and natural. This is what the user hears.
SPOKEN_TEXT = {
    'hello':      'Hello there!',
    'thank_you':  'Thank you.',
    'please':     'Please.',
    'yes':        'Yes.',
    'no':         'No.',
    'i_love_you': 'I love you.',
    'good':       'Good.',
    'bad':        'Bad.',
    'help':       'Please help me.',
    'more':       'More, please.',
    'eat':        'I want to eat.',
    'stop':       'Stop.',
    'sorry':      "I'm sorry.",
    'wait':       'Please wait.',
    'name':       'My name is.',
}

# Display name used on screen, kept in sync with realtime_phrases_demo.py
DISPLAY_NAMES = {
    'hello': 'Hello', 'thank_you': 'Thank You', 'please': 'Please',
    'yes': 'Yes', 'no': 'No', 'i_love_you': 'I Love You',
    'good': 'Good', 'bad': 'Bad', 'help': 'Help', 'more': 'More',
    'eat': 'Eat / Food', 'stop': 'Stop', 'sorry': 'Sorry',
    'wait': 'Wait', 'name': 'Name',
}


class PhraseVoice:
    """Threaded, non-blocking text-to-speech for phrases."""

    def __init__(self, rate=175, volume=1.0, muted=False, enabled=True):
        self.rate = rate
        self.volume = volume
        self.muted = muted
        self.enabled = enabled
        self.available = False
        self.error = None
        self._last_queued = None

        self._queue = queue.Queue(maxsize=5)
        self._thread = None
        self._engine = None

        if not enabled:
            return
        # Start the worker thread. pyttsx3 is created *inside* that
        # thread, because it must be used from the thread that made it.
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # -- public API -------------------------------------------------

    def speak_phrase(self, phrase_key):
        """Speak one phrase by its key, e.g. 'thank_you'."""
        text = SPOKEN_TEXT.get(phrase_key)
        if text is None:
            return False
        return self.speak_text(text)

    def speak_text(self, text):
        """Queue a sentence to be spoken. Returns False if muted/off."""
        if not text or self.muted or not self.enabled:
            return False
        if text == self._last_queued and not self._queue.empty():
            return False  # don't queue the same line twice in a row
        if self._thread is not None and not self._thread.is_alive():
            # Worker died during init (e.g. no espeak) - don't pretend it spoke
            print(f"   [VOICE off] would say: {text}")
            return False
        if not self.available and self._thread is None:
            # No speaker thread at all (disabled) - print as fallback
            print(f"   [VOICE off] would say: {text}")
            return False
        try:
            self._queue.put_nowait(('speak', text))
            self._last_queued = text
            return True
        except queue.Full:
            return False  # busy speaking - drop it, don't lag the video

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.stop()
        return self.muted

    def change_rate(self, delta):
        """Change speed by delta wpm, clamped to 100-250."""
        self.rate = max(100, min(250, self.rate + delta))
        if self._thread is not None:
            try:
                self._queue.put_nowait(('rate', self.rate))
            except queue.Full:
                pass
        return self.rate

    def set_rate(self, rate):
        return self.change_rate(rate - self.rate)

    def stop(self):
        """Stop current speech and clear the queue."""
        self._last_queued = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def shutdown(self):
        if self._thread is not None:
            try:
                self._queue.put_nowait(('shutdown', None))
            except queue.Full:
                pass
            self._thread.join(timeout=2.0)
            self._thread = None

    def status(self):
        if not self.enabled:
            return "Voice: OFF"
        if self.muted:
            return f"Voice: MUTED  ({self.rate} wpm)"
        if not self.available:
            return "Voice: UNAVAILABLE - install pyttsx3 + espeak-ng"
        return f"Voice: ON  ({self.rate} wpm)"

    # -- worker thread ----------------------------------------------

    def _worker(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            self.available = True
        except Exception as e:  # noqa: BLE001 - report, don't crash the demo
            self.available = False
            self.error = str(e)
            print(f"[!] Voice unavailable: {e}")
            print("    Install: pip install pyttsx3  +  sudo apt install espeak-ng")
            return

        while True:
            try:
                cmd, value = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd == 'shutdown':
                break
            if cmd == 'rate':
                try:
                    self._engine.setProperty('rate', value)
                except Exception:
                    pass
            elif cmd == 'speak':
                if self.muted:
                    continue
                try:
                    self._engine.say(value)
                    self._engine.runAndWait()
                except Exception as e:  # noqa: BLE001
                    print(f"[!] Speech failed: {e}")
                finally:
                    if self._last_queued == value:
                        self._last_queued = None
        try:
            self._engine.stop()
        except Exception:
            pass
