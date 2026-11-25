"""
Phrase Voice Output v2 — instant, interruptible
=================================================
Speaks the 15 ASL phrases from gesture_phrases_model.py out loud.

v1 problems this fixes:
  1. DELAY: v1 queued every phrase behind the one still speaking.
     Press SPACE 3 times fast and the 3rd phrase spoke ~5 seconds late.
  2. NO INTERRUPT: v1 used runAndWait() in the worker thread.
     engine.stop() from the main thread could not stop it.
  3. REPEAT BLOCKED: v1 refused to queue the same phrase twice while
     it was still speaking, so pressing SPACE twice for "Yes Yes"
     did nothing the second time.
  4. COLD START: v1 returned before pyttsx3 had finished starting,
     so the first SPACE was always late.

v2 design:
  * The worker thread owns pyttsx3 and runs it in NON-BLOCKING mode:
    engine.startLoop(False) + engine.iterate() in a tight loop.
    That is the only pyttsx3 pattern that lets you stop mid-sentence.
  * speak_now() is the default: it STOPS whatever is speaking right
    now, clears the queue, and starts the new phrase immediately.
    Nothing ever waits behind an old phrase.
  * The engine is WARMED UP in __init__ (wait up to 4s for ready).
    The first SPACE is instant because the engine is already running.
  * If pyttsx3 / espeak is missing, it still fails gracefully:
    voice is disabled, the demo runs, and it prints what it would
    have said.

Usage:
    from phrase_voice import PhraseVoice

    voice = PhraseVoice(rate=190)          # warmed up, ready to speak
    voice.speak_phrase("thank_you")        # interrupts + speaks NOW
    voice.speak_text("Hello, thank you.")  # interrupts + speaks NOW
    voice.speak_all(["Hello there!", "Thank you."])  # V key: full sentence
    voice.toggle_mute()                    # M key
    voice.change_rate(+20)                 # + key
    voice.stop()                           # stop speaking immediately
    voice.shutdown()                       # call on quit
"""

import queue
import threading
import time

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

DISPLAY_NAMES = {
    'hello': 'Hello', 'thank_you': 'Thank You', 'please': 'Please',
    'yes': 'Yes', 'no': 'No', 'i_love_you': 'I Love You',
    'good': 'Good', 'bad': 'Bad', 'help': 'Help', 'more': 'More',
    'eat': 'Eat / Food', 'stop': 'Stop', 'sorry': 'Sorry',
    'wait': 'Wait', 'name': 'Name',
}


class PhraseVoice:
    """Instant, interruptible text-to-speech for phrases."""

    def __init__(self, rate=190, volume=1.0, muted=False, enabled=True,
                 warmup_timeout=4.0):
        self.rate = rate
        self.volume = volume
        self.muted = muted
        self.enabled = enabled
        self.available = False
        self.error = None
        self.speaking = False
        self.current_text = ""

        self._queue = queue.Queue()  # unbounded: speak_now clears it anyway
        self._thread = None
        self._engine = None
        self._ready = threading.Event()
        self._stop_requested = threading.Event()

        if not enabled:
            self._ready.set()
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        # Warm-up: wait until pyttsx3 is actually ready, so the
        # first SPACE does not pay the 1-2s startup cost.
        self._ready.wait(timeout=warmup_timeout)

    # -- public API: everything here is "speak NOW" -----------------

    def speak_phrase(self, phrase_key):
        """SPACE: speak one phrase immediately, interrupting anything else."""
        text = SPOKEN_TEXT.get(phrase_key)
        if text is None:
            return False
        return self.speak_now(text)

    def speak_text(self, text):
        """Speak any text immediately, interrupting anything else."""
        return self.speak_now(text)

    def speak_all(self, texts):
        """V key: speak a whole list of selected phrases as one sentence.

        texts: list of spoken sentences, e.g.
            ["Hello there!", "Thank you.", "Please."]
        Joins them with a space and speaks it in one go.
        Returns False if there is nothing to speak.
        """
        if not texts:
            return False
        sentence = " ".join(t.strip() for t in texts if t and t.strip())
        return self.speak_now(sentence) if sentence else False

    def speak_now(self, text):
        """Core method: stop current speech + clear queue + speak text NOW.

        Returns True if the text was handed to the speaker thread
        (or printed as fallback when voice is unavailable).
        """
        if not text or not text.strip():
            return False
        text = text.strip()
        if self.muted or not self.enabled:
            return False
        if self._thread is not None and not self._thread.is_alive():
            print(f"   [VOICE off] would say: {text}")
            return False
        if self._thread is None:
            print(f"   [VOICE off] would say: {text}")
            return False
        # Clear anything waiting, then queue only this one.
        # The worker sees 'speak_now' and stops the current utterance
        # in its own thread before starting this one.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(('speak_now', text))
        return True

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.stop()
        return self.muted

    def change_rate(self, delta):
        self.rate = max(100, min(250, self.rate + delta))
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(('rate', self.rate))
        return self.rate

    def set_rate(self, rate):
        return self.change_rate(rate - self.rate)

    def stop(self):
        """Stop speaking immediately."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(('stop', None))
        self.speaking = False
        self.current_text = ""

    def shutdown(self):
        if self._thread is not None:
            self._queue.put(('shutdown', None))
            self._thread.join(timeout=2.0)
            self._thread = None
        self.speaking = False

    def status(self):
        if not self.enabled:
            return "Voice: OFF"
        if self.muted:
            return f"Voice: MUTED  ({self.rate} wpm)"
        if not self.available:
            return "Voice: UNAVAILABLE - install pyttsx3 + espeak-ng"
        if self.speaking and self.current_text:
            return f"Speaking: {self.current_text}  ({self.rate} wpm)"
        return f"Voice: ON  ({self.rate} wpm)"

    # -- worker thread: owns pyttsx3, non-blocking loop --------------

    def _worker(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
            # Non-blocking loop: startLoop(False) + iterate() lets
            # this thread stop mid-sentence. runAndWait() cannot.
            self._engine.startLoop(False)
            self.available = True
        except Exception as e:  # noqa: BLE001
            self.available = False
            self.error = str(e)
            print(f"[!] Voice unavailable: {e}")
            print("    Install: pip install pyttsx3  +  sudo apt install espeak-ng")
            self._ready.set()
            return

        self._ready.set()
        engine = self._engine
        busy = False  # True while an utterance is playing

        while True:
            # Drain commands first, so a new SPACE always wins.
            try:
                cmd, value = self._queue.get_nowait()
            except queue.Empty:
                cmd, value = None, None

            if cmd == 'shutdown':
                break
            if cmd == 'stop':
                try:
                    engine.stop()
                except Exception:
                    pass
                busy = False
                self.speaking = False
                self.current_text = ""
                continue
            if cmd == 'rate':
                try:
                    engine.setProperty('rate', value)
                except Exception:
                    pass
                continue
            if cmd == 'speak_now':
                # Interrupt: stop in THIS thread, then start the new one.
                try:
                    engine.stop()
                except Exception:
                    pass
                busy = False
                if not self.muted:
                    try:
                        engine.say(value)
                        self.current_text = value
                        self.speaking = True
                        busy = True
                    except Exception as e:  # noqa: BLE001
                        print(f"[!] Speech failed: {e}")
                        self.speaking = False
                continue

            # No command: pump the engine once, then rest 20ms.
            # iterate() returns False when the utterance finished.
            if busy:
                try:
                    still_busy = engine.iterate()
                except Exception:
                    still_busy = False
                if not still_busy:
                    busy = False
                    self.speaking = False
                    self.current_text = ""
            time.sleep(0.02)

        try:
            engine.endLoop()
        except Exception:
            pass
        try:
            engine.stop()
        except Exception:
            pass
        self.speaking = False
