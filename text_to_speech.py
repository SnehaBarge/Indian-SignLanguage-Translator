
import pyttsx3
import threading
import queue


class TextToSpeech:

    def __init__(self):
        self.speech_queue = queue.Queue()

        self.thread = threading.Thread(
            target=self.speech_worker,
            daemon=True
        )
        self.thread.start()

    def speech_worker(self):
        engine = pyttsx3.init()

        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)

        while True:
            text = self.speech_queue.get()

            if text is None:
                break

            engine.say(text)
            engine.runAndWait()

        engine.stop()

    def speak(self, text):
        if text.strip():
            self.speech_queue.put(text)

    def close(self):
        self.speech_queue.put(None)
        self.thread.join(timeout=3)
