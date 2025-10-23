#!/usr/bin/env python3
# modules/live_analyzer/tts.py

def speak(text: str, enable: bool = False):
    print(f"[VOICE] {text}")
    if not enable:
        return
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", 170)
        eng.say(text)
        eng.runAndWait()
    except Exception as e:
        print(f"[WARN] TTS failed: {e}")
