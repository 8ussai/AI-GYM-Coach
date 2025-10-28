#!/usr/bin/env python3
# modules/live_analyzer/tts.py
"""
Simple TTS wrapper using pyttsx3 for offline speech synthesis.
"""

import threading
import queue
from typing import Optional

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[WARN] pyttsx3 not installed. Install via: pip install pyttsx3")
    TTS_AVAILABLE = False


class TTSEngine:
    """Thread-safe TTS engine."""
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.enabled = TTS_AVAILABLE
        self.queue: queue.Queue = queue.Queue()
        self.engine: Optional[pyttsx3.Engine] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        
        if not self.enabled:
            return
        
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Try to set Arabic voice if available
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'arabic' in voice.name.lower() or 'ar' in voice.languages:
                    self.engine.setProperty('voice', voice.id)
                    break
            
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            
            print("[TTS] ✅ Initialized successfully")
        except Exception as e:
            print(f"[TTS] ❌ Failed to initialize: {e}")
            self.enabled = False
    
    def _worker(self):
        """Background thread that processes TTS queue."""
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if text and self.engine:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS] Error: {e}")
    
    def speak(self, text: str):
        """Queue text for speech synthesis."""
        if not self.enabled or not text:
            return
        
        # Clear queue to avoid backlog (only speak latest)
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
        
        self.queue.put(text)
    
    def stop(self):
        """Stop the TTS engine."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass


# Global instance
_tts_instance: Optional[TTSEngine] = None


def get_tts() -> TTSEngine:
    """Get or create global TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance


def speak(text: str):
    """Convenience function to speak text."""
    get_tts().speak(text)


def stop_tts():
    """Stop TTS engine."""
    global _tts_instance
    if _tts_instance:
        _tts_instance.stop()
        _tts_instance = None