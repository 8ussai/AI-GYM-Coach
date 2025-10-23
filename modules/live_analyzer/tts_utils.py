"""
Simple background TTS queue based on pyttsx3.
- Non-blocking .speak(text)
- .enable() to toggle on/off
- .clear() to drop pending messages
- Works offline on Windows/Linux/macOS

Fixed version with:
- Better error handling and logging
- Proper resource cleanup
- Context manager support
- More robust threading
- Compatible with LiveFeedbackEngine interface
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pyttsx3  # pip install pyttsx3
    _HAVE_PYTTSX3 = True
except Exception:  # pragma: no cover
    pyttsx3 = None
    _HAVE_PYTTSX3 = False


@dataclass
class TTSConfig:
    """Configuration for TTS engine."""
    rate: int = 180  # Words per minute
    volume: float = 1.0  # 0.0 to 1.0
    voice_substring: Optional[str] = None  # e.g., "Arabic" or "David"


class TTSManager:
    """Background TTS manager with queued speech synthesis.
    
    Usage:
        tts = TTSManager()
        tts.speak("Hello world")
        tts.stop()  # Clean shutdown
    """

    def __init__(self, voice: Optional[str] = None, config: Optional[TTSConfig] = None) -> None:
        """Initialize TTS manager.
        
        Args:
            voice: Voice name substring to search for (e.g., "Arabic")
            config: Optional TTSConfig object. If provided, overrides 'voice' parameter
        """
        # Handle both interfaces: voice parameter (from UI) and config parameter
        if config is None:
            config = TTSConfig(voice_substring=voice)
        elif voice is not None:
            # If both provided, voice parameter takes precedence
            config.voice_substring = voice
            
        self.cfg = config
        self._q: queue.Queue[str] = queue.Queue(maxsize=10)  # Limit queue size
        self._enabled = True
        self._alive = True
        self._engine: Optional[object] = None
        self._lock = threading.Lock()
        
        # Initialize engine
        self._init_engine()
        
        # Start background thread
        self._thread = threading.Thread(target=self._loop, daemon=True, name="TTS-Worker")
        self._thread.start()

    def _init_engine(self) -> None:
        """Initialize pyttsx3 engine with error handling."""
        if not _HAVE_PYTTSX3:
            print("[TTS] pyttsx3 not available - install with: pip install pyttsx3")
            return
        
        try:
            self._engine = pyttsx3.init()
            
            # Set properties
            if self._engine:
                try:
                    self._engine.setProperty("rate", self.cfg.rate)
                except Exception as e:
                    print(f"[TTS] Failed to set rate: {e}")
                
                try:
                    self._engine.setProperty("volume", float(self.cfg.volume))
                except Exception as e:
                    print(f"[TTS] Failed to set volume: {e}")
                
                # Set voice if specified
                if self.cfg.voice_substring:
                    self._set_voice(self.cfg.voice_substring)
                    
        except Exception as e:
            print(f"[TTS] Engine initialization failed: {e}")
            self._engine = None

    def _set_voice(self, voice_substring: str) -> None:
        """Set voice by searching for substring in available voices."""
        if not self._engine:
            return
        
        try:
            voices = self._engine.getProperty("voices")
            if not voices:
                print("[TTS] No voices available")
                return
            
            # Search for matching voice
            voice_substring_lower = voice_substring.lower()
            for v in voices:
                # Build searchable string from voice properties
                name = getattr(v, 'name', '')
                langs = getattr(v, 'languages', [])
                lang_str = ' '.join(str(l) for l in langs) if langs else ''
                search_str = f"{name} {lang_str}".lower()
                
                if voice_substring_lower in search_str:
                    self._engine.setProperty("voice", v.id)
                    print(f"[TTS] Using voice: {name}")
                    return
            
            print(f"[TTS] Voice containing '{voice_substring}' not found")
            print(f"[TTS] Available voices: {[getattr(v, 'name', 'unknown') for v in voices]}")
            
        except Exception as e:
            print(f"[TTS] Failed to set voice: {e}")

    def speak(self, text: str, clear_pending: bool = False) -> None:
        """Queue text for speech synthesis (non-blocking).
        
        Args:
            text: Text to speak
            clear_pending: If True, clear queue before adding new text
        """
        if not text or not isinstance(text, str):
            return
        
        if not self._enabled:
            return
        
        if clear_pending:
            self.clear()
        
        # Try to add to queue, drop if full
        try:
            self._q.put_nowait(text.strip())
        except queue.Full:
            print(f"[TTS] Queue full, dropping message: {text[:50]}...")

    def enable(self, enabled: bool) -> None:
        """Enable or disable TTS output.
        
        Args:
            enabled: True to enable, False to disable
        """
        self._enabled = bool(enabled)
        if not enabled:
            self.clear()

    def clear(self) -> None:
        """Clear all pending messages and stop current speech."""
        # Clear queue
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        
        # Stop current speech
        if self._engine:
            with self._lock:
                try:
                    self._engine.stop()
                except Exception as e:
                    print(f"[TTS] Error stopping engine: {e}")

    def stop(self) -> None:
        """Stop TTS thread and clean up resources."""
        self._alive = False
        self.clear()
        
        # Wait for thread to finish (with timeout)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        # Clean up engine
        if self._engine:
            with self._lock:
                try:
                    self._engine.stop()
                except Exception:
                    pass
                
                # Some engines need explicit cleanup
                try:
                    if hasattr(self._engine, 'endLoop'):
                        self._engine.endLoop()
                except Exception:
                    pass

    def is_alive(self) -> bool:
        """Check if TTS thread is running."""
        return self._alive and self._thread.is_alive()

    # ---------------- Internal loop ----------------
    def _loop(self) -> None:
        """Background worker thread that processes speech queue."""
        while self._alive:
            try:
                # Wait for text with timeout to allow checking _alive flag
                text = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            
            # Process text
            try:
                if self._engine and self._enabled:
                    with self._lock:
                        self._engine.say(text)
                        self._engine.runAndWait()
                elif not _HAVE_PYTTSX3:
                    # Simulate speech time if pyttsx3 not available
                    duration = min(2.0 + 0.03 * len(text), 6.0)
                    print(f"[TTS Simulation] {text} (would take ~{duration:.1f}s)")
                    time.sleep(duration)
            except Exception as e:
                # Swallow TTS errors to keep system alive
                print(f"[TTS] Speech error: {e}")
                time.sleep(0.1)
            
            # Mark task as done
            try:
                self._q.task_done()
            except Exception:
                pass

    # ---------------- Context manager support ----------------
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop()
        except Exception:
            pass


# Backward compatibility alias
class TTS(TTSManager):
    """Backward compatibility alias for TTS class."""
    
    def say(self, text: str, clear_pending: bool = False) -> None:
        """Alias for speak() method."""
        self.speak(text, clear_pending)


# Simple test/demo
if __name__ == "__main__":
    import sys
    
    print("Testing TTS Manager...")
    print(f"pyttsx3 available: {_HAVE_PYTTSX3}")
    
    # Test with context manager
    with TTSManager() as tts:
        tts.speak("Testing TTS system")
        time.sleep(2)
        
        tts.speak("This is a second message")
        time.sleep(2)
        
        print("Clearing queue...")
        tts.clear()
        
        tts.speak("This message should play after clear")
        time.sleep(2)
    
    print("Test complete!")