"""Offline WAV export; fixed explicit voice and rate, no guessed voice index."""
from pathlib import Path
import platform
import subprocess
import tempfile
import wave

def voices():
    if platform.system() == 'Darwin':
        subprocess.run(['say', '-v', '?'], check=True)
    else:
        import pyttsx3
        engine = pyttsx3.init()
        try:
            for voice in engine.getProperty('voices'):
                print(voice.id, '::', voice.name)
        finally:
            engine.stop()

def save(text, output, voice, rate=165):
    output = Path(output)
    if output.suffix.lower() != '.wav':
        raise ValueError('Use a .wav filename')
    if not voice or not isinstance(rate, int) or not 80 <= rate <= 300:
        raise ValueError('Set an installed voice ID and an integer rate between 80 and 300')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids treating a stale audio file as new successful output.
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temp = Path(directory) / 'speech.wav'
        if platform.system() == 'Darwin':
            subprocess.run(['say', '-v', voice, '-r', str(rate), '--file-format=WAVE',
                            '--data-format=LEI16@22050', '-o', str(temp), text], check=True)
        else:
            import pyttsx3
            engine = pyttsx3.init()
            try:
                if voice not in {v.id for v in engine.getProperty('voices')}:
                    raise ValueError('Voice ID is not installed; run --list-voices')
                engine.setProperty('voice', voice)
                engine.setProperty('rate', rate)
                engine.save_to_file(text, str(temp))
                engine.runAndWait()
            finally:
                engine.stop()
        if not temp.exists():
            raise RuntimeError('TTS produced no file; check the OS speech engine')
        with wave.open(str(temp), 'rb') as wav:
            if wav.getnframes() <= 0:
                raise RuntimeError('TTS produced empty audio')
        temp.replace(output)
    return output
