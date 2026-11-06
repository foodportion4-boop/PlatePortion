"""Speak a lint-checked result downloaded from Kaggle; no model required."""
import argparse
from pathlib import Path
from plateclock.audio.tts import save
from plateclock.language.style_guide import load_style
from plateclock.language.style_lint import require_valid

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--text', required=True)
    p.add_argument('--voice', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=['full', 'summary'], default='full')
    p.add_argument('--rate', type=int, default=165)
    a = p.parse_args()
    text = Path(a.text).read_text(encoding='utf-8').strip()
    require_valid(text, a.mode, load_style())
    print(save(text, a.out, a.voice, a.rate))
