import argparse
import json
from pathlib import Path
from time import perf_counter
from PIL import Image, ImageOps
from plateclock.pipeline import run, load_config, public_result
from plateclock.language.style_guide import load_style

def main():
    parser = argparse.ArgumentParser(description='PlateClock Week 2: plate photo to text and WAV')
    parser.add_argument('--image')
    parser.add_argument('--checkpoint')
    parser.add_argument('--config', default='configs/week2.yaml')
    parser.add_argument('--style', default='configs/style_guide.yaml')
    parser.add_argument('--out', default='results/demo/plate')
    parser.add_argument('--mode', choices=['full','summary'], default='full')
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--list-voices', action='store_true')
    parser.add_argument('--voice')
    args = parser.parse_args()
    from plateclock.audio.tts import save, voices
    if args.list_voices:
        voices(); return
    if not args.image:
        parser.error('--image is required')
    cfg = load_config(args.config)
    voice = args.voice or cfg['audio']['voice']
    if not args.no_audio and not voice:
        parser.error('Select --voice from --list-voices or set audio.voice in config')
    import numpy as np
    from plateclock.plate.segment_food import FoodSegmenter
    start = perf_counter()
    model = FoodSegmenter(args.checkpoint or cfg['model']['checkpoint'])
    with Image.open(args.image) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    inference_start = perf_counter()
    result = run(image, model, cfg, load_style(args.style), args.mode)
    payload = public_result(result)
    payload['inference_seconds'] = perf_counter()-inference_start
    prefix = Path(args.out); prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix)+'.txt').write_text(result['text']+'\n', encoding='utf-8')
    Image.fromarray(result['overlay']).save(str(prefix)+'_overlay.png')
    if not args.no_audio:
        save(result['text'], str(prefix)+'.wav', voice, cfg['audio']['rate'])
    payload['total_seconds'] = perf_counter()-start
    payload['audio'] = None if args.no_audio else {'voice':voice, 'rate':cfg['audio']['rate']}
    Path(str(prefix)+'.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(result['text'])
    print('Status:', result['status'], '| outputs:', prefix)

if __name__ == '__main__':
    main()
