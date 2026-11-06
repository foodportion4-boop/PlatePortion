"""Local Flask website; keeps image bytes in memory and never stores uploads."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
import base64
import threading
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from plateclock.integrated import IntegratedService

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web'
app=Flask(__name__,static_folder=None)
app.config['MAX_CONTENT_LENGTH']=18*1024*1024
service=IntegratedService(ROOT/'checkpoints/foodseg-b0',ROOT/'checkpoints/nutrition-direct/best.pt',ROOT/'checkpoints/nutrition-calibration.json')
lock=threading.Lock()

@app.get('/')
def home():return send_from_directory(WEB,'index.html')

@app.get('/assets/<path:filename>')
def assets(filename):return send_from_directory(WEB,filename)

@app.get('/api/health')
def health():return jsonify({'models':service.readiness(), 'local_only':True})

@app.post('/api/analyze')
def analyze():
    file=request.files.get('photo')
    if not file or not file.filename:
        return jsonify({'error':'Choose a plate photo first.'}),400
    if not file.mimetype.startswith('image/'):
        return jsonify({'error':'Photo must be an image.'}),400
    try:
        with Image.open(BytesIO(file.read())) as im:
            rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        if rgb.shape[0]*rgb.shape[1]>16_000_000:
            raise ValueError('Photo is too large; resize it below 16 megapixels.')
        raw=None
        depth_file=request.files.get('depth')
        if depth_file and depth_file.filename:
            if not depth_file.filename.lower().endswith('.png'):
                raise ValueError('Depth must be a 16-bit raw PNG.')
            with Image.open(BytesIO(depth_file.read())) as depth_im:
                raw=np.asarray(depth_im)
        demo=request.form.get('demo')=='on'
        mode=request.form.get('mode','full')
        if mode not in {'full','summary'}:raise ValueError('Invalid narration mode.')
        with lock:
            output=service.analyze(rgb,raw,mode,demo=demo)
        if '_demo_depth' in output:
            synthetic=output.pop('_demo_depth')
            depth_buffer=BytesIO(); Image.fromarray(synthetic).save(depth_buffer,format='PNG')
            output['demo_depth_png']='data:image/png;base64,'+base64.b64encode(depth_buffer.getvalue()).decode('ascii')
        if '_overlay' in output:
            overlay=output.pop('_overlay')
            buffer=BytesIO();Image.fromarray(overlay).save(buffer,format='PNG')
            output['overlay']='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
        return jsonify(output)
    except (ValueError,UnidentifiedImageError) as exc:
        return jsonify({'error':str(exc)}),400
    except FileNotFoundError as exc:
        return jsonify({'error':f'Model/data file missing: {exc}'}),503
    except Exception:
        app.logger.exception('Analysis failed')
        return jsonify({'error':'Analysis failed. Check the VS Code terminal for the full error; verify model compatibility and inputs.'}),500

@app.errorhandler(RequestEntityTooLarge)
def too_large(exc):return jsonify({'error':'Combined upload exceeds 18 MB.'}),413

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5050,debug=False)
