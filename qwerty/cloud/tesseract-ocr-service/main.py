import io
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": os.environ.get('ALLOWED_ORIGIN', '*').split(',')}})

TESS_LANG = os.environ.get('TESS_LANG', 'eng')
RESPONSE_MODE = os.environ.get('RESPONSE_MODE', 'digits').lower()

@app.route('/ocr-masked', methods=['POST'])
@app.route('/', methods=['POST'])
def ocr_masked():
    if not request.data:
        return jsonify({ 'error': 'expected image body' }), 400
    try:
        img = Image.open(io.BytesIO(request.data)).convert('L')
        # Slight binarize to help OCR on masked images
        img = img.point(lambda p: 255 if p > 160 else 0)
        # Whitelist digits and '@'
        custom = r'-c tessedit_char_whitelist=0123456789@ --psm 6'
        data = pytesseract.image_to_data(img, lang=TESS_LANG, config=custom, output_type=pytesseract.Output.DICT)
        text = ' '.join(data.get('text', [])).strip()
        # find first cell containing '@' and compute its bbox
        atBox = None
        n = len(data.get('text', []))
        for i in range(n):
            if '@' in (data['text'][i] or ''):
                atBox = {
                    'x0': int(data['left'][i]),
                    'y0': int(data['top'][i]),
                    'x1': int(data['left'][i] + data['width'][i]),
                    'y1': int(data['top'][i] + data['height'][i])
                }
                break
        if RESPONSE_MODE == 'full':
            return jsonify({ 'text': text, 'atBox': atBox })
        digits = ''.join([c for c in (text or '') if (c.isdigit() or c in ('@','-'))])
        return jsonify({ 'digits': digits, 'text': text, 'atBox': atBox })
    except Exception:
        return jsonify({ 'error': 'ocr_failed' }), 500

@app.get('/healthz')
def health():
    return jsonify({ 'ok': True })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')))


