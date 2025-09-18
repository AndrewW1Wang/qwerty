import express from 'express';
import cors from 'cors';
import vision from '@google-cloud/vision';

// Config via env
// - GCP authentication: use default service account on Cloud Run or set GOOGLE_APPLICATION_CREDENTIALS locally
// - ALLOWED_ORIGIN: optional CORS allowlist (comma-separated)
// - RESPONSE_MODE: 'digits' | 'full' (default 'digits')

const app = express();
app.use(cors({
  origin: (origin, cb) => {
    const allow = process.env.ALLOWED_ORIGIN;
    if (!allow) return cb(null, true);
    const list = allow.split(',').map(s => s.trim());
    if (!origin || list.includes(origin)) return cb(null, true);
    cb(new Error('Not allowed by CORS'));
  }
}));

// Accept raw image/webp payload up to ~2MB
app.use((req, res, next) => {
  const ct = req.headers['content-type'] || '';
  if (ct.startsWith('image/')) {
    const chunks = [];
    req.on('data', (d) => chunks.push(d));
    req.on('end', () => { req.rawBody = Buffer.concat(chunks); next(); });
  } else { next(); }
});

const client = new vision.ImageAnnotatorClient();

app.post(['/ocr-masked', '/'], async (req, res) => {
  try {
    if (!req.rawBody || !Buffer.isBuffer(req.rawBody)) {
      return res.status(400).json({ error: 'expected image body' });
    }
    const [result] = await client.textDetection({ image: { content: req.rawBody } });
    const detections = result.textAnnotations || [];
    const fullText = (detections[0] && detections[0].description) ? detections[0].description : '';
  // try to find '@' bounding box
  let atBox = null;
  for (let i = 1; i < detections.length; i++) {
    const d = detections[i];
    const desc = (d.description || '').toString();
    if (!desc.includes('@')) continue;
    const v = (d.boundingPoly && d.boundingPoly.vertices) || [];
    if (v.length) {
      const xs = v.map(p => p.x || 0); const ys = v.map(p => p.y || 0);
      const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys);
      atBox = { x0, y0, x1, y1 };
      break;
    }
  }
    const mode = (process.env.RESPONSE_MODE || 'digits').toLowerCase();
    if (mode === 'full') {
    return res.json({ text: fullText, atBox });
    }
  const digits = (fullText || '').replace(/[^0-9@-]/g, '');
  res.json({ digits, text: fullText, atBox });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'ocr_failed' });
  }
});

app.get('/healthz', (req, res) => res.json({ ok: true }));

const port = process.env.PORT || 8080;
app.listen(port, () => console.log('OCR service listening on', port));


