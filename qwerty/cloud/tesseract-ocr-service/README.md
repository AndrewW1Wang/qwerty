Tesseract OCR (Masked Preview) on Cloud Run
==========================================

What this is
------------
Minimal Flask service that accepts an image (WebP/JPEG/PNG) in the body, runs Tesseract, and returns digits/"@" only (and full text for debugging). No Vision API cost.

Deploy
------

```bash
gcloud auth login
gcloud config set project PROJECT_ID

gcloud builds submit --tag gcr.io/PROJECT_ID/tesseract-ocr
gcloud run deploy tesseract-ocr \
  --image gcr.io/PROJECT_ID/tesseract-ocr \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars RESPONSE_MODE=digits
```

Copy the URL (https://tesseract-ocr-xxxxx-uc.a.run.app) and paste into the app’s Cloud OCR URL field.

Cost
----
- No per-image OCR charges (Tesseract).
- Cloud Run compute is usually inside free tier at 0.1–0.5 fps.

Local test
----------

```bash
docker build -t local-tess .
docker run -p 8080:8080 local-tess
```

Then POST an image:

```bash
curl -s -X POST -H "Content-Type: image/webp" --data-binary @sample.webp http://localhost:8080/ocr-masked | jq
```


