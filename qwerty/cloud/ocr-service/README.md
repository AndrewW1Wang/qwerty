Cloud OCR (Masked Preview) – Google Vision on Cloud Run
=======================================================

What this is
------------
Minimal Express service that accepts an image (WebP/JPEG/PNG) in the request body, calls Google Cloud Vision Text Detection, and returns the digits/"@"-only string (and the full text for debugging).

You deploy this to Cloud Run. Your webapp posts masked preview snapshots to it at a low fps (0.2–1 fps) to keep costs low.

Costs
-----
- Vision OCR is billed per image/request. Ballpark ~$1–$2 per 1,000 images.
- Use the client's fps and de-dup to minimize calls.
- Cloud Run itself is effectively free at this scale.

Prerequisites
-------------
1) Install gcloud CLI and log in.
2) Create a GCP project and enable these APIs:
   - Cloud Vision API
   - Cloud Run Admin API
   - Artifact Registry API
3) Grant Cloud Run service account permission to use Vision (roles/vision.user) or keep the default if using the default Compute service account.

Quickstart (from this folder)
-----------------------------
1) Build and deploy to Cloud Run:

   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/masked-ocr
   gcloud run deploy masked-ocr \
     --image gcr.io/PROJECT_ID/masked-ocr \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars RESPONSE_MODE=digits
   ```

   Replace PROJECT_ID and region.

2) Copy the service URL printed by Cloud Run (looks like https://masked-ocr-xxxxx-uc.a.run.app).

3) In the web app, paste it into the Cloud OCR URL field and turn on the toggle.

Optional hardening
------------------
- Set ALLOWED_ORIGIN to your site origin to restrict CORS:

  ```bash
  gcloud run services update masked-ocr \
    --region us-central1 \
    --set-env-vars ALLOWED_ORIGIN=https://your-site.example
  ```

- If you want to call Vision using an API key (not recommended from server-to-server), you can instead use the REST API directly. This sample uses the Vision SDK with service account auth provided by Cloud Run by default.

Local testing
-------------

```bash
npm i
node index.js
```

Then POST an image:

```bash
curl -s -X POST -H "Content-Type: image/webp" --data-binary @sample.webp http://localhost:8080/ocr-masked | jq
```


