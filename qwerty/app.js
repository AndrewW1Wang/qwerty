(() => {
  const watchBtn = document.getElementById('watchBtn');
  const resetBtn = document.getElementById('resetBtn');
  const sourceVideo = document.getElementById('sourceVideo');
  const previewCanvas = document.getElementById('previewCanvas');
  const overlayCanvas = document.getElementById('overlayCanvas');
  const previewWrap = document.getElementById('previewWrap');
  const trackerWrap = document.getElementById('trackerWrap');
  const trackCanvas = document.getElementById('trackCanvas');
  const trackOverlayCanvas = document.getElementById('trackOverlayCanvas');
  const stableWrap = document.getElementById('stableWrap');
  const stableCanvas = document.getElementById('stableCanvas');
  const maskWrap = document.getElementById('maskWrap');
  const maskCanvas = document.getElementById('maskCanvas');
  const maskStrengthRange = document.getElementById('maskStrengthRange');
  const positionCanvas = document.getElementById('positionCanvas');
  const stableOffsetX = document.getElementById('stableOffsetX');
  const stableOffsetXVal = document.getElementById('stableOffsetXVal');
  // Cloud OCR UI
  const cloudOcrToggle = document.getElementById('cloudOcrToggle');
  const cloudOcrFps = document.getElementById('cloudOcrFps');
  const cloudOcrUrl = document.getElementById('cloudOcrUrl');
  const cloudOcrStatus = document.getElementById('cloudOcrStatus');
  const cloudOcrOutput = document.getElementById('cloudOcrOutput');
  const cloudVisionStat = document.getElementById('cloudVisionStat');
  const fpsSelect = document.getElementById('fpsSelect');
  const gpuToggle = document.getElementById('gpuToggle');
  const showBoxesToggle = document.getElementById('showBoxesToggle');
  const lockToggle = document.getElementById('lockToggle');
  const reacquireBtn = document.getElementById('reacquireBtn');
  const contrastRange = document.getElementById('contrastRange');
  const thresholdRange = document.getElementById('thresholdRange');
  const binarizeToggle = document.getElementById('binarizeToggle');
  const sharpenRange = document.getElementById('sharpenRange');
  const flattenToggle = document.getElementById('flattenToggle');
  const levelsRange = document.getElementById('levelsRange');
  const applyPreviewToggle = document.getElementById('applyPreviewToggle');

  /** @type {CanvasRenderingContext2D} */
  const previewCtx = previewCanvas.getContext('2d', { alpha: false });
  /** @type {CanvasRenderingContext2D} */
  const overlayCtx = overlayCanvas.getContext('2d');
  /** @type {CanvasRenderingContext2D} */
  const trackCtx = trackCanvas.getContext('2d', { alpha: false });
  /** @type {CanvasRenderingContext2D} */
  const trackOverlayCtx = trackOverlayCanvas.getContext('2d');
  /** @type {CanvasRenderingContext2D} */
  const stableCtx = stableCanvas.getContext('2d', { alpha: false });
  /** @type {CanvasRenderingContext2D} */
  const maskCtx = maskCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
  const positionCtx = positionCanvas ? positionCanvas.getContext('2d', { alpha: false, willReadFrequently: true }) : null;
  const visibilityState = {
    suspended: false,
    hiddenAt: 0,
    needsResume: false,
    resumeBox: null,
    resumeCenter: null,
    resumeExact: false
  };
  let visibilityOcrSuspended = false;
  let posWinLastAt = 0;
  const posWinIntervalMs = 1000 / 6; // ~6 fps to keep main thread light
  let maskLastAt = 0;
  const maskIntervalMs = 1000 / 10; // ~10 fps for masked processing

  let mediaStream = null;
  let animationFrameId = null;
  let cropRect = null; // { x, y, w, h } in source video space
  let ocrWorker = null;
  let ocrActive = false;
  let ocrBusy = false; // true while Tesseract.recognize is running
  // OCR source canvas (2D) we feed into tesseract
  const ocrCanvas = document.createElement('canvas');
  const ocrCtx = ocrCanvas.getContext('2d', { willReadFrequently: true });
  let ocrScaleX = 1, ocrScaleY = 1;
  let ocrOffsetX = 0, ocrOffsetY = 0; // ROI offset in trackCanvas space
  let ocrRoi = null; // {x,y,w,h} in trackCanvas space
  let ocrNoHitCount = 0;
  const roiPadPx = 40;
  const roiMaxMisses = 8;
  const roiMinSize = 80;
  const roiGrowthFactor = 1.6;
  let ocrIntervalMs = 1000 / 4; // default 4 fps
  let useGpuPreprocess = !!(gpuToggle && gpuToggle.checked);
  let showOcrBoxes = !!(showBoxesToggle && showBoxesToggle.checked);
  let lockExact = !!(lockToggle && lockToggle.checked);
  let glContrast = parseFloat((contrastRange && contrastRange.value) || '1.6');
  let glThreshold = parseFloat((thresholdRange && thresholdRange.value) || '0.5');
  let glBinarize = !!(binarizeToggle && binarizeToggle.checked);
  let lastLockExactApplied = null;
  let glSharpen = 0; // disabled
  let glFlatten = false;
  let glLevels = 3;
  let applyToPreview = false;
  // Separate masked OCR removed (revert)
  // Cloud OCR state
  let cloudOcrActive = false;
  let cloudOcrIntervalMs = 2000; // default 0.5 fps
  let cloudOcrTimer = null;
  let lastSig = null;
  let lastSentAt = 0;

  // WebGL state for GPU pre-processing (grayscale + contrast/threshold)
  let gl = null, glCanvas = null, glProgram = null, glTex = null, glBuffers = null, glUniforms = null, glReady = false;

  // Template tracking state (runs fast between OCR passes)
  const tmplCanvas = document.createElement('canvas');
  const tmplCtx = tmplCanvas.getContext('2d', { willReadFrequently: true });
  const searchCanvas = document.createElement('canvas');
  const searchCtx = searchCanvas.getContext('2d', { willReadFrequently: true });
  let trackerActive = false;
  let trackerTimer = null;
  const trackerFps = 20; // fast local tracking
  const trackerStep = 1; // pixel step per candidate (denser search)
  let template = null; // {data:Uint8ClampedArray,w,h,mean}
  let trackBox = null; // {x,y,w,h}
  let lostFrames = 0;
  // Logging of exact (blue) detections
  const positionLog = [];
  const MAX_LOG_ENTRIES = 300;
  let lastExactCenter = null;
  let lastExactBox = null; // {x,y,w,h} in trackCanvas space

  function addPositionLog(x, y) {
    positionLog.push({ t: Date.now(), x, y });
    if (positionLog.length > MAX_LOG_ENTRIES) positionLog.shift();
    try { console.debug('[POSITION_CENTER]', x, y); } catch {}
  }

  function drawPositionOverlay() {
    if (!positionLog.length) return;
    const dpr = window.devicePixelRatio || 1;
    const ctx = trackOverlayCtx;
    ctx.save();
    // Crosshair at last exact center
    if (lastExactCenter) {
      ctx.strokeStyle = 'rgba(0, 209, 255, 0.9)';
      ctx.lineWidth = 1.5 * dpr;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(lastExactCenter.x - 6 * dpr, lastExactCenter.y);
      ctx.lineTo(lastExactCenter.x + 6 * dpr, lastExactCenter.y);
      ctx.moveTo(lastExactCenter.x, lastExactCenter.y - 6 * dpr);
      ctx.lineTo(lastExactCenter.x, lastExactCenter.y + 6 * dpr);
      ctx.stroke();
    }
    // Recent coordinates (last 8)
    const lines = Math.min(8, positionLog.length);
    const start = positionLog.length - lines;
    ctx.font = `${Math.round(11 * dpr)}px ui-sans-serif`;
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.textBaseline = 'top';
    const pad = Math.round(6 * dpr);
    for (let i = 0; i < lines; i++) {
      const e = positionLog[start + i];
      ctx.fillText(`${e.x}, ${e.y}`, pad, pad + i * Math.round(14 * dpr));
    }
    ctx.restore();
  }

  function setUiCapturing(isCapturing) {
    document.body.classList.toggle('not-capturing', !isCapturing);
    resetBtn.disabled = !isCapturing;
    if (watchBtn) {
      watchBtn.textContent = isCapturing ? 'Stop' : 'Watch';
    }
  }
  // Stable horizontal offset control
  let stableAnchorXOffset = 0; // -0.5 .. 0.5
  if (stableOffsetX) {
    const updateStableOffset = () => {
      stableAnchorXOffset = Math.max(-0.5, Math.min(0.5, parseFloat(stableOffsetX.value || '0') || 0));
      if (stableOffsetXVal) stableOffsetXVal.textContent = stableAnchorXOffset.toFixed(2);
    };
    stableOffsetX.addEventListener('input', updateStableOffset);
    updateStableOffset();
  }

  function resizeCanvases() {
    const dpr = window.devicePixelRatio || 1;
    const rect1 = previewWrap.getBoundingClientRect();
    previewCanvas.width = Math.max(1, Math.floor(rect1.width * dpr));
    previewCanvas.height = Math.max(1, Math.floor(rect1.height * dpr));
    overlayCanvas.width = previewCanvas.width;
    overlayCanvas.height = previewCanvas.height;

    if (trackerWrap) {
      const rect2 = trackerWrap.getBoundingClientRect();
      trackCanvas.width = Math.max(1, Math.floor(rect2.width * dpr));
      trackCanvas.height = Math.max(1, Math.floor(rect2.height * dpr));
      trackOverlayCanvas.width = trackCanvas.width;
      trackOverlayCanvas.height = trackCanvas.height;
    }

    if (stableWrap) {
      const rect3 = stableWrap.getBoundingClientRect();
      stableCanvas.width = Math.max(1, Math.floor(rect3.width * dpr));
      stableCanvas.height = Math.max(1, Math.floor(rect3.height * dpr));
    }

    if (maskWrap) {
      const rect4 = maskWrap.getBoundingClientRect();
      maskCanvas.width = Math.max(1, Math.floor(rect4.width * dpr));
      maskCanvas.height = Math.max(1, Math.floor(rect4.height * dpr));
    }

    if (positionCanvas) {
      const rect5 = positionCanvas.getBoundingClientRect();
      positionCanvas.width = Math.max(1, Math.floor(rect5.width * dpr));
      positionCanvas.height = Math.max(1, Math.floor(rect5.height * dpr));
    }
  }

  function stopStream() {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    visibilityState.suspended = false;
    visibilityState.needsResume = false;
    visibilityState.resumeBox = null;
    visibilityState.resumeCenter = null;
    visibilityState.resumeExact = false;
    visibilityOcrSuspended = false;
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }
    sourceVideo.srcObject = null;
    cropRect = null;
    setUiCapturing(false);
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    trackCtx.clearRect(0, 0, trackCanvas.width, trackCanvas.height);
    trackOverlayCtx.clearRect(0, 0, trackOverlayCanvas.width, trackOverlayCanvas.height);
    stopOcr();
    stopCloudOcr();
  }

  async function startCapture() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          displaySurface: 'browser',
          frameRate: { ideal: 60 },
          cursor: 'always'
        },
        audio: false
      });
      mediaStream = stream;
      sourceVideo.srcObject = stream;
      await sourceVideo.play();
      setUiCapturing(true);
      loop();
      // warm up OCR in background
      ensureOcrWorker().then(() => startOcr());

      const [videoTrack] = stream.getVideoTracks();
      videoTrack.addEventListener('ended', () => {
        stopStream();
      });
    } catch (err) {
      console.error(err);
      alert('Screen capture was blocked or failed.');
    }
  }

  function computeSourceCrop() {
    const vidW = sourceVideo.videoWidth || 1;
    const vidH = sourceVideo.videoHeight || 1;
    if (!cropRect) {
      return { x: 0, y: 0, w: vidW, h: vidH };
    }
    const x = Math.max(0, Math.min(cropRect.x, vidW - 1));
    const y = Math.max(0, Math.min(cropRect.y, vidH - 1));
    const w = Math.max(1, Math.min(cropRect.w, vidW - x));
    const h = Math.max(1, Math.min(cropRect.h, vidH - y));
    return { x, y, w, h };
  }

  function drawZoomToCanvas(ctx, canvas) {
    const cw = canvas.width;
    const ch = canvas.height;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, cw, ch);

    if (!mediaStream || sourceVideo.readyState < 2) return;

    const crop = computeSourceCrop();
    const srcAR = crop.w / crop.h;
    const dstAR = cw / ch;

    let dw, dh, dx, dy;
    if (srcAR > dstAR) {
      dw = cw; dh = Math.round(cw / srcAR); dx = 0; dy = Math.round((ch - dh) / 2);
    } else {
      dh = ch; dw = Math.round(ch * srcAR); dy = 0; dx = Math.round((cw - dw) / 2);
    }

    try {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(sourceVideo, crop.x, crop.y, crop.w, crop.h, dx, dy, dw, dh);
    } catch {}
  }

  function draw() {
    drawZoomToCanvas(previewCtx, previewCanvas);
    drawZoomToCanvas(trackCtx, trackCanvas);
    // Optionally apply GPU processed image back to tracker and preview for visual aid
    if (useGpuPreprocess && applyToPreview && initGl() && trackCanvas.width > 0 && trackCanvas.height > 0) {
      const maxW = 960, maxH = 540;
      const srcW = trackCanvas.width, srcH = trackCanvas.height;
      const scale = Math.min(1, maxW / srcW, maxH / srcH);
      const dstW = Math.max(1, Math.floor(srcW * scale));
      const dstH = Math.max(1, Math.floor(srcH * scale));
      glCanvas.width = dstW; glCanvas.height = dstH;
      gl.viewport(0, 0, dstW, dstH);
      gl.bindTexture(gl.TEXTURE_2D, glTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, trackCanvas);
      gl.useProgram(glProgram);
      gl.uniform1f(glUniforms.threshold, Math.max(0.0, Math.min(1.0, glThreshold)));
      gl.uniform1f(glUniforms.contrast, glContrast);
      gl.uniform1i(glUniforms.binarize, glBinarize ? 1 : 0);
      gl.uniform2f(glUniforms.texel, 1.0 / dstW, 1.0 / dstH);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      // draw processed to trackCanvas for display only
      trackCtx.clearRect(0, 0, trackCanvas.width, trackCanvas.height);
      trackCtx.imageSmoothingEnabled = true;
      trackCtx.imageSmoothingQuality = 'high';
      trackCtx.drawImage(glCanvas, 0, 0, trackCanvas.width, trackCanvas.height);
    }

    // Stabilized view: center around lastExactCenter if available
    if (stableCanvas.width > 0 && stableCanvas.height > 0) {
      stableCtx.fillStyle = '#000';
      stableCtx.fillRect(0, 0, stableCanvas.width, stableCanvas.height);
      const crop = computeSourceCrop();
      if (sourceVideo.readyState >= 2) {
        const srcW = crop.w;
        const srcH = crop.h;
        const dstW = stableCanvas.width;
        const dstH = stableCanvas.height;
        // Map lastExactCenter from trackCanvas space back to current crop-relative coords
        let centerX = Math.round(srcW / 2);
        let centerY = Math.round(srcH / 2);
        if (lastExactCenter) {
          // lastExactCenter is in trackCanvas space (after letterboxing). Compute crop draw params to map back
          const cw = trackCanvas.width;
          const ch = trackCanvas.height;
          const srcAR = srcW / srcH;
          const dstAR = cw / ch;
          let dw, dh, dx, dy;
          if (srcAR > dstAR) { dw = cw; dh = Math.round(cw / srcAR); dx = 0; dy = Math.round((ch - dh) / 2); }
          else { dh = ch; dw = Math.round(ch * srcAR); dy = 0; dx = Math.round((cw - dw) / 2); }
          const u = (lastExactCenter.x - dx) / (dw || 1);
          const v = (lastExactCenter.y - dy) / (dh || 1);
          centerX = Math.max(0, Math.min(srcW - 1, Math.round(crop.x + u * srcW) - crop.x));
          centerY = Math.max(0, Math.min(srcH - 1, Math.round(crop.y + v * srcH) - crop.y));
        }
        // Choose a window around the center matching output aspect
        const outAR = dstW / dstH;
        let winW, winH;
        if (outAR > 1) { // wider
          winH = Math.round(Math.min(srcH, Math.max(80, srcH * 0.5)));
          winW = Math.round(winH * outAR);
        } else {
          winW = Math.round(Math.min(srcW, Math.max(80, srcW * 0.5)));
          winH = Math.round(winW / outAR);
        }
        // Anchor horizontally at center (0.5), vertically slightly above top (negative to push up more)
        const anchorX = 0.5 + stableAnchorXOffset; // horizontal offset from center
        const anchorY = -0.08; // shift up a bit more
        let sx = Math.round(centerX - winW * anchorX);
        let sy = Math.round(centerY - winH * anchorY);
        sx = Math.max(0, Math.min(srcW - winW, sx));
        sy = Math.max(0, Math.min(srcH - winH, sy));
        try {
          stableCtx.imageSmoothingEnabled = true;
          stableCtx.imageSmoothingQuality = 'high';
          stableCtx.drawImage(sourceVideo, crop.x + sx, crop.y + sy, winW, winH, 0, 0, dstW, dstH);
        } catch {}
      }
    }

    // Color mask from stabilized view
    if (stableCanvas.width > 0 && stableCanvas.height > 0 && maskCanvas.width > 0 && maskCanvas.height > 0) {
      // Draw stabilized into mask canvas, then apply color filter
      try {
        const now = performance.now();
        if ((now - maskLastAt) >= maskIntervalMs) {
          maskLastAt = now;
          maskCtx.imageSmoothingEnabled = true;
          maskCtx.imageSmoothingQuality = 'high';
          maskCtx.drawImage(stableCanvas, 0, 0, maskCanvas.width, maskCanvas.height);
          const img = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
          const data = img.data;
          // Colors of interest (linear-ish sRGB in 0-255)
          const targets = [
            [0xE1, 0xE3, 0xE8], // white-ish
            [0x4E, 0x9A, 0x5C], // green
            [0xE8, 0x26, 0x48]  // red
          ];
          const strength = Math.min(100, Math.max(0, parseInt((maskStrengthRange && maskStrengthRange.value) || '35', 10) || 35));
          // Strength mapped to tolerance radius in color space (~Euclidean in RGB)
          const tol = 10 + Math.round(strength * 2.0); // 10..210
          const tolSq = tol * tol;
          for (let i = 0; i < data.length; i += 4) {
            const r = data[i], g = data[i+1], b = data[i+2];
            let keep = false;
            for (let t = 0; t < targets.length; t++) {
              const dr = r - targets[t][0];
              const dg = g - targets[t][1];
              const db = b - targets[t][2];
              if ((dr*dr + dg*dg + db*db) <= tolSq) { keep = true; break; }
            }
            if (!keep) { data[i] = 0; data[i+1] = 0; data[i+2] = 0; }
          }
          maskCtx.putImageData(img, 0, 0);
        }

        // Removed OCR on masked preview per revert request
      } catch {}
    }

    // Position window: whiten non-black pixels from masked view
    if (positionCanvas && maskCanvas.width > 0 && maskCanvas.height > 0) {
      try {
        const now = performance.now();
        if ((now - posWinLastAt) >= posWinIntervalMs) {
          posWinLastAt = now;
          positionCtx.imageSmoothingEnabled = true;
          positionCtx.imageSmoothingQuality = 'high';
          // Draw from maskCanvas into positionCanvas keeping aspect
          positionCtx.clearRect(0, 0, positionCanvas.width, positionCanvas.height);
          // letterbox-fit
          const srcW = maskCanvas.width, srcH = maskCanvas.height;
          const dstW = positionCanvas.width, dstH = positionCanvas.height;
          const srcAR = srcW / srcH, dstAR = dstW / dstH;
          let dw, dh, dx, dy;
          if (srcAR > dstAR) { dw = dstW; dh = Math.round(dstW / srcAR); dx = 0; dy = Math.round((dstH - dh) / 2); }
          else { dh = dstH; dw = Math.round(dstH * srcAR); dy = 0; dx = Math.round((dstW - dw) / 2); }
          // Read mask image, convert any non-black to white
          const bufCanvas = document.createElement('canvas');
          bufCanvas.width = srcW; bufCanvas.height = srcH;
          const bufCtx = bufCanvas.getContext('2d', { willReadFrequently: true });
          bufCtx.drawImage(maskCanvas, 0, 0);
          const img = bufCtx.getImageData(0, 0, srcW, srcH);
          const d = img.data;
          for (let i = 0; i < d.length; i += 4) {
            if (d[i] | d[i+1] | d[i+2]) { d[i] = 255; d[i+1] = 255; d[i+2] = 255; } // non-black -> white
          }
          bufCtx.putImageData(img, 0, 0);
          positionCtx.drawImage(bufCanvas, 0, 0, srcW, srcH, dx, dy, dw, dh);
        }
      } catch {}
    }
  }

  function loop() {
    draw();
    animationFrameId = requestAnimationFrame(loop);
  }

  function resetView() {
    cropRect = null;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    trackOverlayCtx.clearRect(0, 0, trackOverlayCanvas.width, trackOverlayCanvas.height);
  }

  // Selection handling on overlayCanvas in canvas pixel space, mapped to source video space
  let isDragging = false;
  let dragStart = null; // {x,y} in overlay canvas pixels

  function canvasToVideoSpace(point) {
    const cw = previewCanvas.width;
    const ch = previewCanvas.height;
    const crop = computeSourceCrop();
    const srcAR = crop.w / crop.h;
    const dstAR = cw / ch;

    let dw, dh, dx, dy;
    if (srcAR > dstAR) {
      dw = cw; dh = Math.round(cw / srcAR); dx = 0; dy = Math.round((ch - dh) / 2);
    } else {
      dh = ch; dw = Math.round(ch * srcAR); dy = 0; dx = Math.round((cw - dw) / 2);
    }

    const within = {
      x: Math.max(0, Math.min(point.x - dx, dw)),
      y: Math.max(0, Math.min(point.y - dy, dh))
    };
    const u = within.x / (dw || 1);
    const v = within.y / (dh || 1);
    return {
      x: Math.round(crop.x + u * crop.w),
      y: Math.round(crop.y + v * crop.h)
    };
  }

  function drawOverlayRect(a, b) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.save();
    overlayCtx.strokeStyle = 'rgba(79,140,255,1)';
    overlayCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
    overlayCtx.setLineDash([8, 6]);
    overlayCtx.fillStyle = 'rgba(79,140,255,0.15)';
    const x = Math.min(a.x, b.x);
    const y = Math.min(a.y, b.y);
    const w = Math.abs(a.x - b.x);
    const h = Math.abs(a.y - b.y);
    overlayCtx.fillRect(x, y, w, h);
    overlayCtx.strokeRect(x, y, w, h);
    overlayCtx.restore();
  }

  overlayCanvas.addEventListener('mousedown', (e) => {
    if (!mediaStream) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    dragStart = {
      x: Math.round((e.clientX - rect.left) * dpr),
      y: Math.round((e.clientY - rect.top) * dpr)
    };
    isDragging = true;
    document.body.classList.add('selecting');
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const rect = overlayCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cur = {
      x: Math.round((e.clientX - rect.left) * dpr),
      y: Math.round((e.clientY - rect.top) * dpr)
    };
    drawOverlayRect(dragStart, cur);
  });

  window.addEventListener('mouseup', (e) => {
    if (!isDragging) return;
    isDragging = false;
    document.body.classList.remove('selecting');
    const rect = overlayCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const dragEnd = {
      x: Math.round((e.clientX - rect.left) * dpr),
      y: Math.round((e.clientY - rect.top) * dpr)
    };
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const minSel = 10 * dpr;
    if (Math.abs(dragEnd.x - dragStart.x) < minSel || Math.abs(dragEnd.y - dragStart.y) < minSel) {
      return; // ignore tiny selections
    }

    const p1 = canvasToVideoSpace(dragStart);
    const p2 = canvasToVideoSpace(dragEnd);
    const x = Math.min(p1.x, p2.x);
    const y = Math.min(p1.y, p2.y);
    const w = Math.max(1, Math.abs(p1.x - p2.x));
    const h = Math.max(1, Math.abs(p1.y - p2.y));
    cropRect = { x, y, w, h };
  });

  function handleVisibilityChange() {
    if (document.hidden) {
      visibilityState.hiddenAt = Date.now();
      visibilityState.suspended = true;
      visibilityOcrSuspended = true;
      if (!mediaStream) {
        visibilityState.needsResume = false;
        return;
      }
      visibilityState.needsResume = true;
      if (lastExactBox) {
        visibilityState.resumeBox = { ...lastExactBox };
        visibilityState.resumeExact = true;
        visibilityState.resumeCenter = lastExactCenter ? { ...lastExactCenter } : null;
      } else if (trackBox) {
        visibilityState.resumeBox = { ...trackBox };
        visibilityState.resumeExact = false;
        visibilityState.resumeCenter = null;
      } else {
        visibilityState.resumeBox = null;
        visibilityState.resumeExact = false;
        visibilityState.resumeCenter = null;
      }
      if (trackerActive) {
        stopTemplateTracking();
      }
    } else {
      visibilityState.suspended = false;
      visibilityOcrSuspended = false;
      if (!mediaStream || !visibilityState.needsResume) {
        visibilityState.needsResume = false;
        visibilityState.resumeBox = null;
        visibilityState.resumeCenter = null;
        visibilityState.resumeExact = false;
        return;
      }
      maskLastAt = 0;
      posWinLastAt = 0;
      const hiddenDuration = visibilityState.hiddenAt ? (Date.now() - visibilityState.hiddenAt) : 0;
      const resumeBox = visibilityState.resumeBox ? { ...visibilityState.resumeBox } : null;
      const resumeExact = visibilityState.resumeExact && !!resumeBox;
      if (resumeExact) {
        lastExactBox = { ...resumeBox };
        const center = visibilityState.resumeCenter || {
          x: resumeBox.x + Math.round(resumeBox.w / 2),
          y: resumeBox.y + Math.round(resumeBox.h / 2)
        };
        lastExactCenter = center;
        let pad = roiPadPx;
        if (hiddenDuration > 8000) pad = Math.round(pad * 2.5);
        else if (hiddenDuration > 4000) pad = Math.round(pad * 1.5);
        ocrRoi = clampRect(
          resumeBox.x - pad,
          resumeBox.y - pad,
          resumeBox.w + pad * 2,
          resumeBox.h + pad * 2,
          trackCanvas.width,
          trackCanvas.height
        );
        ocrNoHitCount = 0;
        drawPositionOverlay();
      } else {
        ocrRoi = null;
        ocrNoHitCount = 0;
      }
      if (resumeBox) {
        if (!resumeExact && hiddenDuration > 6000) {
          resumeBox.x = Math.max(0, resumeBox.x - Math.round(resumeBox.w * 0.35));
          resumeBox.y = Math.max(0, resumeBox.y - Math.round(resumeBox.h * 0.35));
          const maxW = Math.max(1, trackCanvas.width - resumeBox.x);
          const maxH = Math.max(1, trackCanvas.height - resumeBox.y);
          resumeBox.w = Math.min(maxW, Math.max(1, Math.round(resumeBox.w * 1.7)));
          resumeBox.h = Math.min(maxH, Math.max(1, Math.round(resumeBox.h * 1.7)));
        }
        resumeTemplateTracking(resumeBox);
      }
      visibilityState.needsResume = false;
      visibilityState.resumeBox = null;
      visibilityState.resumeCenter = null;
      visibilityState.resumeExact = false;
      visibilityState.hiddenAt = 0;
      if (mediaStream && !animationFrameId) {
        loop();
      }
    }
  }

  watchBtn.addEventListener('click', () => {
    if (mediaStream) {
      stopStream();
    } else {
      startCapture();
    }
  });

  resetBtn.addEventListener('click', () => {
    resetView();
  });

  window.addEventListener('resize', resizeCanvases);
  const ro = new ResizeObserver(resizeCanvases);
  ro.observe(previewWrap);
  resizeCanvases();
  setUiCapturing(false);
  document.addEventListener('visibilitychange', handleVisibilityChange);

  // UI handlers for OCR speed and GPU toggle
  if (fpsSelect) {
    const setFps = () => {
      const v = parseInt(fpsSelect.value, 10);
      if (Number.isFinite(v) && v > 0) {
        ocrIntervalMs = Math.max(10, Math.floor(1000 / v));
      }
    };
    fpsSelect.addEventListener('change', setFps);
    setFps();
  }
  // ---------- Cloud OCR UI ----------
  ;(() => {
    try {
      const savedUrl = localStorage.getItem('cloudOcrUrl');
      if (savedUrl && cloudOcrUrl) cloudOcrUrl.value = savedUrl;
      const savedFps = localStorage.getItem('cloudOcrFps');
      if (savedFps && cloudOcrFps) cloudOcrFps.value = savedFps;
    } catch {}
    const updateRate = () => {
      const v = parseFloat((cloudOcrFps && cloudOcrFps.value) || '0.5');
      if (Number.isFinite(v) && v > 0) cloudOcrIntervalMs = Math.max(250, Math.floor(1000 / v));
      try { localStorage.setItem('cloudOcrFps', (cloudOcrFps && cloudOcrFps.value) || '0.5'); } catch {}
      if (cloudOcrActive) { stopCloudOcr(); startCloudOcr(); }
    };
    if (cloudOcrFps) cloudOcrFps.addEventListener('change', updateRate);
    const updateUrl = () => {
      try { localStorage.setItem('cloudOcrUrl', (cloudOcrUrl && cloudOcrUrl.value) || ''); } catch {}
    };
    if (cloudOcrUrl) cloudOcrUrl.addEventListener('change', updateUrl);
    if (cloudOcrToggle) {
      cloudOcrToggle.addEventListener('change', () => {
        if (cloudOcrToggle.checked) startCloudOcr(); else stopCloudOcr();
      });
    }
    updateRate();
  })();

  function setCloudStatus(msg, ok = true) {
    if (!cloudOcrStatus) return;
    cloudOcrStatus.textContent = msg || '';
    cloudOcrStatus.style.color = ok ? '#9aa0a6' : '#ff6b6b';
  }

  function startCloudOcr() {
    if (cloudOcrActive) return;
    if (!cloudOcrUrl || !cloudOcrUrl.value) { setCloudStatus('Set Cloud Run URL first', false); if (cloudOcrToggle) cloudOcrToggle.checked = false; return; }
    if (!mediaStream) { setCloudStatus('Start Watch first', false); if (cloudOcrToggle) cloudOcrToggle.checked = false; return; }
    cloudOcrActive = true; lastSig = null; lastSentAt = 0; inFlight = false;
    tickCloudOcr();
  }

  function stopCloudOcr() {
    cloudOcrActive = false;
    if (cloudOcrTimer) { clearTimeout(cloudOcrTimer); cloudOcrTimer = null; }
    setCloudStatus('');
  }

  let inFlight = false;
  function scheduleCloudTick() {
    if (!cloudOcrActive) return;
    cloudOcrTimer = setTimeout(tickCloudOcr, cloudOcrIntervalMs);
  }

  async function tickCloudOcr() {
    if (!cloudOcrActive) return;
    try {
      const can = positionCanvas && positionCanvas.width > 0 ? positionCanvas : maskCanvas; // prefer position window
      if (!can || can.width < 2 || can.height < 2) { scheduleCloudTick(); return; }
      // Compute simple signature to de-dup
      const sig = computeSignature(can, 64, 36);
      const now = Date.now();
      const minGap = Math.max(cloudOcrIntervalMs * 0.75, 300);
      if (sig && lastSig && sig === lastSig && (now - lastSentAt) < 5000) { // skip identical frames for up to 5s
        scheduleCloudTick();
        return;
      }
      if (inFlight) { scheduleCloudTick(); return; }
      inFlight = true;
      setCloudStatus('Sending…');
      const blob = await canvasToWebp(can, 0.6);
      const url = cloudOcrUrl.value.trim();
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'image/webp' },
        body: blob
      });
      inFlight = false;
      lastSentAt = now;
      lastSig = sig;
      if (!resp.ok) {
        setCloudStatus(`Error ${resp.status}`, false);
      } else {
        const json = await resp.json().catch(() => null);
        if (json) {
          const raw = (json.digits != null) ? ('' + json.digits) : (json.text || '');
          const cleaned = raw.replace(/[^0-9@-]/g, '');
          const beforeAt = (cleaned.split('@')[0] || '');
          if (cloudOcrOutput) cloudOcrOutput.textContent = beforeAt;
          if (cloudVisionStat) cloudVisionStat.textContent = beforeAt;
          // If we received an @ box, crop position window to rows <= box bottom
          if (json.atBox && positionCanvas && maskCanvas.width > 0) {
            const box = json.atBox; // in source image pixels
            // compute scale from source (can) to positionCanvas letterboxed area
            const srcW = can.width, srcH = can.height;
            const dstW = positionCanvas.width, dstH = positionCanvas.height;
            const srcAR = srcW / srcH, dstAR = dstW / dstH;
            let dw, dh, dx, dy;
            if (srcAR > dstAR) { dw = dstW; dh = Math.round(dstW / srcAR); dx = 0; dy = Math.round((dstH - dh) / 2); }
            else { dh = dstH; dw = Math.round(dstH * srcAR); dy = 0; dx = Math.round((dstW - dw) / 2); }
            const scaleX = dw / srcW; const scaleY = dh / srcH;
            const yCut = dy + Math.round(box.y1 * scaleY);
            // Redraw positionCanvas keeping only rows up to yCut
            try {
              const buf = document.createElement('canvas');
              buf.width = srcW; buf.height = box.y1; // cut at bottom of @
              const bctx = buf.getContext('2d');
              bctx.drawImage(can, 0, 0, srcW, box.y1, 0, 0, srcW, box.y1);
              // whiten non-black like before
              const im = bctx.getImageData(0, 0, buf.width, buf.height); const d = im.data;
              for (let i = 0; i < d.length; i += 4) { if (d[i] | d[i+1] | d[i+2]) { d[i]=255; d[i+1]=255; d[i+2]=255; } }
              bctx.putImageData(im, 0, 0);
              positionCtx.clearRect(0, 0, positionCanvas.width, positionCanvas.height);
              positionCtx.drawImage(buf, 0, 0, buf.width, buf.height, dx, dy, Math.round(buf.width*scaleX), Math.round(buf.height*scaleY));
              // draw a guide line at cut
              positionCtx.strokeStyle = 'rgba(79,140,255,0.8)';
              positionCtx.lineWidth = 1;
              positionCtx.beginPath(); positionCtx.moveTo(0, yCut + 0.5); positionCtx.lineTo(dstW, yCut + 0.5); positionCtx.stroke();
            } catch {}
          }
        }
        setCloudStatus('OK');
      }
    } catch (e) {
      inFlight = false;
      setCloudStatus('Send failed', false);
    }
    scheduleCloudTick();
  }

  function computeSignature(canvas, dw, dh) {
    try {
      const t = computeSignature._c || (computeSignature._c = document.createElement('canvas'));
      const x = computeSignature._x || (computeSignature._x = t.getContext('2d', { willReadFrequently: true }));
      t.width = dw; t.height = dh;
      x.drawImage(canvas, 0, 0, dw, dh);
      const img = x.getImageData(0, 0, dw, dh).data;
      let h1 = 2166136261 >>> 0, h2 = 0x9e3779b9 >>> 0; // simple FNV-like mix
      for (let i = 0; i < img.length; i += 16) {
        const r = img[i], g = img[i+1], b = img[i+2];
        const v = (r * 77 + g * 150 + b * 29) | 0;
        h1 ^= v; h1 = (h1 * 16777619) >>> 0;
        h2 += v; h2 = (h2 ^ (h2 >>> 13)) >>> 0;
      }
      return (h1.toString(16) + ':' + h2.toString(16));
    } catch { return null; }
  }

  function canvasToWebp(canvas, quality) {
    return new Promise((resolve, reject) => {
      try {
        canvas.toBlob((b) => { if (!b) reject(new Error('toBlob failed')); else resolve(b); }, 'image/webp', quality);
      } catch (e) { reject(e); }
    });
  }
  if (gpuToggle) {
    gpuToggle.addEventListener('change', () => {
      useGpuPreprocess = !!gpuToggle.checked;
      if (useGpuPreprocess) initGl();
    });
  }
  if (showBoxesToggle) {
    showBoxesToggle.addEventListener('change', () => {
      showOcrBoxes = !!showBoxesToggle.checked;
      if (!showOcrBoxes) trackOverlayCtx.clearRect(0, 0, trackOverlayCanvas.width, trackOverlayCanvas.height);
    });
  }
  if (lockToggle) {
    lockToggle.addEventListener('change', () => { lockExact = !!lockToggle.checked; });
  }
  if (reacquireBtn) {
    reacquireBtn.addEventListener('click', () => {
      // clear ROI so next OCR is full-frame, and seed tracker off
      ocrRoi = null;
      stopTemplateTracking();
      lastExactCenter = null;
    });
  }
  if (contrastRange) {
    contrastRange.addEventListener('input', () => { glContrast = parseFloat(contrastRange.value || '1.6'); });
  }
  if (thresholdRange) {
    thresholdRange.addEventListener('input', () => {
      const v = parseFloat(thresholdRange.value || '0.45');
      glThreshold = Math.max(-0.25, Math.min(1, isFinite(v) ? v : 0.45));
    });
  }
  if (binarizeToggle) {
    binarizeToggle.addEventListener('change', () => { glBinarize = !!binarizeToggle.checked; });
  }
  // Sharpen/Flatten controls removed

  // ---------- OCR Word Tracker ----------
  async function ensureOcrWorker() {
    if (ocrWorker || !(window.Tesseract && Tesseract.createWorker)) return;
    ocrWorker = Tesseract.createWorker({ logger: () => {} });
    try {
      await ocrWorker.load();
      await ocrWorker.loadLanguage('eng');
      await ocrWorker.initialize('eng');
      // Favor sparse text for UI labels
      await ocrWorker.setParameters({ tessedit_pageseg_mode: '11' });
    } catch (e) {
      console.warn('OCR init failed', e);
      ocrWorker = null;
    }
  }

  function stopOcr() {
    ocrActive = false;
    trackOverlayCtx.clearRect(0, 0, trackOverlayCanvas.width, trackOverlayCanvas.height);
    stopTemplateTracking();
  }

  async function startOcr() {
    if (!ocrWorker || ocrActive) return;
    ocrActive = true;
    let lastFullAt = 0;
    const fullEveryMs = 2500; // periodic full-frame OCR to re-acquire
    while (ocrActive && mediaStream) {
      try {
        if (visibilityOcrSuspended) {
          await new Promise(r => setTimeout(r, 200));
          continue;
        }
        // Update OCR parameters dynamically when lockExact toggles
        if (lastLockExactApplied !== lockExact) {
          lastLockExactApplied = lockExact;
          try {
            if (lockExact) {
              await ocrWorker.setParameters({
                tessedit_pageseg_mode: '11',
                tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                preserve_interword_spaces: '1'
              });
            } else {
              await ocrWorker.setParameters({ tessedit_pageseg_mode: '11' });
            }
          } catch {}
        }
        const nowForceFull = (Date.now() - lastFullAt) >= fullEveryMs;
        const prepared = prepareOcrFrame(nowForceFull);
        if (prepared && !ocrBusy) {
          ocrBusy = true;
          const result = await ocrWorker.recognize(ocrCanvas).catch(() => null);
          ocrBusy = false;
          if (nowForceFull) lastFullAt = Date.now();
          drawTrackerOverlay(result);
        } else {
          await new Promise(r => setTimeout(r, 100));
        }
      } catch (e) {
        // keep loop alive on errors
        ocrBusy = false;
      }
      await new Promise(r => setTimeout(r, ocrIntervalMs));
    }
  }

  // Masked OCR removed (revert)

  // Build WebGL pipeline once
  function initGl() {
    if (glReady) return true;
    glCanvas = document.createElement('canvas');
    gl = (
      glCanvas.getContext('webgl2', { preserveDrawingBuffer: true, powerPreference: 'high-performance' }) ||
      glCanvas.getContext('webgl', { preserveDrawingBuffer: true, powerPreference: 'high-performance' })
    );
    if (!gl) {
      glReady = false;
      return false;
    }
    const vsSrc = `
      attribute vec2 a_pos; attribute vec2 a_uv; varying vec2 v_uv;
      void main(){ v_uv=a_uv; gl_Position=vec4(a_pos,0.0,1.0); }
    `;
    const fsSrc = `
      precision mediump float; varying vec2 v_uv; uniform sampler2D u_tex;
      uniform float u_threshold; uniform float u_contrast; uniform int u_binarize; uniform vec2 u_texel;
      void main(){
        vec3 c = texture2D(u_tex, v_uv).rgb;
        float g = dot(c, vec3(0.299,0.587,0.114));
        g = (g - 0.5) * u_contrast + 0.5;
        float bw = step(u_threshold, g);
        float outv = (u_binarize==1) ? bw : g;
        gl_FragColor = vec4(outv, outv, outv, 1.0);
      }
    `;
    const compile = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      return s;
    };
    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    glProgram = gl.createProgram(); gl.attachShader(glProgram, vs); gl.attachShader(glProgram, fs); gl.linkProgram(glProgram);
    gl.useProgram(glProgram);
    // quad
    const posLoc = gl.getAttribLocation(glProgram, 'a_pos');
    const uvLoc = gl.getAttribLocation(glProgram, 'a_uv');
    const posBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1,  1, -1, -1,  1,
       1, -1,  1,  1, -1,  1
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(posLoc); gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    const uvBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
       0, 1,  1, 1, 0, 0,
       1, 1,  1, 0, 0, 0
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(uvLoc); gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 0, 0);
    // texture
    glTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, glTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(gl.getUniformLocation(glProgram, 'u_tex'), 0);
    glUniforms = {
      threshold: gl.getUniformLocation(glProgram, 'u_threshold'),
      contrast: gl.getUniformLocation(glProgram, 'u_contrast'),
      binarize: gl.getUniformLocation(glProgram, 'u_binarize'),
      texel: gl.getUniformLocation(glProgram, 'u_texel')
    };
    glReady = true;
    return true;
  }

  function clampRect(x, y, w, h, maxW, maxH) {
    const cx = Math.max(0, Math.min(x, Math.max(0, maxW - 1)));
    const cy = Math.max(0, Math.min(y, Math.max(0, maxH - 1)));
    const cw = Math.max(1, Math.min(w, maxW - cx));
    const ch = Math.max(1, Math.min(h, maxH - cy));
    return { x: cx, y: cy, w: cw, h: ch };
  }

  function prepareOcrFrame(forceFull = false) {
    if (trackCanvas.width < 2 || trackCanvas.height < 2) return false;
    // ROI: use last known region with padding; else whole frame
    let srcRect = (!forceFull && ocrRoi) ? { ...ocrRoi } : { x: 0, y: 0, w: trackCanvas.width, h: trackCanvas.height };
    srcRect = clampRect(srcRect.x, srcRect.y, srcRect.w, srcRect.h, trackCanvas.width, trackCanvas.height);
    // Smaller target size for faster OCR
    const maxW = 512, maxH = 288;
    const scale = Math.min(1, maxW / srcRect.w, maxH / srcRect.h);
    const dstW = Math.max(1, Math.floor(srcRect.w * scale));
    const dstH = Math.max(1, Math.floor(srcRect.h * scale));

    if (useGpuPreprocess && initGl()) {
      glCanvas.width = dstW; glCanvas.height = dstH;
      gl.viewport(0, 0, dstW, dstH);
      gl.bindTexture(gl.TEXTURE_2D, glTex);
      // Upload ROI from the tracker canvas
      // Draw ROI to a temporary 2D canvas to feed WebGL
      if (!prepareOcrFrame._roiCanvas) {
        prepareOcrFrame._roiCanvas = document.createElement('canvas');
        prepareOcrFrame._roiCtx = prepareOcrFrame._roiCanvas.getContext('2d', { willReadFrequently: true });
      }
      const rCanvas = prepareOcrFrame._roiCanvas;
      const rCtx = prepareOcrFrame._roiCtx;
      rCanvas.width = srcRect.w; rCanvas.height = srcRect.h;
      rCtx.drawImage(trackCanvas, srcRect.x, srcRect.y, srcRect.w, srcRect.h, 0, 0, srcRect.w, srcRect.h);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, rCanvas);
      gl.useProgram(glProgram);
      // Shift threshold into [0,1] domain even when slider goes below 0.
      gl.uniform1f(glUniforms.threshold, Math.max(0.0, Math.min(1.0, glThreshold)));
      gl.uniform1f(glUniforms.contrast, glContrast);
      gl.uniform1i(glUniforms.binarize, glBinarize ? 1 : 0);
      // uniforms for sharpen/flatten/levels were removed in shader; keep minimal pipeline
      gl.uniform2f(glUniforms.texel, 1.0 / dstW, 1.0 / dstH);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      // Copy to 2D canvas for Tesseract
      ocrCanvas.width = dstW; ocrCanvas.height = dstH;
      ocrCtx.drawImage(glCanvas, 0, 0, dstW, dstH);
    } else {
      ocrCanvas.width = dstW; ocrCanvas.height = dstH;
      ocrCtx.imageSmoothingEnabled = true;
      ocrCtx.imageSmoothingQuality = 'low';
      ocrCtx.drawImage(trackCanvas, srcRect.x, srcRect.y, srcRect.w, srcRect.h, 0, 0, dstW, dstH);
    }
    // Map OCR coords back to tracker canvas
    ocrScaleX = srcRect.w / dstW;
    ocrScaleY = srcRect.h / dstH;
    ocrOffsetX = srcRect.x;
    ocrOffsetY = srcRect.y;
    return true;
  }

  function extractTemplateFromBox(box) {
    const pad = Math.round(Math.max(2, Math.min(box.w, box.h) * 0.15));
    const x = Math.max(0, box.x - pad);
    const y = Math.max(0, box.y - pad);
    const w = Math.min(trackCanvas.width - x, box.w + pad * 2);
    const h = Math.min(trackCanvas.height - y, box.h + pad * 2);
    if (w < 5 || h < 5) return null;
    tmplCanvas.width = w; tmplCanvas.height = h;
    tmplCtx.drawImage(trackCanvas, x, y, w, h, 0, 0, w, h);
    const img = tmplCtx.getImageData(0, 0, w, h);
    const gray = new Uint8ClampedArray(w * h);
    let sum = 0;
    for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
      const r = img.data[i], g = img.data[i+1], b = img.data[i+2];
      const v = (r * 77 + g * 150 + b * 29) >> 8; // 0..255
      gray[j] = v; sum += v;
    }
    const mean = sum / gray.length;
    return { data: gray, w, h, mean };
  }

  function startTemplateTracking(box) {
    template = extractTemplateFromBox(box);
    if (!template) return;
    trackBox = { ...box };
    lostFrames = 0;
    if (!trackerActive) {
      trackerActive = true;
      scheduleTrackerTick();
    }
  }

  function resumeTemplateTracking(box) {
    const target = box ? { ...box } : null;
    if (!target) return;
    let attempts = 0;
    const maxAttempts = 6;
    const tryStart = () => {
      if (!mediaStream || document.hidden) return;
      if (trackCanvas.width < 2 || trackCanvas.height < 2) {
        if (attempts++ < maxAttempts) {
          requestAnimationFrame(tryStart);
        }
        return;
      }
      const safeBox = clampRect(
        target.x,
        target.y,
        target.w,
        target.h,
        trackCanvas.width,
        trackCanvas.height
      );
      startTemplateTracking(safeBox);
    };
    requestAnimationFrame(tryStart);
  }

  function stopTemplateTracking() {
    trackerActive = false;
    if (trackerTimer) { clearTimeout(trackerTimer); trackerTimer = null; }
    template = null; trackBox = null; lostFrames = 0;
  }

  function scheduleTrackerTick() {
    if (!trackerActive) return;
    trackerTimer = setTimeout(trackerTick, Math.max(1, Math.floor(1000 / trackerFps)));
  }

  function trackerTick() {
    if (!trackerActive || !template || !trackBox) { scheduleTrackerTick(); return; }
    // Define a small search window around previous location
    const margin = Math.round(Math.max(10, Math.min(trackCanvas.width, trackCanvas.height) * 0.1));
    const sx = Math.max(0, trackBox.x - margin);
    const sy = Math.max(0, trackBox.y - margin);
    const ex = Math.min(trackCanvas.width - template.w, trackBox.x + margin);
    const ey = Math.min(trackCanvas.height - template.h, trackBox.y + margin);
    if (ex <= sx || ey <= sy) { lostFrames++; scheduleTrackerTick(); return; }

    const sw = ex - sx + 1; const sh = ey - sy + 1;
    searchCanvas.width = sw; searchCanvas.height = sh;
    searchCtx.drawImage(trackCanvas, sx, sy, sw, sh, 0, 0, sw, sh);
    const sImg = searchCtx.getImageData(0, 0, sw, sh);

    // Normalized cross-correlation (approximate, step through pixels)
    let bestScore = -1, bestX = 0, bestY = 0;
    const tData = template.data, tw = template.w, th = template.h, tLen = tw * th, tMean = template.mean;
    const sData = sImg.data;
    const rowStride = sw * 4;
    for (let y = 0; y <= sh - th; y += trackerStep) {
      for (let x = 0; x <= sw - tw; x += trackerStep) {
        let sum = 0, sumSq = 0, cross = 0;
        let idx = y * rowStride + x * 4;
        for (let j = 0; j < th; j++) {
          let idxRow = idx;
          for (let i = 0; i < tw; i++) {
            const r = sData[idxRow], g = sData[idxRow+1], b = sData[idxRow+2];
            const v = (r * 77 + g * 150 + b * 29) >> 8;
            sum += v; sumSq += v * v; cross += (v - 128) * (tData[j * tw + i] - 128);
            idxRow += 4;
          }
          idx += rowStride;
        }
        const mean = sum / tLen;
        const denom = Math.sqrt(Math.max(1, sumSq - tLen * mean * mean)) * Math.sqrt(Math.max(1, tLen * 128 * 128));
        const score = cross / denom;
        if (score > bestScore) { bestScore = score; bestX = x; bestY = y; }
      }
    }

    // Threshold for acceptance
    if (bestScore > 0.12) {
      // Exponential smoothing for steadier motion
      const alpha = 0.4;
      trackBox.x = Math.round((1 - alpha) * trackBox.x + alpha * (sx + bestX));
      trackBox.y = Math.round((1 - alpha) * trackBox.y + alpha * (sy + bestY));
      lostFrames = 0;
      drawTrackedBoxes(trackBox);
    } else {
      lostFrames++;
      // Draw weak candidate to visualize tracking even when low confidence
      const visBox = { x: sx + bestX, y: sy + bestY, w: template.w, h: template.h };
      drawTrackedBoxes(visBox, true);
    }

    // Reacquire with OCR if we've been lost for a while
    if (lostFrames > Math.round(trackerFps * 1.5)) {
      stopTemplateTracking();
    }
    scheduleTrackerTick();
  }

  function drawTrackedBoxes(box, lowConfidence = false) {
    trackOverlayCtx.clearRect(0, 0, trackOverlayCanvas.width, trackOverlayCanvas.height);
    const x = Math.round(box.x), y = Math.round(box.y), w = Math.round(box.w), h = Math.round(box.h);
    trackOverlayCtx.save();
    trackOverlayCtx.strokeStyle = lowConfidence ? 'rgba(255,191,0,1)' : 'rgba(79,140,255,1)';
    trackOverlayCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
    trackOverlayCtx.setLineDash([8, 6]);
    trackOverlayCtx.strokeRect(x, y, w, h);
    const gap = Math.round(0.35 * h);
    const whiteW = Math.round(1.2 * w);
    const whiteH = Math.round(0.65 * h);
    const centerX = x + Math.round(w / 2);
    const whiteX = centerX - Math.round(whiteW / 2);
    const whiteY = y + h + gap;
    trackOverlayCtx.setLineDash([]);
    trackOverlayCtx.strokeStyle = '#ffffff';
    trackOverlayCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
    trackOverlayCtx.strokeRect(whiteX, whiteY, whiteW, whiteH);
    trackOverlayCtx.restore();
    drawPositionOverlay();
  }

  function findBestWordPosition(result) {
    if (!result || !result.data) return null;
    const targetsLoose = ['position', 'pos1tion', 'posit1on', 'postion'];
    const targetsStrict = ['position'];
    const targets = lockExact ? targetsStrict : targetsLoose;
    let best = null;
    let bestAny = null;
    const scan = (arr, getText) => {
      for (const w of arr || []) {
        const text = getText(w).toString().trim().toLowerCase();
        if (!text) continue;
        const normalized = text.replace(/[^a-z0-9]/g, '');
        const isExact = targetsStrict.some(t => normalized === t);
        const isHit = targets.some(t => normalized.includes(t));
        if (isHit) {
          if (!best || (w.confidence || 0) > (best.confidence || 0)) best = w;
        }
        if (!bestAny || (w.confidence || 0) > (bestAny.confidence || 0)) bestAny = w;
        // Prefer exact match if lockExact is enabled
        if (lockExact && isExact) best = w;
      }
    };
    scan(result.data.words, w => w.text || w.word || '');
    if (!best) scan(result.data.lines, l => l.text || '');
    const node = best || (lockExact ? null : bestAny);
    if (!node) return null;
    return { node, exact: best && targetsStrict.some(t => (best.text || best.word || '').toString().trim().toLowerCase().replace(/[^a-z0-9]/g, '') === t) };
  }

  function drawTrackerOverlay(result) {
    trackOverlayCtx.clearRect(0, 0, trackOverlayCanvas.width, trackOverlayCanvas.height);
    const found = findBestWordPosition(result);
    if (!found) {
      // If no match, expand ROI gradually and show a faint amber ROI box to indicate activity
      ocrNoHitCount++;
      growRoiIfNeeded();
      if (ocrRoi && showOcrBoxes) {
        trackOverlayCtx.save();
        trackOverlayCtx.strokeStyle = 'rgba(255,191,0,0.6)';
        trackOverlayCtx.setLineDash([6, 6]);
        trackOverlayCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
        trackOverlayCtx.strokeRect(ocrRoi.x, ocrRoi.y, ocrRoi.w, ocrRoi.h);
        trackOverlayCtx.restore();
      }
      return;
    }
    const wnode = found.node;
    const b = (wnode.bbox || wnode.bbox0 || wnode);
    if (!b || (b.x0 == null || b.x1 == null || b.y0 == null || b.y1 == null)) return;
    const x = Math.round(ocrOffsetX + b.x0 * ocrScaleX);
    const y = Math.round(ocrOffsetY + b.y0 * ocrScaleY);
    const w = Math.max(1, Math.round((b.x1 - b.x0) * ocrScaleX));
    const h = Math.max(1, Math.round((b.y1 - b.y0) * ocrScaleY));

    if (showOcrBoxes) {
      trackOverlayCtx.save();
      // Blue for exact, amber for fuzzy
      trackOverlayCtx.strokeStyle = found.exact ? 'rgba(79,140,255,1)' : 'rgba(255,191,0,1)';
      trackOverlayCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
      trackOverlayCtx.setLineDash([8, 6]);
      trackOverlayCtx.strokeRect(x, y, w, h);
      // White box below, center-aligned
      const gap = Math.round(0.35 * h);
      const whiteW = Math.round(1.2 * w);
      const whiteH = Math.round(0.65 * h);
      const centerX = x + Math.round(w / 2);
      const whiteX = centerX - Math.round(whiteW / 2);
      const whiteY = y + h + gap;
      trackOverlayCtx.setLineDash([]);
      trackOverlayCtx.strokeStyle = '#ffffff';
      trackOverlayCtx.lineWidth = 2 * (window.devicePixelRatio || 1);
      trackOverlayCtx.strokeRect(whiteX, whiteY, whiteW, whiteH);
      trackOverlayCtx.restore();
    }

    // Update ROI and seed tracker
    if (!lockExact || found.exact) {
      startTemplateTracking({ x, y, w, h });
    }
    // Tighten ROI around detection with padding
    const pad = roiPadPx;
    ocrRoi = clampRect(x - pad, y - pad, w + pad * 2, h + pad * 2, trackCanvas.width, trackCanvas.height);
    ocrNoHitCount = 0;
    if (found.exact) {
      const cx = x + Math.round(w / 2);
      const cy = y + Math.round(h / 2);
      lastExactCenter = { x: cx, y: cy };
      lastExactBox = { x, y, w, h };
      addPositionLog(cx, cy);
    }
    drawPositionOverlay();
  }

  function growRoiIfNeeded() {
    if (!ocrRoi) return;
    if (ocrNoHitCount >= roiMaxMisses) { ocrRoi = null; return; }
    const cx = ocrRoi.x + ocrRoi.w / 2;
    const cy = ocrRoi.y + ocrRoi.h / 2;
    let nw = Math.max(roiMinSize, Math.min(trackCanvas.width, Math.floor(ocrRoi.w * roiGrowthFactor)));
    let nh = Math.max(roiMinSize, Math.min(trackCanvas.height, Math.floor(ocrRoi.h * roiGrowthFactor)));
    let nx = Math.round(cx - nw / 2);
    let ny = Math.round(cy - nh / 2);
    ocrRoi = clampRect(nx, ny, nw, nh, trackCanvas.width, trackCanvas.height);
  }
})();


