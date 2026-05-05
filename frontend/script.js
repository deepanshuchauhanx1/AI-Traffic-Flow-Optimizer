/*
  app.js — Emergency Vehicle Detection Frontend
  ==============================================
  HOW THIS FILE IS ORGANIZED:
  1. CONFIG         → API server URL
  2. STATE          → variables we track
  3. INIT           → runs when page loads
  4. API HEALTH     → checks if backend is running
  5. TAB SWITCHER   → Image / Video tabs
  6. FILE SELECTION → when user picks a file
  7. PREVIEW        → show image/video thumbnail
  8. DRAG AND DROP  → drag files onto the upload zone
  9. ANALYZE IMAGE  → calls POST /detect/image
  10. ANALYZE VIDEO → calls POST /detect/video
  11. RENDER IMAGE RESULTS → builds HTML for image response
  12. RENDER VIDEO RESULTS → builds HTML for video response
  13. HELPERS       → small reusable functions
*/


/* ============================================================
   1. CONFIG
   Change API_BASE if your FastAPI server runs on a different
   port or host (e.g. "http://192.168.1.5:8000")
   ============================================================ */
const API_BASE = "http://localhost:8000";


/* ============================================================
   2. STATE
   These variables remember which files the user selected
   ============================================================ */
let selectedImageFile = null;   // stores the chosen image File object
let selectedVideoFile = null;   // stores the chosen video File object


/* ============================================================
   3. INIT — runs once when the page finishes loading
   ============================================================ */
window.addEventListener("DOMContentLoaded", () => {
  checkAPIHealth();                        // ping /health endpoint
  setupDragDrop("imageZone", "imageFile", "image");  // image drag-drop
  setupDragDrop("videoZone", "videoFile", "video");  // video drag-drop
});


/* ============================================================
   4. API HEALTH CHECK
   Hits GET /health and updates the dot + text in the header.
   AbortSignal.timeout(4000) cancels the fetch after 4 seconds
   so we don't hang forever if the server is off.
   ============================================================ */
async function checkAPIHealth() {
  const dot  = document.getElementById("statusDot");
  const text = document.getElementById("statusText");

  try {
    const res  = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(4000)
    });
    const data = await res.json();

    dot.className   = "status-dot online";
    text.textContent = data.model_loaded
      ? "API ONLINE · MODEL LOADED"
      : "API ONLINE · ⚠ MODEL MISSING";

  } catch {
    // fetch failed = server not running
    dot.className   = "status-dot offline";
    text.textContent = "API OFFLINE — start your server!";
  }
}


/* ============================================================
   5. TAB SWITCHER
   Shows the matching tab panel and marks the button active.
   Called by onclick on each tab button in index.html.
   ============================================================ */
function switchTab(tab) {
  // Update button styles
  document.querySelectorAll(".tab-btn").forEach((btn, index) => {
    btn.classList.toggle("active", (tab === "image" ? index === 0 : index === 1));
  });

  // Show correct panel
  document.querySelectorAll(".tab-panel").forEach((panel, index) => {
    panel.classList.toggle("active", (tab === "image" ? index === 0 : index === 1));
  });
}


/* ============================================================
   6. FILE SELECTION
   Called by onchange on the <input type="file"> elements.
   Saves the file, shows a preview, enables the analyze button.
   ============================================================ */
function onFileSelected(type) {
  const inputId = type === "image" ? "imageFile" : "videoFile";
  const input   = document.getElementById(inputId);
  const file    = input.files[0];

  if (!file) return;  // user cancelled the dialog

  if (type === "image") {
    selectedImageFile = file;
    showImagePreview(file);
    document.getElementById("imageBtn").disabled = false;
    clearResults("image");   // clear any old results
  } else {
    selectedVideoFile = file;
    showVideoPreview(file);
    document.getElementById("videoBtn").disabled = false;
    clearResults("video");
  }
}


/* ============================================================
   7. PREVIEWS
   Creates a local URL from the File object so we can display
   it without uploading it first (URL.createObjectURL).
   ============================================================ */
function showImagePreview(file) {
  const img = document.getElementById("imagePreviewEl");
  img.src   = URL.createObjectURL(file);          // temporary browser URL

  document.getElementById("imagePreviewName").textContent =
    `${file.name}  (${formatSize(file.size)})`;

  document.getElementById("imagePreview").classList.add("visible");
}

function showVideoPreview(file) {
  const video = document.getElementById("videoPreviewEl");
  video.src   = URL.createObjectURL(file);

  document.getElementById("videoPreviewName").textContent =
    `${file.name}  (${formatSize(file.size)})`;

  document.getElementById("videoPreview").classList.add("visible");
}


/* ============================================================
   8. DRAG AND DROP
   Makes the upload zone accept files dragged from File Explorer.
   dragover  — file is hovering over the zone
   dragleave — file left the zone without dropping
   drop      — file was released onto the zone
   ============================================================ */
function setupDragDrop(zoneId, inputId, type) {
  const zone = document.getElementById(zoneId);

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();                      // MUST prevent default to allow drop
    zone.classList.add("drag-over");         // highlights the border
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");

    // Transfer the dropped file into the hidden <input type="file">
    // DataTransfer lets us do this programmatically
    const dt    = new DataTransfer();
    dt.items.add(e.dataTransfer.files[0]);

    const input = document.getElementById(inputId);
    input.files = dt.files;

    onFileSelected(type);  // same logic as clicking and choosing a file
  });
}


/* ============================================================
   9. ANALYZE IMAGE
   Builds a FormData with the file and POSTs to /detect/image.
   FormData is how browsers send files in HTTP requests.
   ============================================================ */
async function analyzeImage() {
  if (!selectedImageFile) return;

  setLoading("image", true);   // show progress bar
  clearResults("image");       // hide old results

  try {
    // FormData = like a form submission with a file attached
    const formData = new FormData();
    formData.append("file", selectedImageFile);

    const res = await fetch(`${API_BASE}/detect/image`, {
      method: "POST",
      body:   formData
      // NOTE: do NOT set Content-Type header manually!
      // The browser sets it automatically with the correct boundary string
    });

    // If server returned an error (4xx / 5xx), read the message
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    renderImageResults(data);   // build and show result cards

  } catch (e) {
    showError("image", e.message);
  } finally {
    setLoading("image", false);  // always hide progress bar
  }
}


/* ============================================================
   10. ANALYZE VIDEO
   Same pattern as analyzeImage but hits /detect/video.
   Videos are large so this may take longer.
   ============================================================ */
async function analyzeVideo() {
  if (!selectedVideoFile) return;

  setLoading("video", true);
  clearResults("video");

  try {
    const formData = new FormData();
    formData.append("file", selectedVideoFile);

    // sample_every_n_frames=15 tells the backend to skip frames for speed
    const res = await fetch(`${API_BASE}/detect/video?sample_every_n_frames=15`, {
      method: "POST",
      body:   formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    renderVideoResults(data);

  } catch (e) {
    showError("video", e.message);
  } finally {
    setLoading("video", false);
  }
}


/* ============================================================
   11. RENDER IMAGE RESULTS
   Takes the API response object and builds HTML to display it.
   Uses template literals (backtick strings) to embed values.
   ============================================================ */
function renderImageResults(data) {
  const isEmergency = data.emergency_vehicle_count > 0;
  const density     = data.density;
  const pred        = data.prediction;

  const html = `
    <div class="results-header">
      <div class="results-title">DETECTION RESULT</div>
      <div class="results-summary ${isEmergency ? "emergency" : "clear"}">
        ${data.summary}
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat-card ${isEmergency ? "" : "green"}">
        <div class="stat-label">VEHICLES DETECTED</div>
        <div class="stat-value">${data.emergency_vehicle_count}</div>
        <div class="stat-unit">emergency vehicle(s)</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">CONFIDENCE</div>
        <div class="stat-value">${Math.round(pred.confidence * 100)}<span style="font-size:18px">%</span></div>
        <div class="stat-unit">model confidence</div>
      </div>
      <div class="stat-card orange">
        <div class="stat-label">DENSITY</div>
        <div class="stat-value">${density.vehicles_per_million_pixels}</div>
        <div class="stat-unit">per million pixels</div>
        <div><span class="density-badge density-${density.density_level}">${density.density_level}</span></div>
      </div>
    </div>

    <div class="details-grid">
      <div class="detail-block">
        <div class="detail-block-title">// IMAGE INFO</div>
        <div class="detail-row">
          <span class="detail-key">FILENAME</span>
          <span class="detail-val">${data.filename}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">RESOLUTION</span>
          <span class="detail-val">${data.image_size.width} × ${data.image_size.height}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">AREA</span>
          <span class="detail-val">${(data.image_size.width * data.image_size.height).toLocaleString()} px²</span>
        </div>
      </div>
      <div class="detail-block">
        <div class="detail-block-title">// PREDICTION INFO</div>
        <div class="detail-row">
          <span class="detail-key">LABEL</span>
          <span class="detail-val">${pred.label}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">EMERGENCY?</span>
          <span class="detail-val" style="color:${pred.is_emergency_vehicle ? "var(--accent)" : "var(--green)"}">
            ${pred.is_emergency_vehicle ? "⚡ YES" : "✓ NO"}
          </span>
        </div>
        <div class="detail-row">
          <span class="detail-key">DENSITY INFO</span>
          <span class="detail-val">${density.explanation}</span>
        </div>
      </div>
    </div>

    <button class="json-toggle" onclick="toggleJson('imageJson')">VIEW RAW JSON</button>
    <div class="json-block" id="imageJson">${JSON.stringify(data, null, 2)}</div>
  `;

  const el = document.getElementById("imageResults");
  el.innerHTML = html;
  el.classList.add("visible");
}


/* ============================================================
   12. RENDER VIDEO RESULTS
   Same idea as renderImageResults but includes:
   - video info (fps, resolution, frames analyzed)
   - density per second
   - timeline showing WHEN vehicles appeared
   ============================================================ */
function renderVideoResults(data) {
  const isEmergency = data.emergency_vehicle_count > 0;
  const density     = data.density;
  const info        = data.video_info;
  const timeline    = data.timeline || [];

  // Build timeline bar chart HTML
  // Each bar width = (confidence / max_confidence) * 100%
  const maxConf = Math.max(...timeline.map(t => t.confidence), 0.01);

  const timelineHtml = timeline.length === 0
    ? `<div class="timeline-empty">No emergency vehicles detected in this video.</div>`
    : timeline.map(t => `
        <div class="timeline-item">
          <div class="timeline-time">${t.at_second}s</div>
          <div class="timeline-bar-wrap">
            <div class="timeline-bar" style="width:${Math.round((t.confidence / maxConf) * 100)}%"></div>
          </div>
          <div class="timeline-conf">${Math.round(t.confidence * 100)}%</div>
        </div>
      `).join("");

  const html = `
    <div class="results-header">
      <div class="results-title">VIDEO ANALYSIS RESULT</div>
      <div class="results-summary ${isEmergency ? "emergency" : "clear"}">
        ${data.summary}
      </div>
    </div>

    <div class="stat-grid">
      <div class="stat-card ${isEmergency ? "" : "green"}">
        <div class="stat-label">TOTAL DETECTIONS</div>
        <div class="stat-value">${data.emergency_vehicle_count}</div>
        <div class="stat-unit">across all frames</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">DENSITY / SECOND</div>
        <div class="stat-value">${density.vehicles_per_second}</div>
        <div class="stat-unit">vehicles per second</div>
        <div><span class="density-badge density-${density.density_level}">${density.density_level}</span></div>
      </div>
      <div class="stat-card orange">
        <div class="stat-label">PEAK IN FRAME</div>
        <div class="stat-value">${density.peak_count_in_single_frame}</div>
        <div class="stat-unit">highest single-frame count</div>
      </div>
    </div>

    <div class="details-grid">
      <div class="detail-block">
        <div class="detail-block-title">// VIDEO INFO</div>
        <div class="detail-row">
          <span class="detail-key">DURATION</span>
          <span class="detail-val">${info.duration_seconds}s</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">FPS</span>
          <span class="detail-val">${info.fps}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">RESOLUTION</span>
          <span class="detail-val">${info.resolution}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">TOTAL FRAMES</span>
          <span class="detail-val">${info.total_frames.toLocaleString()}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">FRAMES ANALYZED</span>
          <span class="detail-val">${info.frames_analyzed} (${info.sampling})</span>
        </div>
      </div>
      <div class="detail-block">
        <div class="detail-block-title">// DENSITY STATS</div>
        <div class="detail-row">
          <span class="detail-key">VEHICLES / SECOND</span>
          <span class="detail-val">${density.vehicles_per_second}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">AVG PER FRAME</span>
          <span class="detail-val">${density.average_vehicles_per_frame}</span>
        </div>
        <div class="detail-row">
          <span class="detail-key">DENSITY LEVEL</span>
          <span class="detail-val"><span class="density-badge density-${density.density_level}">${density.density_level}</span></span>
        </div>
        <div class="detail-row">
          <span class="detail-key">EXPLANATION</span>
          <span class="detail-val">${density.explanation}</span>
        </div>
      </div>
    </div>

    <div class="timeline-wrap">
      <div class="timeline-title">// DETECTION TIMELINE — when vehicles appeared</div>
      ${timelineHtml}
    </div>

    <button class="json-toggle" onclick="toggleJson('videoJson')">VIEW RAW JSON</button>
    <div class="json-block" id="videoJson">${JSON.stringify(data, null, 2)}</div>
  `;

  const el = document.getElementById("videoResults");
  el.innerHTML = html;
  el.classList.add("visible");
}


/* ============================================================
   13. HELPERS — small reusable utility functions
   ============================================================ */

/*
  setLoading(type, on)
  Shows/hides the progress bar and loading text.
  Also disables the button while loading so user can't double-submit.
*/
function setLoading(type, on) {
  document.getElementById(`${type}LoadingBar`).classList.toggle("visible", on);
  document.getElementById(`${type}LoadingText`).classList.toggle("visible", on);
  document.getElementById(`${type}Btn`).disabled = on;
}

/*
  clearResults(type)
  Wipes out old result HTML and hides the error box.
  Called every time user starts a new analysis.
*/
function clearResults(type) {
  const el = document.getElementById(`${type}Results`);
  el.innerHTML = "";
  el.classList.remove("visible");
  document.getElementById(`${type}Error`).classList.remove("visible");
}

/*
  showError(type, msg)
  Displays a red error box with the message.
  Called in the catch block of analyze functions.
*/
function showError(type, msg) {
  const el = document.getElementById(`${type}Error`);
  el.textContent = `⚠ ERROR: ${msg}`;
  el.classList.add("visible");
}

/*
  toggleJson(id)
  Shows/hides the raw JSON block when user clicks "VIEW RAW JSON".
*/
function toggleJson(id) {
  document.getElementById(id).classList.toggle("visible");
}

/*
  formatSize(bytes)
  Converts raw byte count into a human-readable string.
  e.g. 1500000 → "1.4 MB"
*/
function formatSize(bytes) {
  if (bytes < 1024)            return bytes + " B";
  if (bytes < 1024 * 1024)     return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}