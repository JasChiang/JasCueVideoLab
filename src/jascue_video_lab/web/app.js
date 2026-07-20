const $ = (selector) => document.querySelector(selector);

const state = {
  session: null,
  review: null,
  canvasImage: null,
  correction: null,
  drawing: false,
  drawEnabled: false,
  startPoint: null,
};

function busy(title, copy = "請勿關閉此頁。") {
  $("#busy-title").textContent = title;
  $("#busy-copy").textContent = copy;
  $("#busy").classList.remove("hidden");
}

function idle() { $("#busy").classList.add("hidden"); }

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.add("hidden"), 6000);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function jsonPost(body) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

function ms(value) { return `${(value / 1000).toFixed(3)} s`; }

function fpsText(rate) {
  if (!rate) return "—";
  return `${(rate.numerator / rate.denominator).toFixed(3)} fps`;
}

function unlock(selector) { $(selector).classList.remove("locked"); }
function updateStage(stage) { $("#stage-badge").textContent = stage.replaceAll("_", " ").toUpperCase(); }

function setSessionUrl(sessionId) {
  const url = new URL(location.href);
  url.searchParams.set("session", sessionId);
  history.replaceState({}, "", url);
}

function renderBase(session) {
  state.session = session;
  $("#upload-section").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  $("#session-id").textContent = session.session_id;
  $("#stage-badge").textContent = session.stage.replaceAll("_", " ").toUpperCase();
  $("#export-link").href = `/api/sessions/${session.session_id}/export`;
  $("#source-video").src = session.video_url;
  $("#asset-name").textContent = session.original_filename;
  $("#media-duration").textContent = ms(session.media.duration_ms);
  $("#media-dimensions").textContent = `${session.media.video.display_width} × ${session.media.video.display_height}`;
  $("#media-fps").textContent = fpsText(session.media.video.average_frame_rate);
  $("#media-hash").textContent = session.media.sha256;
  setSessionUrl(session.session_id);
  if (session.candidate_map) renderCandidates(session.candidate_map, session.file_api_cache);
  if (session.selection) renderSelection(session.selection);
  if (session.moment_map) renderMoments(session.moment_map);
  const pending = (session.review_states || []).find((item) => item.status === "pending_human_review");
  if (pending) renderRestoredReview(pending);
  else {
    const reviewed = [...(session.review_states || [])].reverse().find((item) => item.status === "reviewed");
    if (reviewed) api(`/api/sessions/${session.session_id}/reviews/${reviewed.review_id}/reveal`).then(renderReveal).catch((error) => toast(error.message));
  }
}

function renderCandidates(candidateMap, cache = null) {
  const grid = $("#candidate-grid");
  grid.innerHTML = "";
  const meta = $("#candidate-meta");
  meta.classList.remove("hidden");
  meta.textContent = cache?.reused
    ? "已重用 48 小時保存期內的 Gemini File API 物件；本次沒有重複上傳。"
    : "本次建立了新的 Gemini File API 物件；後續分析會在保存期內重用。";
  candidateMap.candidates.forEach((candidate) => {
    const card = document.createElement("article");
    card.className = "candidate";
    card.innerHTML = `
      <header><h3></h3><span class="kind"></span></header>
      <p class="description"></p><p class="features"></p>
      <time></time><button class="secondary-button">選這個 target</button>`;
    card.querySelector("h3").textContent = candidate.label;
    card.querySelector(".kind").textContent = candidate.entity_kind;
    card.querySelector(".description").textContent = candidate.target_description;
    card.querySelector(".features").textContent = `辨識特徵：${candidate.distinguishing_features}`;
    card.querySelector("time").textContent = `代表時刻 ${candidate.representative_timestamp_mmss}`;
    card.querySelector("button").addEventListener("click", () => selectCandidate(candidate.candidate_id));
    grid.appendChild(card);
  });
}

function renderSelection(selection) {
  const node = $("#selection-summary");
  node.classList.remove("hidden");
  node.innerHTML = "";
  const title = document.createElement("strong");
  title.textContent = `已由使用者鎖定：${selection.target_id}`;
  const description = document.createElement("span");
  description.textContent = selection.target_description;
  node.append(title, description);
  updateStage("target_selected");
  unlock("#moment-step");
}

function renderMoments(momentMap) {
  unlock("#moment-step");
  const grid = $("#moment-grid");
  grid.innerHTML = "";
  momentMap.moments.forEach((moment) => {
    const card = document.createElement("article");
    card.className = "moment";
    card.innerHTML = `<header><h3></h3><time></time></header><p></p><button class="secondary-button">抽原始幀並 Ground</button>`;
    card.querySelector("h3").textContent = moment.label;
    card.querySelector("time").textContent = moment.timestamp_mmss;
    card.querySelector("p").textContent = moment.observable_evidence;
    card.querySelector("time").addEventListener("click", () => {
      const [minutes, seconds] = moment.timestamp_mmss.split(":").map(Number);
      $("#source-video").currentTime = minutes * 60 + seconds;
      $("#source-video").play();
    });
    card.querySelector("button").addEventListener("click", () => groundMoment(moment.moment_id));
    grid.appendChild(card);
  });
  unlock("#review-step");
  updateStage("moments_ready");
}

async function uploadVideo(file) {
  busy("正在建立本機 session", "先驗證影片、媒體資訊與 SHA-256；尚未呼叫 Gemini。");
  const form = new FormData();
  form.append("file", file);
  form.append("use_analysis_proxy", $("#use-proxy").checked ? "true" : "false");
  try {
    const session = await api("/api/sessions", { method: "POST", body: form });
    renderBase(session);
  } catch (error) { toast(error.message); } finally { idle(); }
}

async function suggestCandidates() {
  busy("Gemini 正在提出候選", "只建立可選 target，不產生 bbox 或 tracking data。");
  try {
    const result = await api(`/api/sessions/${state.session.session_id}/candidates`, jsonPost({ runs: 1 }));
    renderCandidates(result.runs[0], { reused: result.file_api_object_reused });
    state.session = await api(`/api/sessions/${state.session.session_id}`);
  } catch (error) { toast(error.message); } finally { idle(); }
}

async function selectCandidate(candidateId) {
  try {
    const selection = await api(`/api/sessions/${state.session.session_id}/target`, jsonPost({ candidate_id: candidateId }));
    renderSelection(selection);
  } catch (error) { toast(error.message); }
}

async function selectManual(event) {
  event.preventDefault();
  const targetId = $("#manual-target-id").value.trim();
  const targetDescription = $("#manual-target-description").value.trim();
  if (!targetId || !targetDescription) return toast("請同時填寫 Target ID 與精確描述。");
  try {
    const selection = await api(`/api/sessions/${state.session.session_id}/target`, jsonPost({ target_id: targetId, target_description: targetDescription }));
    renderSelection(selection);
  } catch (error) { toast(error.message); }
}

async function analyzeMoments() {
  busy("Gemini 正在尋找代表時刻", "使用已鎖定的 target；MM:SS 仍只是 coarse semantic time。");
  try {
    const result = await api(`/api/sessions/${state.session.session_id}/moments`, jsonPost({ runs: 1 }));
    renderMoments(result.runs[0]);
  } catch (error) { toast(error.message); } finally { idle(); }
}

async function groundMoment(momentId) {
  busy("正在抽原始影格並 Grounding", "FFmpeg 會保存 exact frame PTS；Gemini 接收的是這張單幀影像。");
  try {
    const review = await api(`/api/sessions/${state.session.session_id}/ground`, jsonPost({ moment_id: momentId }));
    renderReview(review);
  } catch (error) { toast(error.message); } finally { idle(); }
}

function renderRestoredReview(manifest) {
  renderReview({
    review_id: manifest.review_id,
    target_description: manifest.target_description,
    requested_timestamp_mmss: manifest.requested_timestamp_mmss,
    requested_time_ms: manifest.requested_time_ms,
    frame_pts: manifest.frame_pts,
    frame_time_ms: manifest.frame_time_ms,
    frame_hash: manifest.frame_hash,
    blind_image_url: `/api/sessions/${state.session.session_id}/reviews/${manifest.review_id}/blind-image`,
  });
}

function renderReview(review) {
  state.review = review;
  state.correction = null;
  $("#review-empty").classList.add("hidden");
  $("#review-card").classList.remove("hidden");
  $("#reveal-card").classList.add("hidden");
  $("#review-target").textContent = review.target_description;
  $("#review-request-time").textContent = `${review.requested_timestamp_mmss} (${review.requested_time_ms} ms)`;
  $("#review-frame-time").textContent = `${review.frame_time_ms} ms · PTS ${review.frame_pts}`;
  updateStage("blind_review_pending");
  loadCanvas(review.blind_image_url);
  $("#review-step").scrollIntoView({ behavior: "smooth", block: "start" });
}

function loadCanvas(url) {
  const image = new Image();
  image.onload = () => {
    state.canvasImage = image;
    const canvas = $("#review-canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    redrawCanvas();
  };
  image.src = `${url}?v=${Date.now()}`;
}

function redrawCanvas(preview = null) {
  const canvas = $("#review-canvas");
  const context = canvas.getContext("2d");
  if (!state.canvasImage) return;
  context.drawImage(state.canvasImage, 0, 0, canvas.width, canvas.height);
  const box = preview || state.correction;
  if (box) {
    context.strokeStyle = "#65baff";
    context.lineWidth = Math.max(3, Math.round(Math.min(canvas.width, canvas.height) / 220));
    context.setLineDash([12, 8]);
    context.strokeRect(box.x, box.y, box.w, box.h);
    context.setLineDash([]);
  }
}

function canvasPoint(event) {
  const canvas = $("#review-canvas");
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * canvas.width / rect.width;
  const y = (event.clientY - rect.top) * canvas.height / rect.height;
  return { x: Math.max(0, Math.min(canvas.width, x)), y: Math.max(0, Math.min(canvas.height, y)) };
}

function normalizedCorrection() {
  if (!state.correction) return null;
  const canvas = $("#review-canvas");
  const x1 = Math.min(state.correction.x, state.correction.x + state.correction.w);
  const x2 = Math.max(state.correction.x, state.correction.x + state.correction.w);
  const y1 = Math.min(state.correction.y, state.correction.y + state.correction.h);
  const y2 = Math.max(state.correction.y, state.correction.y + state.correction.h);
  return [x1, y1, x2, y2].map((value, index) => Math.round(value * 1000 / (index % 2 === 0 ? canvas.width : canvas.height)));
}

function updateCorrectionValue() {
  const value = normalizedCorrection();
  $("#correction-value").textContent = value ? `[${value.join(", ")}]` : "尚未提供人工框";
}

async function submitReview(event) {
  event.preventDefault();
  const verdict = new FormData(event.currentTarget).get("verdict");
  if (!verdict) return toast("請先選擇你的判定。");
  busy("正在鎖定人工判定", "寫入成功後才會揭露模型 confidence 與理由。");
  try {
    const reveal = await api(`/api/sessions/${state.session.session_id}/reviews/${state.review.review_id}`, jsonPost({
      verdict,
      notes: $("#review-notes").value,
      reviewer_name: $("#reviewer-name").value,
      corrected_box_2d: normalizedCorrection(),
    }));
    renderReveal(reveal);
  } catch (error) { toast(error.message); } finally { idle(); }
}

function renderReveal(reveal) {
  $("#review-card").classList.add("hidden");
  const card = $("#reveal-card");
  card.classList.remove("hidden");
  $("#revealed-image").src = reveal.revealed_image_url;
  $("#reveal-json").textContent = JSON.stringify({ human_annotation: reveal.annotation, gemini_proposal: reveal.proposal }, null, 2);
  updateStage("reviewed");
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

const dropZone = $("#drop-zone");
dropZone.addEventListener("click", () => $("#video-file").click());
dropZone.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) $("#video-file").click(); });
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
dropZone.addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) uploadVideo(file); });
$("#video-file").addEventListener("change", (event) => { const file = event.target.files[0]; if (file) uploadVideo(file); });
$("#suggest-button").addEventListener("click", suggestCandidates);
$("#manual-target-form").addEventListener("submit", selectManual);
$("#moments-button").addEventListener("click", analyzeMoments);
$("#review-form").addEventListener("submit", submitReview);
$("#draw-correction").addEventListener("click", () => { state.drawEnabled = true; toast("請在影像上拖曳出人工修正框。"); });
$("#clear-correction").addEventListener("click", () => { state.correction = null; redrawCanvas(); updateCorrectionValue(); });

const canvas = $("#review-canvas");
canvas.addEventListener("pointerdown", (event) => {
  if (!state.drawEnabled) return;
  state.drawing = true;
  state.startPoint = canvasPoint(event);
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.drawing) return;
  const point = canvasPoint(event);
  redrawCanvas({ x: state.startPoint.x, y: state.startPoint.y, w: point.x - state.startPoint.x, h: point.y - state.startPoint.y });
});
canvas.addEventListener("pointerup", (event) => {
  if (!state.drawing) return;
  const point = canvasPoint(event);
  const candidate = { x: state.startPoint.x, y: state.startPoint.y, w: point.x - state.startPoint.x, h: point.y - state.startPoint.y };
  state.correction = Math.abs(candidate.w) >= 3 && Math.abs(candidate.h) >= 3 ? candidate : null;
  state.drawing = false;
  state.drawEnabled = false;
  redrawCanvas();
  updateCorrectionValue();
});

(async function restore() {
  const sessionId = new URL(location.href).searchParams.get("session");
  if (!sessionId) return;
  busy("正在還原本機 session");
  try { renderBase(await api(`/api/sessions/${sessionId}`)); }
  catch (error) { toast(error.message); }
  finally { idle(); }
})();
