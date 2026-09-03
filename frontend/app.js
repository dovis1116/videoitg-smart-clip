const $ = (id) => document.getElementById(id);
const labels = {
  ACCEPT: "采用",
  IRRELEVANT: "不相关",
  START_TOO_EARLY: "开始太早",
  START_TOO_LATE: "开始太晚",
  END_TOO_EARLY: "结束太早",
  END_TOO_LATE: "结束太晚",
  MISS: "视频中无目标",
  DUPLICATE: "结果重复",
};

let currentTask = null;
let pollTimer = null;
let localPreviewUrl = null;
let selectedCandidateId = null;

function api(path) {
  return `${$("api-base").value.replace(/\/$/, "")}${path}`;
}

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

async function request(path, options = {}) {
  const response = await fetch(api(path), options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail?.message || (typeof payload.detail === "string" ? payload.detail : `HTTP ${response.status}`));
    error.status = response.status;
    throw error;
  }
  return payload;
}

function renderPredictions(result) {
  const predictions = result?.predictions || [];
  const root = $("predictions");
  const timeline = $("timeline");
  root.innerHTML = predictions.length ? "" : `<p class='muted'>${result?.status === "NO_MATCH" ? "未检测到匹配事件。" : "没有返回候选片段。"}</p>`;
  timeline.innerHTML = "";
  const startOf = (item) => Number(item.refined_start ?? item.start_s ?? item.raw_start ?? item.coarse_start ?? 0);
  const endOf = (item) => Number(item.refined_end ?? item.end_s ?? item.raw_end ?? item.coarse_end ?? 0);
  const scoreOf = (item) => Number(item.final_score ?? item.score ?? item.grounding_score ?? 0);
  const duration = Number($("preview").duration) || Math.max(...predictions.map(endOf), 5);
  predictions.forEach((item, index) => {
    const start = startOf(item);
    const end = endOf(item);
    const timelineBar = document.createElement("button");
    timelineBar.className = "timeline-bar";
    timelineBar.type = "button";
    timelineBar.style.left = `${Math.max(0, start / duration) * 100}%`;
    timelineBar.style.width = `${Math.max(3, (end - start) / duration * 100)}%`;
    timelineBar.title = `${start.toFixed(2)}s – ${end.toFixed(2)}s`;
    timelineBar.textContent = `${index + 1}`;
    timelineBar.onclick = () => selectPrediction(item);
    timeline.appendChild(timelineBar);
    const card = document.createElement("div");
    card.className = "prediction";
    card.innerHTML = `<span><strong>#${index + 1}</strong> ${start.toFixed(2)}s – ${end.toFixed(2)}s<br><small>score ${scoreOf(item).toFixed(4)}</small></span>`;
    const choose = document.createElement("button");
    choose.className = "secondary";
    choose.textContent = "用于反馈";
    choose.onclick = () => selectPrediction(item);
    card.appendChild(choose);
    root.appendChild(card);
  });
}

function selectPrediction(item) {
  selectedCandidateId = item.candidate_id || null;
  const start = Number(item.refined_start ?? item.start_s ?? item.raw_start ?? item.coarse_start ?? 0);
  const end = Number(item.refined_end ?? item.end_s ?? item.raw_end ?? item.coarse_end ?? 0);
  $("adjusted-start").value = start.toFixed(2);
  $("adjusted-end").value = end.toFixed(2);
  if ($( "preview").duration) $("preview").currentTime = start;
}

function renderFeedbackButtons() {
  const root = $("feedback-buttons");
  root.innerHTML = "";
  Object.entries(labels).forEach(([value, text]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = text;
    button.onclick = () => submitFeedback(value);
    root.appendChild(button);
  });
}

async function submitFeedback(label) {
  if (!currentTask) return;
  const start = $("adjusted-start").value;
  const end = $("adjusted-end").value;
  try {
    const payload = { task_id: currentTask.task_id, candidate_id: selectedCandidateId, label, comment: $("comment").value || null };
    if (start !== "") payload.adjusted_start_s = Number(start);
    if (end !== "") payload.adjusted_end_s = Number(end);
    const saved = await request("/v1/feedback", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
    $("feedback-message").textContent = `反馈已保存（${saved.feedback_version}）。`;
  } catch (error) {
    $("feedback-message").textContent = `反馈失败：${error.message}`;
    $("feedback-message").classList.add("error");
  }
}

function renderTask(task) {
  currentTask = task;
  $("result-panel").classList.remove("hidden");
  const stage = task.current_stage || task.canonical_status || task.status;
  $("task-meta").textContent = `task_id=${task.task_id} · ${stage} · ${Math.round(Number(task.progress || 0) * 100)}% · model=${task.model_version} · policy=${task.policy_version}`;
  $("task-progress").value = Number(task.progress || 0);
  if (task.result) renderPredictions(task.result);
  $("cancel").disabled = ["succeeded", "failed", "cancelled", "timeout"].includes(task.status);
}

async function pollTask() {
  if (!currentTask) return;
  try {
    const task = await request(`/v1/tasks/${currentTask.task_id}`);
    renderTask(task);
    if (["queued", "running"].includes(task.status)) {
      pollTimer = window.setTimeout(pollTask, 800);
    } else if (task.status === "timeout") {
      // The watchdog exposes TIMEOUT at the budget, while the backend may
      // still be finishing GPU work.  Keep polling the read-only results
      // endpoint so a late Level-2 coarse fallback becomes visible instead
      // of leaving the user with an empty terminal card.
      try {
        const results = await request(`/v1/tasks/${currentTask.task_id}/results`);
        renderTask({ ...task, result: results.result, degraded: results.degraded });
        message("任务超时，已显示可用的降级候选。", true);
      } catch (error) {
        if (error.status === 409) {
          pollTimer = window.setTimeout(pollTask, 1000);
          return;
        }
        message(`超时结果查询失败：${error.message}`, true);
      }
    } else {
      message(`任务${task.status === "succeeded" ? "完成" : "结束"}。`);
    }
  } catch (error) {
    message(`查询失败：${error.message}`, true);
  }
}

async function submitTask() {
  const file = $("video-file").files[0];
  const query = $("query").value.trim();
  if (!file || !query) return message("请先选择视频并填写查询。", true);
  $("submit").disabled = true;
  message("正在提交异步任务……");
  if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
  localPreviewUrl = URL.createObjectURL(file);
  $("preview").src = localPreviewUrl;
  try {
    const data = new FormData();
    data.append("query", query);
    data.append("request_id", `web-${crypto.randomUUID()}`);
    data.append("file", file);
    const task = await request("/v1/tasks/upload", { method: "POST", body: data });
    renderTask(task);
    message(`任务已接收：${task.task_id}`);
    window.clearTimeout(pollTimer);
    pollTimer = window.setTimeout(pollTask, 200);
  } catch (error) {
    message(`提交失败：${error.message}`, true);
  } finally {
    $("submit").disabled = false;
  }
}

async function cancelTask() {
  if (!currentTask) return;
  try {
    renderTask(await request(`/v1/tasks/${currentTask.task_id}`, { method: "DELETE" }));
    message("任务已取消或正在取消。", false);
  } catch (error) {
    message(`取消失败：${error.message}`, true);
  }
}

async function checkHealth() {
  try {
    const health = await request("/healthz");
    message(`服务正常：${health.backend} · ${health.model_version} · queue ${health.queue_depth}/${health.queue_capacity}`);
  } catch (error) {
    message(`服务不可用：${error.message}`, true);
  }
}

$("submit").onclick = submitTask;
$("cancel").onclick = cancelTask;
$("health").onclick = checkHealth;
$("preview").onloadedmetadata = () => {
  if (currentTask?.result) renderPredictions(currentTask.result);
};
renderFeedbackButtons();
