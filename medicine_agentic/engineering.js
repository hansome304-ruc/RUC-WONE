(() => {
  "use strict";

  const API = Object.freeze({
    parameters: "/api/runtime-parameters",
    captureContactZ: "/api/runtime-parameters/capture-contact-z",
    currentPose: "/api/current-pose",
    capturePose: "/api/runtime-poses/capture",
    movePose: "/api/runtime-poses/move",
  });

  const $ = (id) => document.getElementById(id);
  const state = { snapshot: null, taskId: "task1" };

  document.addEventListener("DOMContentLoaded", () => {
    const panel = $("engineering-parameters");
    if (!panel) return;
    $("parameter-task").addEventListener("change", (event) => {
      state.taskId = event.target.value;
      render();
    });
    $("parameter-layer").addEventListener("change", render);
    $("parameter-save").addEventListener("click", save);
    $("parameter-capture-z").addEventListener("click", captureCurrentZ);
    $("pose-read").addEventListener("click", readCurrentPose);
    $("pose-save").addEventListener("click", saveCurrentPose);
    $("pose-arm").addEventListener("change", renderPoseList);
    $("parameter-show-all").addEventListener("click", toggleOverview);
    document.querySelectorAll("[data-task-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const taskId = button.dataset.taskId;
        if (!["task1", "task2", "task3"].includes(taskId)) return;
        state.taskId = taskId;
        $("parameter-task").value = taskId;
        render();
      });
    });
    load().catch((error) => setMessage(error.message, true));
  });

  async function request(url, options = {}, timeoutMs = 5000) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        cache: "no-store",
        credentials: "same-origin",
        ...options,
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("参数接口请求超时");
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function load() {
    const payload = await request(API.parameters);
    state.snapshot = payload.runtime_parameters;
    render();
  }

  function render() {
    if (!state.snapshot) return;
    const task = state.snapshot.tasks?.[state.taskId] || {};
    const isTask1 = state.taskId === "task1";
    $("parameter-layer-field").hidden = !isTask1;
    $("parameter-contact-label").textContent = isTask1
      ? "当前层接触高度 / mm"
      : "接触高度 / mm";
    const layer = $("parameter-layer").value;
    const contact = isTask1
      ? task.contact_flange_z_m_by_layer?.[layer]
      : task.contact_flange_z_m;
    setMm("parameter-contact-z", contact);
    setMm("parameter-clearance", task.pre_contact_clearance_m);
    setMm("parameter-lift", task.test_lift_m);
    setMm("parameter-transit-z", task.transit_z_m);
    $("parameter-revision").textContent = `rev ${state.snapshot.revision || 0}`;
    $("parameter-path").textContent = state.snapshot.path || "—";
    renderPoseList();
    renderOverview();
    setMessage("修改后立即生效，不需要重启 8899。", false);
  }

  function setMm(id, meters) {
    const number = Number(meters);
    $(id).value = Number.isFinite(number) ? (number * 1000).toFixed(2) : "";
  }

  function readMm(id, label) {
    const value = Number($(id).value);
    if (!Number.isFinite(value)) throw new Error(`${label}必须是数字`);
    return value / 1000;
  }

  async function save() {
    const button = $("parameter-save");
    button.disabled = true;
    try {
      const taskId = state.taskId;
      const task = state.snapshot.tasks?.[taskId] || {};
      const values = {
        transit_z_m: readMm("parameter-transit-z", "运输高度"),
        pre_contact_clearance_m: readMm("parameter-clearance", "接近距离"),
        test_lift_m: readMm("parameter-lift", "试抬距离"),
      };
      const contactZ = readMm("parameter-contact-z", "接触高度");
      if (taskId === "task1") {
        values.contact_flange_z_m_by_layer = {
          ...(task.contact_flange_z_m_by_layer || {}),
          [$("parameter-layer").value]: contactZ,
        };
      } else {
        values.contact_flange_z_m = contactZ;
      }
      const payload = await request(API.parameters, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: taskId, values }),
      });
      state.snapshot = payload.runtime_parameters;
      render();
      setMessage("已保存并立即生效。", false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function captureCurrentZ() {
    const button = $("parameter-capture-z");
    button.disabled = true;
    try {
      const body = { task_id: state.taskId };
      if (state.taskId === "task1") body.layer = Number($("parameter-layer").value);
      const payload = await request(API.captureContactZ, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      state.snapshot = payload.runtime_parameters;
      render();
      const millimeters = payload.captured.contact_flange_z_m * 1000;
      setMessage(`已记录当前法兰 Z：${millimeters.toFixed(2)} mm。`, false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function readCurrentPose() {
    const button = $("pose-read");
    button.disabled = true;
    try {
      const arm = $("pose-arm").value;
      const payload = await request(`${API.currentPose}?arm=${encodeURIComponent(arm)}`);
      renderCurrentPose(payload.pose);
      setMessage(`已只读获取${arm === "left" ? "左" : "右"}臂 Pose。`, false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function saveCurrentPose() {
    const button = $("pose-save");
    const name = String($("pose-name").value || "").trim();
    if (!/^[a-z][a-z0-9_]{0,47}$/.test(name)) {
      setMessage("Pose 名称请使用小写字母、数字和下划线。", true);
      return;
    }
    button.disabled = true;
    try {
      const arm = $("pose-arm").value;
      const payload = await request(API.capturePose, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ arm, name }),
      });
      state.snapshot = payload.runtime_parameters;
      renderCurrentPose(payload.pose);
      render();
      setMessage(`已保存 ${arm}.${name}，未移动机械臂。`, false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function renderCurrentPose(pose) {
    if (!pose) return;
    const p = (pose.position_m || []).map((value) => Number(value).toFixed(4));
    const q = (pose.quaternion_xyzw || []).map((value) => Number(value).toFixed(5));
    const j = (pose.joint_positions_rad || []).map((value) => Number(value).toFixed(4));
    $("pose-current").textContent = `P [${p.join(", ")}]  Q [${q.join(", ")}]  J [${j.join(", ")}]`;
  }

  function renderPoseList() {
    if (!state.snapshot) return;
    const arm = $("pose-arm").value;
    const poses = state.snapshot.poses?.[arm] || {};
    const names = Object.keys(poses).sort();
    $("pose-saved").textContent = names.length
      ? `已保存：${names.join(" · ")}`
      : "尚未保存具名 Pose";
    const targets = $("pose-targets");
    targets.innerHTML = `
      <button class="pose-target pose-target--home" type="button" data-pose-home="${arm}">
        <strong>初始位姿</strong><small>预设关节位姿</small>
      </button>
      ${names.map((name) => `
        <button class="pose-target" type="button" data-pose-name="${escapeHtml(name)}">
          <strong>${escapeHtml(name)}</strong><small>已示教 Pose</small>
        </button>`).join("")}`;
    targets.querySelector("[data-pose-home]")?.addEventListener("click", moveHome);
    targets.querySelectorAll("[data-pose-name]").forEach((button) => {
      button.addEventListener("click", () => moveSavedPose(button.dataset.poseName));
    });
  }

  async function moveHome(event) {
    const arm = event.currentTarget.dataset.poseHome;
    const endpoint = arm === "left"
      ? "/api/skills/left-arm/reset-home"
      : "/api/skills/right-arm/reset-home";
    await runPoseMotion(
      event.currentTarget,
      endpoint,
      {},
      `${arm === "left" ? "左" : "右"}臂已到初始位姿。`
    );
  }

  async function moveSavedPose(name) {
    const button = document.querySelector(`[data-pose-name="${CSS.escape(name)}"]`);
    const arm = $("pose-arm").value;
    await runPoseMotion(button, API.movePose, { arm, name }, `已到达 ${arm}.${name}。`);
  }

  async function runPoseMotion(button, endpoint, body, successMessage) {
    if (!button) return;
    document.querySelectorAll(".pose-target").forEach((item) => { item.disabled = true; });
    try {
      await request(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }, 60000);
      setMessage(successMessage, false);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      document.querySelectorAll(".pose-target").forEach((item) => { item.disabled = false; });
    }
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function toggleOverview() {
    const overview = $("parameter-overview");
    overview.hidden = !overview.hidden;
    $("parameter-show-all").textContent = overview.hidden
      ? "查看全部参数"
      : "收起全部参数";
    if (!overview.hidden) renderOverview();
  }

  function renderOverview() {
    if (!state.snapshot) return;
    const tasks = state.snapshot.tasks || {};
    const rows = ["task1", "task2", "task3"].map((taskId) => {
      const task = tasks[taskId] || {};
      const contact = taskId === "task1"
        ? ["1", "2", "3"].map((layer) =>
            `${layer}层 ${formatMm(task.contact_flange_z_m_by_layer?.[layer])}`
          ).join(" / ")
        : formatMm(task.contact_flange_z_m);
      return `
        <tr>
          <th>${taskId.toUpperCase()}</th>
          <td>${contact}</td>
          <td>${formatMm(task.pre_contact_clearance_m)}</td>
          <td>${formatMm(task.test_lift_m)}</td>
          <td>${formatMm(task.transit_z_m)}</td>
        </tr>`;
    });
    $("parameter-overview-body").innerHTML = rows.join("");
    for (const arm of ["left", "right"]) {
      const names = Object.keys(state.snapshot.poses?.[arm] || {}).sort();
      $(`pose-overview-${arm}`).textContent = names.length
        ? names.join(" · ")
        : "尚未保存";
    }
  }

  function formatMm(meters) {
    const value = Number(meters);
    return Number.isFinite(value) ? `${(value * 1000).toFixed(2)} mm` : "—";
  }

  function setMessage(message, isError) {
    const element = $("parameter-message");
    element.textContent = message;
    element.classList.toggle("is-error", isError);
  }
})();
