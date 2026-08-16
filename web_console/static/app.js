const state = {
  selectedJobId: null,
  config: null,
  status: null,
  activeView: "generalist-view",
  cameras: null,
  cameraRenderToken: 0,
  idealNav: {
    pose: { x: 3.0, y: 0.45, yaw: Math.PI / 2 },
    planningStart: { x: 3.0, y: 0.45 },
    target: { x: 5.35, y: 3.05 },
    runs: [],
    loadedRun: null,
    replayTimer: null,
    liveTimer: null,
    replayPoint: null,
    pendingRunId: null,
    pendingRouteRunIds: [],
    activeRouteIndex: 0,
    route: [],
    routeActive: false,
    finalYaw: Math.PI / 2,
    routeMode: "normal",
    avoidObstacles: false,
    isRunning: false,
    mapLayer: null,
    mapPlan: null,
    mapPlanBusy: false,
    routeJobId: null,
    executionGoal: null,
    executionWaypoints: [],
    encounterPoints: [],
    recoveryBusy: false,
    recoveryCount: 0,
    lastObstacleSampleKey: "",
    absoluteState: null,
    localization: null,
    executionMode: "route",
    absoluteTrimActive: false,
    latestCorrection: null,
    competitionWaypoints: [],
    competitionWaypointRevision: 0,
    selectedCompetitionWaypointId: null,
    competitionWaypointBusy: false,
    competitionWaypointExecution: null,
    scene3d: null,
    selectedScene3dFixtureId: "shelf_a",
    selectedScene3dPickPoint: null,
    scene3dBusy: false,
    pregraspPoses: { revision: 0, records: [] },
    pregraspCaptureBusy: false,
    mapMode: "2d",
  },
  graspFlow: {
    phase: "idle",
    payload: null,
    validateJobId: null,
    dryRunJobId: null,
    realJobId: null,
    autoStartingDryRun: false,
    readyForReality: false,
    lastNoticeKey: "",
  },
  task1: {
    activeTab: "task",
    config: null,
    configs: [],
    selectedConfig: "",
    run: null,
    active: false,
    configValidation: null,
    deviceValidation: null,
    validationToken: "",
    validationExpiresAt: 0,
    dryRunApprovedConfig: "",
    busyAction: "",
    statusLoading: false,
    pollTimer: null,
    pollingStarted: false,
    lastError: "",
    lastStatusAt: 0,
    lastTerminalTaskId: "",
    liveHoldTimer: null,
    liveHoldStarted: false,
    liveRequestStarted: false,
    stopRequestedLocally: false,
  },
  auth: {
    csrfToken: "",
    loginPromise: null,
  },
  estop: {
    latched: null,
    epoch: 0,
    transactionState: "loading",
    stopConfirmed: false,
    requiresPhysicalEstop: false,
    reason: "",
    requestInFlight: false,
  },
  suction: {
    port: null,
    writer: null,
    robotAvailable: false,
    robotDevice: "",
    lastTransport: "",
    engaged: false,
    busy: false,
  },
  graspAssist: {
    enabled: false,
    mainCamera: "front",
    targetX: 0.5,
    targetY: 0.55,
    tipOffsetMm: 0,
    warningDistanceMm: 250,
    dangerDistanceMm: 150,
    pollTimer: null,
    pollBusy: false,
    latestDepth: null,
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const DEFAULT_CAMERAS = [
  { name: "left", label: "Left", source: "", stream_url: "/api/cameras/left/stream.mjpg" },
  { name: "front", label: "Front", source: "", stream_url: "/api/cameras/front/stream.mjpg" },
  { name: "right", label: "Right", source: "", stream_url: "/api/cameras/right/stream.mjpg" },
];
const SUCTION_BAUD_RATE = 115200;
const SUCTION_BRIDGE_URL = "http://127.0.0.1:8795";
const SUCTION_ON_COMMANDS = ["#005P2500T0000!", "#006P2500T0000!"];
const SUCTION_OFF_COMMANDS = ["#255P1500T0000!", "#005P1500T0000!", "#006P1500T0000!"];
const IDEAL_MAP = {
  width: 6,
  height: 4,
  robotSizeX: 1.00,
  robotSizeY: 0.90,
  footprintModel: "rectangle",
  robotHalfX: 0.50,
  robotHalfY: 0.45,
  robotTurnRadius: Math.hypot(0.50, 0.45),
  robotRadius: 0.50,
  armMotion: "independent",
  initial: { x: 3.0, y: 0.45, yawDeg: 90, yaw: Math.PI / 2 },
  taskZone: { x: 3.0, y: 0.45, radiusM: 0.42 },
  fixtures: [
    { type: "station", label: "补货台", x: 0.15, y: 0.05, w: 1.20, h: 0.80 },
    { type: "station", label: "交付台", x: 4.65, y: 0.05, w: 1.20, h: 0.80 },
    { type: "shelf", label: "货架 A", x: 1.50, y: 1.99, w: 0.80, h: 1.86 },
    { type: "shelf", label: "货架 B", x: 3.70, y: 1.99, w: 0.80, h: 1.86 },
  ],
};
const COMPETITION_WAYPOINT_KIND_LABELS = {
  receipt: "取单位",
  shelf: "货架位",
  delivery: "交付位",
  finish: "判定区",
  staging: "等待位",
  custom: "自定义",
};
const COMPETITION_WAYPOINT_KIND_COLORS = {
  receipt: "#147a4c",
  shelf: "#b7791f",
  delivery: "#1769aa",
  finish: "#6d4c8d",
  staging: "#475467",
  custom: "#b93822",
};
const COMPETITION_WAYPOINT_DEFAULT_LIFT_MM = 180;
const COMPETITION_WAYPOINT_MIN_LIFT_MM = 0;
const COMPETITION_WAYPOINT_MAX_LIFT_MM = 300;
const IDEAL_AVOIDANCE = {
  scanTopic: "/camera_scan",
  stopDistance: 0.35,
  frontDeg: 60,
  staleS: 1.0,
};
const IDEAL_RECOVERY = {
  autoReplan: false,
  markRadius: 0.2,
  maxReplans: 3,
};
const IDEAL_ABSOLUTE_TRIM = {
  enabled: true,
  maxSpeed: 0.03,
  maxYawSpeed: 0.08,
  timeout: 30,
  posTol: 0.01,
  yawTol: 0.02,
};
const TASK1_ACTIVE_POLL_MS = 1000;
const TASK1_IDLE_POLL_MS = 3000;
const TASK1_LIVE_HOLD_MS = 1200;
const TASK1_TERMINAL_STATES = new Set(["complete", "completed", "failed", "cancelled", "canceled", "stopped"]);
const TASK1_ACTIVE_PROCESS_STATES = new Set(["queued", "starting", "running", "stopping", "stop_requested"]);
const TASK1_PHASE_GROUPS = [
  { label: "准备", states: ["created", "preflight"] },
  { label: "读取订单", states: ["navigate_to_receipt", "read_receipt", "validate_order"] },
  { label: "前往货架", states: ["navigate_to_shelf"] },
  { label: "识别与抓取", states: ["pick", "verify_held", "pick_recovery", "prepare_carry"] },
  { label: "返回桌面", states: ["navigate_to_delivery"] },
  { label: "放置商品", states: ["place", "verify_placed"] },
  { label: "完成", states: ["navigate_to_finish", "complete", "completed"] },
];
const TASK1_STATE_LABELS = {
  created: "任务已创建",
  preflight: "执行安全预检",
  navigate_to_receipt: "前往小票桌",
  read_receipt: "读取小票",
  validate_order: "核对订单",
  navigate_to_shelf: "前往商品货架",
  pick: "识别并抓取商品",
  verify_held: "确认商品已夹持",
  pick_recovery: "抓取恢复",
  prepare_carry: "切换运输姿态",
  navigate_to_delivery: "返回交付桌",
  place: "放置商品",
  verify_placed: "确认商品已放置",
  navigate_to_finish: "返回结束点",
  complete: "任务完成",
  completed: "任务完成",
  failed: "任务失败",
};
const TASK1_EVENT_LABELS = {
  mission_created: "任务已创建",
  state_transition: "状态切换",
  receipt_attempt: "开始读取小票",
  receipt_rejected: "小票内容未通过校验",
  order_accepted: "订单已确认",
  navigation_attempt: "开始导航",
  navigation_arrived: "已到达目标航点",
  navigation_failed: "导航失败",
  navigation_cancel_failed: "导航停止确认失败",
  navigation_result: "导航结果",
  pick_result: "抓取结果",
  hold_verification: "夹持验证",
  place_result: "放置结果",
  placement_verification: "放置验证",
  job_delivered: "一件商品已交付",
  mission_complete: "全部商品交付完成",
  mission_failed: "任务安全终止",
  stop_requested: "收到停止请求",
};

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => {
    el.hidden = true;
  }, 4200);
}

function renderSuctionState(message = "") {
  const chip = $("#teleop-suction-chip");
  const status = $("#teleop-suction-status");
  const button = $("#teleop-suction-toggle");
  const connected = Boolean(state.suction.robotAvailable || state.suction.writer);
  if (chip) {
    chip.textContent = state.suction.busy
      ? "切换中"
      : state.suction.engaged
        ? "吸紧"
        : connected
          ? "泄劲"
          : "未连接";
    chip.classList.toggle("is-active", state.suction.engaged);
    chip.classList.toggle("is-busy", state.suction.busy);
    chip.classList.toggle("is-disconnected", !connected);
  }
  if (status) {
    status.textContent = message || (
      state.suction.engaged
        ? "吸盘已吸紧，按 F 泄劲"
        : state.suction.robotAvailable
          ? `机器人侧吸盘已连接：${state.suction.robotDevice || "串口设备"}`
          : state.suction.writer
            ? "浏览器串口已连接，按 F 切换吸紧 / 泄劲"
        : "按 F 会优先使用机器人侧 /api/suction；失败时再尝试本机 bridge / CH340 串口"
    );
  }
  if (button) {
    button.disabled = state.suction.busy;
    button.textContent = state.suction.engaged ? "F 泄劲" : "F 吸紧";
  }
}

function assertSuctionSerialAvailable() {
  if (!window.isSecureContext) {
    throw new Error("吸盘本机桥接服务不可用；请启动 suction bridge，或打开本机 http://127.0.0.1:8766 的 Teleop 页面。");
  }
  if (!("serial" in navigator)) {
    throw new Error("当前浏览器不支持 Web Serial；请使用 Chrome 或 Edge。");
  }
}

async function ensureSuctionSerial() {
  assertSuctionSerialAvailable();
  if (state.suction.port?.readable || state.suction.port?.writable) {
    return;
  }
  state.suction.port = await navigator.serial.requestPort({
    filters: [{ usbVendorId: 0x1a86, usbProductId: 0x7523 }],
  });
  await state.suction.port.open({ baudRate: SUCTION_BAUD_RATE, dataBits: 8, stopBits: 1, parity: "none" });
  state.suction.writer = state.suction.port.writable.getWriter();
}

async function writeSuctionCommands(commands) {
  await ensureSuctionSerial();
  const encoder = new TextEncoder();
  for (const command of commands) {
    await state.suction.writer.write(encoder.encode(command));
    await new Promise((resolve) => window.setTimeout(resolve, 80));
  }
}

async function setSuctionViaBridge(engaged) {
  const response = await fetch(`${SUCTION_BRIDGE_URL}/suction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ engaged }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `吸盘本机桥接请求失败：${response.status}`);
  }
  return payload;
}

async function setSuctionViaRobotApi(engaged) {
  const payload = await api("/api/suction", {
    method: "POST",
    body: JSON.stringify({ engaged }),
  });
  state.suction.robotAvailable = true;
  state.suction.robotDevice = payload.resolved_device || payload.device || "";
  state.suction.lastTransport = "robot";
  return payload;
}

async function refreshSuctionStatus() {
  try {
    const payload = await api("/api/suction/status");
    state.suction.robotAvailable = payload.available === true;
    state.suction.robotDevice = payload.resolved_device
      || payload.configured_device
      || "";
    if (typeof payload.engaged === "boolean") {
      state.suction.engaged = payload.engaged;
    }
    renderSuctionState(
      payload.available
        ? ""
        : `机器人侧吸盘不可用：${payload.error || "未找到串口设备"}`,
    );
    return payload;
  } catch (error) {
    state.suction.robotAvailable = false;
    renderSuctionState(`吸盘状态读取失败：${error.message}`);
    throw error;
  }
}

async function setSuctionEngaged(engaged) {
  if (state.suction.busy) return;
  state.suction.busy = true;
  renderSuctionState(engaged ? "正在吸紧..." : "正在泄劲...");
  try {
    try {
      await setSuctionViaRobotApi(engaged);
    } catch (apiError) {
      try {
        await setSuctionViaBridge(engaged);
        state.suction.lastTransport = "bridge";
      } catch (bridgeError) {
        try {
          await writeSuctionCommands(engaged ? SUCTION_ON_COMMANDS : SUCTION_OFF_COMMANDS);
          state.suction.lastTransport = "browser-serial";
        } catch (serialError) {
          throw new Error(
            `机器人侧失败：${apiError.message}；`
            + `本机 bridge 失败：${bridgeError.message}；`
            + `浏览器串口失败：${serialError.message}`,
          );
        }
      }
    }
    state.suction.engaged = engaged;
    renderSuctionState(engaged ? "吸盘已吸紧，按 F 泄劲" : "吸盘已泄劲，按 F 吸紧");
    toast(engaged ? "吸盘已吸紧" : "吸盘已泄劲");
  } catch (error) {
    renderSuctionState(error.message);
    toast(error.message);
  } finally {
    state.suction.busy = false;
    renderSuctionState();
  }
}

function toggleSuction() {
  setSuctionEngaged(!state.suction.engaged).catch((error) => {
    renderSuctionState(error.message);
    toast(error.message);
  });
}

function shouldIgnoreShortcut(event) {
  const target = event.target;
  if (!target) return false;
  const tagName = target.tagName?.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select" || target.isContentEditable;
}

function resetGraspFlow(overrides = {}) {
  state.graspFlow = {
    phase: "idle",
    payload: null,
    validateJobId: null,
    dryRunJobId: null,
    realJobId: null,
    autoStartingDryRun: false,
    readyForReality: false,
    lastNoticeKey: "",
    ...overrides,
  };
}

function requestOperatorToken(errorMessage = "") {
  const dialog = $("#operator-login");
  const form = $("#operator-login-form");
  const input = $("#operator-token");
  const error = $("#operator-login-error");
  const cancelButton = $("#operator-login-cancel");
  if (!dialog || !form || !input || !error || !cancelButton) {
    return Promise.reject(new Error("操作员登录界面未加载"));
  }

  input.value = "";
  error.textContent = errorMessage;
  error.hidden = !errorMessage;

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      form.removeEventListener("submit", handleSubmit);
      dialog.removeEventListener("cancel", handleCancel);
      cancelButton.removeEventListener("click", handleCancel);
    };
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      input.value = "";
      if (dialog.open) dialog.close();
      callback(value);
    };
    const handleSubmit = (event) => {
      event.preventDefault();
      const token = input.value.trim();
      if (!token) {
        error.textContent = "请输入操作员令牌";
        error.hidden = false;
        input.focus();
        return;
      }
      finish(resolve, token);
    };
    const handleCancel = (event) => {
      event.preventDefault();
      finish(reject, new Error("需要操作员认证才能访问机器人控制台"));
    };

    form.addEventListener("submit", handleSubmit);
    dialog.addEventListener("cancel", handleCancel);
    cancelButton.addEventListener("click", handleCancel);
    if (!dialog.open) dialog.showModal();
    window.requestAnimationFrame(() => input.focus());
  });
}

async function ensureAuthenticated() {
  if (state.auth.loginPromise) return state.auth.loginPromise;
  state.auth.loginPromise = (async () => {
    const sessionResponse = await fetch("/api/auth/session", {
      cache: "no-store",
      credentials: "same-origin",
    });
    const session = await sessionResponse.json().catch(() => ({}));
    if (sessionResponse.ok && session.authenticated) {
      state.auth.csrfToken = session.csrf_token || "";
      if (!state.auth.csrfToken) {
        throw new Error("服务器未提供写操作安全令牌");
      }
      return;
    }
    if (sessionResponse.ok && session.authentication_required === false) {
      throw new Error("匿名安全会话初始化失败");
    }
    let loginError = "";
    while (true) {
      const token = await requestOperatorToken(loginError);
      const loginResponse = await fetch("/api/auth/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      const login = await loginResponse.json().catch(() => ({}));
      if (!loginResponse.ok || !login.authenticated) {
        loginError = login.error || "操作员认证失败";
        continue;
      }
      state.auth.csrfToken = login.csrf_token || "";
      if (!state.auth.csrfToken) {
        loginError = "服务器未返回安全会话令牌";
        continue;
      }
      return;
    }
  })();
  try {
    await state.auth.loginPromise;
  } finally {
    state.auth.loginPromise = null;
  }
}

async function api(path, options = {}, allowAuthRetry = true) {
  const target = new URL(path, window.location.href);
  if (target.origin !== window.location.origin || !target.pathname.startsWith("/api/")) {
    throw new Error("Refusing a non-local API request");
  }
  const method = String(options.method || "GET").toUpperCase();
  const safeMethod = ["GET", "HEAD", "OPTIONS"].includes(method);
  if (!safeMethod && !state.auth.csrfToken) {
    await ensureAuthenticated();
  }
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (!safeMethod && state.auth.csrfToken) {
    headers.set("X-CSRF-Token", state.auth.csrfToken);
  }
  const response = await fetch(target.pathname + target.search, {
    ...options,
    method,
    credentials: "same-origin",
    headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401 && (payload.auth_required || payload.session_required) && allowAuthRetry) {
    state.auth.csrfToken = "";
    await ensureAuthenticated();
    if (["GET", "HEAD"].includes(method)) {
      return api(path, options, false);
    }
    throw new Error("认证已恢复，请再次确认并执行本次操作");
  }
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function applyEstopState(payload = {}) {
  const latchKnown = typeof payload.latched === "boolean";
  state.estop.latched = latchKnown ? payload.latched : null;
  state.estop.epoch = Number(payload.epoch || 0);
  state.estop.transactionState = latchKnown
    ? String(payload.transaction_state || (state.estop.latched ? "unconfirmed" : "clear"))
    : "unknown";
  state.estop.stopConfirmed = latchKnown && payload.stop_confirmed === true;
  state.estop.requiresPhysicalEstop = !latchKnown || payload.requires_physical_estop === true;
  state.estop.reason = String(payload.reason || "");
  renderEstopState();
  window.dispatchEvent(new CustomEvent("ruc:estop-state", {
    detail: { ...state.estop },
  }));
}

function renderEstopState() {
  const reset = $("#estop-reset-button");
  const stop = $("#estop-button");
  const teleopNotice = $("#teleop-safety-notice");
  const teleopNoticeMessage = $("#teleop-safety-message");
  if (!reset || !stop) return;
  const showRecoveryControl = state.activeView === "generalist-view"
    && state.estop.latched === true
    && state.estop.transactionState !== "stopping";
  reset.hidden = !showRecoveryControl
    || state.estop.transactionState === "stopping";
  reset.disabled = state.estop.requestInFlight;
  stop.disabled = state.estop.requestInFlight;
  stop.textContent = state.estop.requestInFlight ? "全局急停…" : "全局急停";
  if (teleopNotice) {
    const teleopBlocked = state.activeView === "teleop-view" && state.estop.latched !== false;
    teleopNotice.hidden = !teleopBlocked;
    if (state.estop.latched === null) {
      teleopNoticeMessage.textContent = "全局急停状态读取中，遥操作暂不可用。";
    } else if (state.estop.transactionState === "stopping") {
      teleopNoticeMessage.textContent = "全局急停正在执行，遥操作已锁定。";
    } else {
      teleopNoticeMessage.textContent = "全局急停已锁存。请先确认机器人和现场安全，再前往 Generalist 恢复控制。";
    }
  }
}

async function refreshEstopStatus({ silent = true } = {}) {
  try {
    const payload = await api("/api/estop/status");
    applyEstopState(payload.estop || {});
    return payload;
  } catch (error) {
    applyEstopState({ reason: `status_unavailable:${error.message}` });
    if (!silent) toast(`STOP 状态读取失败：${error.message}`);
    throw error;
  }
}

async function fireGlobalEstop() {
  if (state.estop.requestInFlight) return null;
  state.estop.requestInFlight = true;
  state.estop.latched = true;
  state.estop.transactionState = "stopping";
  state.estop.stopConfirmed = false;
  state.estop.requiresPhysicalEstop = true;
  renderEstopState();
  window.dispatchEvent(new CustomEvent("ruc:estop-state", { detail: { ...state.estop } }));
  try {
    const payload = await api("/api/estop", {
      method: "POST",
      body: JSON.stringify({}),
    });
    applyEstopState(payload.estop || {
      latched: true,
      transaction_state: payload.stop_state,
      stop_confirmed: payload.stop_confirmed,
      requires_physical_estop: payload.requires_physical_estop,
    });
    if (payload.stop_confirmed === true) {
      toast("STOP 已由全部软件安全通道确认；检查现场后方可恢复操作");
    } else {
      toast("STOP 未获得全部通道确认：立即按下物理急停，并检查红色安全状态");
    }
    return payload;
  } catch (error) {
    state.estop.latched = true;
    state.estop.transactionState = "unconfirmed";
    state.estop.stopConfirmed = false;
    state.estop.requiresPhysicalEstop = true;
    state.estop.reason = `request_failed:${error.message}`;
    renderEstopState();
    window.dispatchEvent(new CustomEvent("ruc:estop-state", { detail: { ...state.estop } }));
    toast(`STOP 请求失败：${error.message}。立即使用物理急停。`);
    throw error;
  } finally {
    state.estop.requestInFlight = false;
    renderEstopState();
  }
}

async function resetGlobalEstop() {
  if (!state.estop.latched) return;
  if (!window.confirm("确认机器人已停止、现场无人且可以恢复控制？")) return;
  state.estop.requestInFlight = true;
  renderEstopState();
  try {
    const payload = await api("/api/estop/reset", {
      method: "POST",
      body: JSON.stringify({
        confirm: "RESET_ESTOP",
        physical_estop_reset: true,
        area_clear: true,
      }),
    });
    applyEstopState(payload.estop || {});
    toast("STOP 状态已解除；实机动作仍需重新完成设备预检");
  } catch (error) {
    toast(`STOP 状态解除失败：${error.message}`);
    await refreshEstopStatus({ silent: true }).catch(() => {});
  } finally {
    state.estop.requestInFlight = false;
    renderEstopState();
  }
}

window.rucFireEstop = fireGlobalEstop;

function statusClass(ok) {
  return ok ? "status-dot ok" : "status-dot bad";
}

function fmtTime(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function task1Text(selector, value) {
  const element = $(selector);
  if (element) element.textContent = String(value ?? "");
}

function task1SetStateClass(element, stateName) {
  if (!element) return;
  ["is-active", "is-success", "is-failed", "is-warning", "is-muted", "checking", "ready", "warning"].forEach((name) => {
    element.classList.remove(name);
  });
  if (stateName) {
    element.classList.add(`is-${stateName}`);
    if (stateName === "active") element.classList.add("checking");
    if (stateName === "success") element.classList.add("ready");
    if (["failed", "warning"].includes(stateName)) element.classList.add("warning");
  }
}

function task1Timestamp(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value > 10_000_000_000 ? value / 1000 : value;
  }
  if (typeof value === "string" && value.trim()) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric > 10_000_000_000 ? numeric / 1000 : numeric;
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed / 1000;
  }
  return 0;
}

function task1Duration(seconds) {
  const safe = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const remaining = safe % 60;
  return hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function task1ConfigKey(config) {
  if (!config || typeof config !== "object") return "";
  return String(config.name || config.config_name || config.path || "").trim();
}

function task1ConfigLabel(config) {
  if (!config || typeof config !== "object") return "未命名配置";
  return String(config.name || config.task_name || config.path || "未命名配置");
}

function task1SelectedConfig() {
  const selected = state.task1.selectedConfig;
  return state.task1.configs.find((item) => task1ConfigKey(item) === selected)
    || state.task1.config
    || null;
}

function task1ConfigRequestValue() {
  return state.task1.selectedConfig || task1ConfigKey(task1SelectedConfig());
}

function task1Mode(run = state.task1.run) {
  const raw = String(run?.mode || run?.run_mode || "").toLowerCase();
  return ["execute", "live", "real", "reality"].includes(raw) ? "execute" : "dry-run";
}

function task1Process(run = state.task1.run) {
  return run?.lifecycle?.process || run?.process || run?.job || {};
}

function task1Mission(run = state.task1.run) {
  return run?.mission || {};
}

function task1MissionState(run = state.task1.run) {
  return String(task1Mission(run).state || run?.state || "").toLowerCase();
}

function task1RunIsActive(run = state.task1.run) {
  if (!run) return false;
  const process = task1Process(run);
  if (process.alive === true) return true;
  const lifecyclePhase = String(run?.lifecycle?.phase || "").toLowerCase();
  const processStatus = String(process.status || run?.job?.status || "").toLowerCase();
  if (TASK1_ACTIVE_PROCESS_STATES.has(lifecyclePhase) || TASK1_ACTIVE_PROCESS_STATES.has(processStatus)) return true;
  if (["success", "failed", "cancelled", "canceled", "stopped"].includes(processStatus)) return false;
  const missionState = task1MissionState(run);
  return Boolean(missionState && !TASK1_TERMINAL_STATES.has(missionState) && !process.ended_at);
}

function task1RunSucceeded(run = state.task1.run) {
  if (!run) return false;
  const mission = task1Mission(run);
  const outcome = String(mission.outcome || "").toLowerCase();
  const processStatus = String(task1Process(run).status || run?.job?.status || "").toLowerCase();
  const missionState = task1MissionState(run);
  return ["success", "succeeded", "complete", "completed"].includes(outcome)
    || ["complete", "completed"].includes(missionState)
    || (processStatus === "success" && !mission.error);
}

function task1CurrentTaskId(run = state.task1.run) {
  return String(run?.task_id || run?.id || "").trim();
}

function task1ValidationExpiry(value) {
  return task1Timestamp(value);
}

function task1ValidationTokenIsValid() {
  const expiresAt = Number(state.task1.validationExpiresAt) || 0;
  return Boolean(state.task1.validationToken && (!expiresAt || expiresAt > Date.now() / 1000 + 2));
}

function clearTask1ExecutionApproval() {
  state.task1.validationToken = "";
  state.task1.validationExpiresAt = 0;
  state.task1.deviceValidation = null;
}

function task1Blockers(value) {
  if (!value) return [];
  const items = Array.isArray(value) ? value : [value];
  return items.map((item) => {
    if (typeof item === "string") return { code: "BLOCKED", message: item };
    if (!item || typeof item !== "object") return { code: "BLOCKED", message: String(item) };
    return {
      code: String(item.code || item.id || "BLOCKED"),
      message: String(item.message || item.detail || item.error || item.label || "检查未通过"),
    };
  }).filter((item) => item.message);
}

function task1Checks(value) {
  if (!value) return [];
  const raw = Array.isArray(value) ? value : Array.isArray(value.checks) ? value.checks : [];
  if (raw.length) {
    return raw.map((item, index) => ({
      id: String(item?.id || item?.code || `check-${index + 1}`),
      label: String(item?.label || item?.name || item?.id || `检查 ${index + 1}`),
      ok: typeof item?.ok === "boolean" ? item.ok : null,
      detail: String(item?.detail || item?.message || item?.error || ""),
    }));
  }
  if (typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, item]) => {
    if (["blockers", "ready", "status", "updated_at"].includes(key)) return [];
    if (typeof item === "boolean") {
      return [{ id: key, label: key.replaceAll("_", " "), ok: item, detail: item ? "通过" : "未通过" }];
    }
    if (item && typeof item === "object" && typeof item.ok === "boolean") {
      return [{
        id: key,
        label: String(item.label || key.replaceAll("_", " ")),
        ok: item.ok,
        detail: String(item.detail || item.message || item.error || ""),
      }];
    }
    return [];
  });
}

function normalizeTask1Config(payload) {
  const primary = payload?.config && typeof payload.config === "object" ? payload.config : null;
  let configs = Array.isArray(payload?.configs)
    ? payload.configs.filter((item) => item && typeof item === "object")
    : primary ? [primary] : [];
  if (Array.isArray(payload?.available_configs) && payload.available_configs.length) {
    configs = payload.available_configs.map((name) => {
      const safeName = String(name || "").trim();
      return safeName && safeName === String(primary?.name || "")
        ? primary
        : { name: safeName, path: safeName, execute_ready: false, blockers: [] };
    }).filter((item) => item?.name);
    if (primary && !configs.some((item) => task1ConfigKey(item) === task1ConfigKey(primary))) configs.unshift(primary);
  }
  state.task1.config = primary || configs[0] || null;
  state.task1.configs = configs.length ? configs : state.task1.config ? [state.task1.config] : [];

  const select = $("#task1-config-select");
  if (!select) return;
  const previous = state.task1.selectedConfig || String(state.task1.run?.config_name || "") || select.value;
  const options = state.task1.configs.map((config) => {
    const option = document.createElement("option");
    option.value = task1ConfigKey(config);
    option.textContent = task1ConfigLabel(config);
    return option;
  });
  if (!options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "没有可用的 Task 1 配置";
    options.push(option);
  }
  select.replaceChildren(...options);
  const available = new Set(state.task1.configs.map(task1ConfigKey));
  state.task1.selectedConfig = available.has(previous)
    ? previous
    : task1ConfigKey(state.task1.configs[0]);
  state.task1.config = state.task1.configs.find((item) => task1ConfigKey(item) === state.task1.selectedConfig)
    || state.task1.config;
  select.value = state.task1.selectedConfig;
}

function normalizeTask1Current(payload) {
  const run = payload?.run && typeof payload.run === "object"
    ? payload.run
    : payload?.snapshot && typeof payload.snapshot === "object"
      ? payload.snapshot
      : null;
  state.task1.run = run;
  state.task1.active = typeof payload?.active === "boolean" ? payload.active : task1RunIsActive(run);
  state.task1.lastStatusAt = Date.now();
  state.task1.stopRequestedLocally = state.task1.active
    ? state.task1.stopRequestedLocally || Boolean(run?.lifecycle?.stop?.requested)
    : false;
  const runConfigName = String(run?.config_name || "");
  const matchingConfig = state.task1.configs.find((item) => (
    task1ConfigKey(item) === runConfigName || String(item?.name || "") === runConfigName
  ));
  if (matchingConfig) {
    state.task1.selectedConfig = task1ConfigKey(matchingConfig);
    state.task1.config = matchingConfig;
    const select = $("#task1-config-select");
    if (select) select.value = state.task1.selectedConfig;
  }

  if (run && !state.task1.active && task1Mode(run) === "dry-run" && task1RunSucceeded(run)) {
    const runConfig = String(run.config_name || run.config || task1ConfigRequestValue());
    state.task1.dryRunApprovedConfig = runConfig || task1ConfigRequestValue();
  }
  const taskId = task1CurrentTaskId(run);
  if (run && !state.task1.active && task1Mode(run) === "execute" && taskId && state.task1.lastTerminalTaskId !== taskId) {
    state.task1.lastTerminalTaskId = taskId;
    state.task1.dryRunApprovedConfig = "";
    clearTask1ExecutionApproval();
  }
}

function switchAgenticTab(tabName) {
  const tab = tabName === "debug" ? "debug" : "task";
  state.task1.activeTab = tab;
  $$('[data-agentic-tab]').forEach((button) => {
    const active = button.dataset.agenticTab === tab;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  const taskPanel = $("#agentic-task-panel");
  const debugPanel = $("#agentic-debug-panel");
  if (taskPanel) taskPanel.hidden = tab !== "task";
  if (debugPanel) debugPanel.hidden = tab !== "debug";
  if (tab === "task") {
    refreshTask1Current({ silent: true }).catch(() => {});
  } else {
    refreshArtifacts().catch((error) => toast(error.message));
    refreshJobs().catch((error) => toast(error.message));
  }
}

function task1ConfigMatches(value) {
  const textValue = String(value || "").trim();
  if (!textValue) return false;
  const config = task1SelectedConfig();
  return [
    state.task1.selectedConfig,
    task1ConfigKey(config),
    config?.name,
    config?.path,
    config?.task_name,
  ].some((candidate) => String(candidate || "").trim() === textValue);
}

function task1DryRunApproved() {
  return task1ConfigMatches(state.task1.dryRunApprovedConfig);
}

function task1LifecycleInfo() {
  const run = state.task1.run;
  const mission = task1Mission(run);
  const process = task1Process(run);
  const stateName = task1MissionState(run);
  const stopped = Boolean(run?.lifecycle?.stop?.requested || state.task1.stopRequestedLocally);
  const telemetryAgeMs = state.task1.lastStatusAt
    ? Date.now() - state.task1.lastStatusAt
    : Number.POSITIVE_INFINITY;
  if (state.task1.active && state.task1.lastError && telemetryAgeMs > 2500) {
    return { text: "遥测失联 · 状态未知", modifier: "failed" };
  }
  if (state.task1.busyAction === "stop" || (state.task1.active && stopped)) {
    return { text: "安全停止中", modifier: "warning" };
  }
  if (state.task1.active) {
    return { text: TASK1_STATE_LABELS[stateName] || "任务执行中", modifier: "active" };
  }
  if (run && task1RunSucceeded(run)) return { text: "任务已完成", modifier: "success" };
  if (run && (mission.error || ["failed", "cancelled", "canceled"].includes(String(process.status || "").toLowerCase()))) {
    return { text: stopped ? "任务已停止" : "任务失败", modifier: stopped ? "warning" : "failed" };
  }
  if (state.task1.lastError) return { text: "状态同步失败", modifier: "failed" };
  if (state.task1.config) return { text: "待命", modifier: "muted" };
  return { text: "读取配置中", modifier: "active" };
}

function renderTask1Overview() {
  const run = state.task1.run;
  const modeBadge = $("#task1-mode-badge");
  const statusBadge = $("#task1-status-badge");
  const lifecycle = task1LifecycleInfo();
  if (modeBadge) {
    modeBadge.textContent = run ? (task1Mode(run) === "execute" ? "实机任务" : "软件预演") : "尚未运行";
    task1SetStateClass(modeBadge, run ? (task1Mode(run) === "execute" ? "warning" : "muted") : "muted");
  }
  if (statusBadge) {
    statusBadge.textContent = lifecycle.text;
    task1SetStateClass(statusBadge, lifecycle.modifier);
  }
  const process = task1Process(run);
  const startedAt = task1Timestamp(process.started_at || run?.started_at);
  const endedAt = task1Timestamp(process.ended_at || run?.ended_at) || Date.now() / 1000;
  task1Text("#task1-elapsed", startedAt ? task1Duration(endedAt - startedAt) : "00:00");
}

function task1ReadinessChecks() {
  const config = task1SelectedConfig();
  const checks = [];
  if (state.task1.configValidation) {
    checks.push(...task1Checks(state.task1.configValidation));
    if (!task1Checks(state.task1.configValidation).length) {
      checks.push({
        id: "config",
        label: "任务配置",
        ok: Boolean(state.task1.configValidation.ready),
        detail: state.task1.configValidation.ready ? "配置检查通过" : "配置检查未通过",
      });
    }
  } else {
    checks.push({
      id: "config",
      label: "任务配置",
      ok: config ? null : false,
      detail: config ? "已加载，等待检查" : "未读取到配置",
    });
  }

  const deviceSource = state.task1.deviceValidation || state.task1.run?.preflight;
  const deviceChecks = task1Checks(deviceSource);
  if (deviceChecks.length) {
    checks.push(...deviceChecks);
  } else {
    ["导航服务", "相机与感知", "机械臂与夹爪"].forEach((label, index) => {
      checks.push({ id: `device-${index}`, label, ok: null, detail: "未执行设备预检" });
    });
  }
  if (state.task1.deviceValidation) {
    checks.push({
      id: "execution-token",
      label: "实机启动许可",
      ok: task1ValidationTokenIsValid(),
      detail: task1ValidationTokenIsValid() ? "设备预检许可有效" : "许可已失效，请重新预检",
    });
  }
  return checks;
}

function renderTask1Readiness() {
  const root = $("#task1-readiness-list");
  const checks = task1ReadinessChecks();
  if (root) {
    const elements = checks.map((check) => {
      const article = document.createElement("article");
      article.className = "task1-readiness-item";
      const modifier = check.ok === true ? "success" : check.ok === false ? "failed" : "muted";
      task1SetStateClass(article, modifier);
      const dot = document.createElement("span");
      dot.className = `status-dot ${check.ok === true ? "ok" : check.ok === false ? "bad" : "muted"}`;
      const label = document.createElement("span");
      label.textContent = check.label;
      const detail = document.createElement("strong");
      detail.textContent = check.detail || (check.ok === true ? "通过" : check.ok === false ? "未通过" : "等待");
      article.append(dot, label, detail);
      return article;
    });
    root.replaceChildren(...elements);
  }

  const config = task1SelectedConfig();
  const blockers = task1Blockers(config?.blockers);
  let summary = "可先执行软件预演；实机启动前必须重新完成设备预检";
  if (!config) summary = "未读取到可用配置";
  else if (state.task1.active) summary = "任务运行中，启动配置已锁定";
  else if (blockers.length) summary = `发现 ${blockers.length} 个实机启动阻塞项`;
  else if (task1ValidationTokenIsValid()) {
    summary = state.task1.validationExpiresAt
      ? `设备预检通过，许可有效至 ${fmtTime(state.task1.validationExpiresAt)}`
      : "设备预检通过，实机启动许可有效";
  }
  task1Text("#task1-readiness-summary", summary);
}

function task1EffectiveMissionState() {
  const run = state.task1.run;
  const current = task1MissionState(run);
  if (current && current !== "failed") return current;
  const events = Array.isArray(run?.events) ? run.events : [];
  const previous = [...events].reverse().find((event) => {
    const name = String(event?.state || "").toLowerCase();
    return name && name !== "failed";
  });
  return String(previous?.state || current || "created").toLowerCase();
}

function renderTask1Phases() {
  const root = $("#task1-phase-strip");
  const stateName = task1EffectiveMissionState();
  let activeIndex = TASK1_PHASE_GROUPS.findIndex((group) => group.states.includes(stateName));
  if (activeIndex < 0) activeIndex = state.task1.run ? 0 : -1;
  const failed = task1MissionState() === "failed" || Boolean(task1Mission().error);
  const complete = task1RunSucceeded();
  if (root) {
    const steps = TASK1_PHASE_GROUPS.map((group, index) => {
      const article = document.createElement("article");
      article.className = "task1-phase-step";
      let modifier = "";
      let status = "等待";
      if (complete || (activeIndex >= 0 && index < activeIndex)) {
        modifier = "success";
        status = "完成";
      } else if (index === activeIndex) {
        modifier = failed ? "failed" : state.task1.active ? "active" : "warning";
        status = failed ? "失败" : state.task1.active ? "进行中" : "停留";
      }
      task1SetStateClass(article, modifier);
      const number = document.createElement("span");
      number.textContent = String(index + 1);
      const label = document.createElement("strong");
      label.textContent = group.label;
      const stateLabel = document.createElement("small");
      stateLabel.textContent = status;
      article.append(number, label, stateLabel);
      return article;
    });
    root.replaceChildren(...steps);
  }
  const progress = state.task1.run?.progress || {};
  const planned = Number(progress.planned_jobs) || 0;
  const delivered = Number(progress.delivered_jobs) || 0;
  const currentLabel = TASK1_STATE_LABELS[task1MissionState()] || (state.task1.run ? "等待任务状态" : "等待环境检查");
  task1Text("#task1-phase-summary", planned ? `${currentLabel} · 已交付 ${delivered}/${planned}` : currentLabel);
}

function task1ReceiptLines() {
  const run = state.task1.run;
  const receiptLines = Array.isArray(run?.receipt?.lines) ? run.receipt.lines : [];
  if (receiptLines.length) return receiptLines;
  return Array.isArray(task1Mission(run)?.order) ? task1Mission(run).order : [];
}

function renderTask1Order() {
  const run = state.task1.run;
  const receipt = run?.receipt || {};
  const lines = task1ReceiptLines();
  const jobs = Array.isArray(run?.jobs) ? run.jobs : [];
  const progress = run?.progress || {};
  const delivered = Number(progress.delivered_jobs) || 0;
  const planned = Number(progress.planned_jobs) || jobs.length || 0;
  const currentJobId = String(progress.current_job_id || "");
  task1Text(
    "#task1-order-summary",
    lines.length ? `已识别 ${lines.length} 类商品 · 已交付 ${delivered}/${planned || lines.length}` : "尚未读取小票",
  );
  const receiptStatus = String(receipt.status || "").toLowerCase();
  const receiptText = receipt.error
    ? `小票读取失败：${receipt.error}`
    : receiptStatus
      ? `小票状态：${receiptStatus}${receipt.attempt ? ` · 第 ${receipt.attempt} 次` : ""}`
      : "等待小票图像与 OCR 结果";
  task1Text("#task1-receipt-preview", receiptText);

  const root = $("#task1-order-list");
  if (!root) return;
  if (!lines.length && !jobs.length) {
    const empty = document.createElement("div");
    empty.className = "task1-empty-state";
    empty.textContent = "订单解析后将在这里显示商品";
    root.replaceChildren(empty);
    return;
  }
  const source = lines.length ? lines : jobs;
  const items = source.map((line, index) => {
    const matchingJob = jobs.find((job) => {
      const lineSku = String(line.sku_id || "");
      return (lineSku && String(job.sku_id || "") === lineSku) || (!lineSku && index === Number(job.sequence || index + 1) - 1);
    });
    const jobId = String(matchingJob?.job_id || matchingJob?.id || "");
    const article = document.createElement("article");
    article.className = "task1-order-item";
    if (matchingJob?.delivered || matchingJob?.status === "delivered") task1SetStateClass(article, "success");
    else if (jobId && jobId === currentJobId) task1SetStateClass(article, "active");
    const title = document.createElement("strong");
    title.textContent = String(line.name || matchingJob?.name || line.sku_id || `商品 ${index + 1}`);
    const spec = document.createElement("span");
    spec.textContent = String(line.spec || matchingJob?.spec || "规格未提供");
    const quantity = document.createElement("small");
    quantity.textContent = `数量 ${Number(line.quantity || line.unit_count || 1)}${matchingJob?.shelf_waypoint ? ` · ${matchingJob.shelf_waypoint}` : ""}`;
    article.append(title, spec, quantity);
    return article;
  });
  root.replaceChildren(...items);
}

function renderTask1CurrentAction() {
  const run = state.task1.run;
  const progress = run?.progress || {};
  const stateName = task1MissionState(run);
  const jobs = Array.isArray(run?.jobs) ? run.jobs : [];
  const currentJobId = String(progress.current_job_id || "");
  const currentJob = jobs.find((job) => String(job.job_id || job.id || "") === currentJobId);
  const action = run ? (TASK1_STATE_LABELS[stateName] || "等待状态机更新") : "等待开始任务";
  const planned = Number(progress.planned_jobs) || jobs.length || 0;
  const delivered = Number(progress.delivered_jobs) || 0;
  const details = [];
  if (currentJob) details.push(`${currentJob.name || currentJob.sku_id || currentJobId}${currentJob.spec ? ` · ${currentJob.spec}` : ""}`);
  if (planned) details.push(`任务进度 ${delivered}/${planned}`);
  if (state.task1.active && task1Mode(run) === "execute") details.push("实机运动已启用，请持续观察现场");
  if (!details.length) details.push(run ? "等待下一条结构化任务事件" : "完成配置检查和设备预检后，可以启动任务。 ");
  task1Text("#task1-current-action", action);
  task1Text("#task1-current-detail", details.join(" · "));

  const navigation = run?.navigation || {};
  const navStatus = String(navigation.status || "");
  task1Text("#task1-navigation-status", navStatus ? `导航：${navStatus}` : "底盘待命");
  const navDetails = [
    `当前航点：${navigation.waypoint || currentJob?.shelf_waypoint || "--"}`,
    navigation.attempt ? `第 ${navigation.attempt} 次尝试` : "",
    navigation.error ? `异常：${navigation.error}` : "",
  ].filter(Boolean);
  task1Text("#task1-navigation-detail", navDetails.join(" · "));

  const modeLabel = state.task1.busyAction
    ? ({ config: "配置检查中", preflight: "设备预检中", dry: "启动预演中", live: "启动实机中", stop: "停止请求中" }[state.task1.busyAction] || "处理中")
    : state.task1.active
      ? task1Mode(run) === "execute" ? "实机执行中" : "软件预演中"
      : "待命";
  task1Text("#task1-run-mode", modeLabel);
}

function task1AllBlockers() {
  const run = state.task1.run;
  return [
    ...task1Blockers(task1SelectedConfig()?.blockers),
    ...task1Blockers(state.task1.configValidation?.blockers),
    ...task1Blockers(state.task1.deviceValidation?.blockers),
    ...task1Blockers(run?.preflight?.blockers),
    ...task1Blockers(run?.blockers),
  ].filter((item, index, all) => all.findIndex((candidate) => (
    candidate.code === item.code && candidate.message === item.message
  )) === index);
}

function renderTask1Errors() {
  const panel = $("#task1-error-panel");
  if (!panel) return;
  const run = state.task1.run;
  const mission = task1Mission(run);
  const receiptError = run?.receipt?.error;
  const navigationError = run?.navigation?.error;
  const runtimeError = mission.error || run?.error || receiptError || navigationError || state.task1.lastError;
  const blockers = task1AllBlockers();
  const show = Boolean(runtimeError || blockers.length);
  panel.hidden = !show;
  if (!show) return;
  const firstBlocker = blockers[0];
  const code = mission.error_code || run?.error_code || firstBlocker?.code || "TASK1_ERROR";
  task1Text("#task1-error-code", `错误码：${code}`);
  task1Text(
    "#task1-error-message",
    runtimeError || (blockers.length ? "当前条件不满足实机启动要求，请处理以下阻塞项。" : "任务发生异常。"),
  );
  const root = $("#task1-blocker-list");
  if (root) {
    const items = blockers.map((blocker) => {
      const item = document.createElement("div");
      item.textContent = `${blocker.code} · ${blocker.message}`;
      return item;
    });
    root.replaceChildren(...items);
  }
}

function task1EventDescription(event) {
  const eventName = String(event?.event || event?.type || "event");
  const parts = [TASK1_EVENT_LABELS[eventName] || eventName.replaceAll("_", " ")];
  if (event?.waypoint) parts.push(`航点 ${event.waypoint}`);
  if (event?.job) parts.push(`商品 ${event.job}`);
  if (event?.attempt) parts.push(`第 ${event.attempt} 次`);
  if (event?.delivered_jobs !== undefined) parts.push(`已交付 ${event.delivered_jobs}`);
  else if (event?.delivered !== undefined) parts.push(`已交付 ${event.delivered}`);
  if (event?.error) parts.push(String(event.error));
  return parts.join(" · ");
}

function renderTask1Events() {
  const root = $("#task1-event-list");
  if (!root) return;
  const timeline = Array.isArray(state.task1.run?.events) ? state.task1.run.events : [];
  const logWarnings = Array.isArray(state.task1.run?.log_health?.warnings)
    ? state.task1.run.log_health.warnings
      .filter((warning) => warning?.code !== "partial_tail_ignored")
      .slice(-5)
      .map((warning) => ({
        event: "runtime_warning",
        timestamp: state.task1.run?.mission?.last_event_at,
        error: String(warning?.message || warning?.code || "任务日志状态异常"),
      }))
    : [];
  const events = [...timeline, ...logWarnings].slice(-40).reverse();
  if (!events.length) {
    const item = document.createElement("li");
    item.className = "task1-event-item is-muted";
    const time = document.createElement("time");
    time.textContent = "--:--:--";
    const description = document.createElement("span");
    description.textContent = "等待任务事件";
    item.append(time, description);
    root.replaceChildren(item);
    return;
  }
  const items = events.map((event) => {
    const item = document.createElement("li");
    item.className = "task1-event-item";
    const eventName = String(event?.event || event?.type || "");
    if (["mission_complete", "job_delivered"].includes(eventName)) task1SetStateClass(item, "success");
    else if (eventName === "runtime_warning") task1SetStateClass(item, "warning");
    else if (["mission_failed", "receipt_rejected"].includes(eventName) || event?.error) task1SetStateClass(item, "failed");
    else if (eventName === "stop_requested") task1SetStateClass(item, "warning");
    const time = document.createElement("time");
    const timestamp = task1Timestamp(event?.timestamp || event?.time);
    time.textContent = timestamp ? new Date(timestamp * 1000).toLocaleTimeString() : "--:--:--";
    const description = document.createElement("span");
    description.textContent = task1EventDescription(event);
    item.append(time, description);
    return item;
  });
  root.replaceChildren(...items);
}

function task1SafeArtifactUrl(value) {
  if (typeof value !== "string" || !value) return "";
  try {
    const parsed = new URL(value, window.location.origin);
    const allowed = parsed.origin === window.location.origin
      && (parsed.pathname.startsWith("/artifacts/") || parsed.pathname.startsWith("/api/task1/"));
    return allowed ? `${parsed.pathname}${parsed.search}` : "";
  } catch (_error) {
    return "";
  }
}

function renderTask1Observability() {
  const run = state.task1.run;
  const log = String(run?.job?.log || run?.log || "");
  task1Text("#task1-log", log ? log.slice(-40_000) : run ? "任务已创建，等待日志输出…" : "等待任务启动…");
  const root = $("#task1-artifacts");
  if (!root) return;
  const artifacts = Array.isArray(run?.artifacts) ? run.artifacts : [];
  if (!artifacts.length) {
    const empty = document.createElement("div");
    empty.className = "task1-empty-state";
    empty.textContent = "暂无任务产物";
    root.replaceChildren(empty);
    return;
  }
  const items = artifacts.slice(0, 30).map((artifact, index) => {
    const name = String(artifact?.name || artifact?.label || `任务产物 ${index + 1}`);
    const url = task1SafeArtifactUrl(artifact?.url || artifact?.href || "");
    const article = document.createElement("article");
    article.className = "task1-artifact";
    if (!url) {
      const item = document.createElement("span");
      item.textContent = name;
      article.append(item);
      return article;
    }
    const suffix = String(artifact?.suffix || name.slice(name.lastIndexOf("."))).toLowerCase();
    if ([".png", ".jpg", ".jpeg", ".webp"].includes(suffix)) {
      const preview = document.createElement("img");
      preview.src = url;
      preview.alt = name;
      preview.loading = "lazy";
      article.append(preview);
    }
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = name;
    article.append(link);
    return article;
  });
  root.replaceChildren(...items);
}

function renderTask1CameraStatus() {
  const cameras = Array.isArray(state.cameras) ? state.cameras : [];
  if (!cameras.length) {
    task1Text("#task1-camera-status", "相机状态未知");
    return;
  }
  const configured = cameras.filter((camera) => camera.configured || camera.source).length;
  const streaming = cameras.filter((camera) => camera.runtime?.ok || camera.runtime?.message === "streaming").length;
  const liveExclusive = state.task1.active && task1Mode() === "execute";
  task1Text(
    "#task1-camera-status",
    liveExclusive && streaming < configured ? "自主任务占用相机" : `${streaming || configured}/${cameras.length} 路可用`,
  );
}

function task1LiveBlockedReason() {
  const config = task1SelectedConfig();
  if (state.estop.latched !== false) return "Web Console STOP 状态尚未解除";
  if (state.task1.active) return "已有 Task 1 正在运行";
  if (state.task1.busyAction) return "当前操作尚未完成";
  if (!config) return "没有可用的任务配置";
  if (!config.hardware_actions_enabled && !state.task1.config?.hardware_actions_enabled) return "Web Console 未启用硬件动作";
  if (config.execute_ready === false) return "任务配置尚未满足实机标定要求";
  if (task1Blockers(config.blockers).length) return "任务配置仍有实机阻塞项";
  if (!task1DryRunApproved()) return "请先完成当前配置的软件预演";
  if (!task1ValidationTokenIsValid()) return "请重新执行设备预检以获取实机启动许可";
  return "";
}

function renderTask1Controls() {
  const active = state.task1.active;
  const busy = Boolean(state.task1.busyAction);
  const hasConfig = Boolean(task1ConfigRequestValue());
  const stopRequested = Boolean(state.task1.run?.lifecycle?.stop?.requested || state.task1.stopRequestedLocally);
  const select = $("#task1-config-select");
  if (select) select.disabled = active || busy || !state.task1.configs.length;
  const validate = $("#task1-validate");
  const preflight = $("#task1-preflight");
  const dryRun = $("#task1-dry-run");
  const liveRun = $("#task1-start-live");
  const stop = $("#task1-software-stop");
  if (validate) validate.disabled = active || busy || !hasConfig;
  if (preflight) preflight.disabled = active || busy || !hasConfig;
  const config = task1SelectedConfig();
  const dryRunUnavailable = config?.dry_run_ready === false;
  if (dryRun) {
    dryRun.disabled = active || busy || !hasConfig || dryRunUnavailable;
    const dryBlockers = task1Blockers(config?.dry_run_blockers);
    dryRun.title = dryRunUnavailable
      ? dryBlockers.map((item) => item.message).join("；") || "软件预演配置或小票样例未就绪"
      : "仅运行状态机预演，不会下发机器人运动";
  }
  const liveReason = task1LiveBlockedReason();
  if (liveRun) {
    liveRun.disabled = Boolean(liveReason);
    liveRun.title = liveReason || "按住按钮 1.2 秒，并在二次确认后开始实机任务";
    if (!state.task1.liveHoldStarted) liveRun.textContent = liveReason ? "实机启动未就绪" : "按住开始实机任务";
  }
  if (stop) {
    stop.disabled = !active || busy || !task1CurrentTaskId();
    stop.textContent = stopRequested ? "重试安全停止" : "软件停止";
  }
}

function renderTask1() {
  renderTask1Overview();
  renderTask1Readiness();
  renderTask1Phases();
  renderTask1Order();
  renderTask1CurrentAction();
  renderTask1Errors();
  renderTask1Events();
  renderTask1Observability();
  renderTask1CameraStatus();
  renderTask1Controls();
}

async function refreshTask1Config(options = {}) {
  try {
    const query = options.name ? `?name=${encodeURIComponent(options.name)}` : "";
    const payload = await api(`/api/task1/config${query}`);
    normalizeTask1Config(payload);
    if (state.task1.lastError.startsWith("配置读取失败")) state.task1.lastError = "";
    renderTask1();
    return payload;
  } catch (error) {
    state.task1.lastError = `配置读取失败：${error.message}`;
    renderTask1();
    if (!options.silent) toast(state.task1.lastError);
    throw error;
  }
}

async function refreshTask1Current(options = {}) {
  if (state.task1.statusLoading) return null;
  state.task1.statusLoading = true;
  try {
    const payload = await api("/api/task1/runs/current/status");
    normalizeTask1Current(payload);
    state.task1.lastError = "";
    renderTask1();
    return payload;
  } catch (error) {
    state.task1.lastError = `任务状态读取失败：${error.message}`;
    renderTask1();
    if (!options.silent) toast(state.task1.lastError);
    throw error;
  } finally {
    state.task1.statusLoading = false;
  }
}

function scheduleTask1Poll(delay = null) {
  if (!state.task1.pollingStarted) return;
  window.clearTimeout(state.task1.pollTimer);
  const interval = delay ?? (state.task1.active ? TASK1_ACTIVE_POLL_MS : TASK1_IDLE_POLL_MS);
  state.task1.pollTimer = window.setTimeout(async () => {
    try {
      await refreshTask1Current({ silent: true });
    } catch (_error) {
      // The error panel retains the failure. Polling stays alive for recovery.
    } finally {
      scheduleTask1Poll();
    }
  }, Math.max(0, interval));
}

function startTask1Polling() {
  if (state.task1.pollingStarted) return;
  state.task1.pollingStarted = true;
  scheduleTask1Poll();
}

function task1ActionPayload(mode) {
  return {
    config: task1ConfigRequestValue(),
    mode,
  };
}

async function validateTask1(scope) {
  const device = scope === "device";
  const actionName = device ? "preflight" : "config";
  clearTask1ExecutionApproval();
  if (!device) state.task1.configValidation = null;
  state.task1.busyAction = actionName;
  renderTask1();
  try {
    const payload = await api("/api/task1/validate", {
      method: "POST",
      body: JSON.stringify({
        ...task1ActionPayload(device ? "execute" : "dry-run"),
        scope: device ? "devices" : "config",
      }),
    });
    if (device) {
      state.task1.deviceValidation = payload;
      if (payload.ready && payload.validation_token) {
        state.task1.validationToken = String(payload.validation_token);
        state.task1.validationExpiresAt = task1ValidationExpiry(payload.expires_at);
      } else {
        clearTask1ExecutionApproval();
        state.task1.deviceValidation = payload;
      }
    } else {
      state.task1.configValidation = payload;
    }
    toast(payload.ready
      ? device ? "设备预检通过，实机许可已生成" : "任务配置检查通过"
      : device ? "设备预检未通过，请查看阻塞项" : "任务配置检查未通过");
    return payload;
  } catch (error) {
    clearTask1ExecutionApproval();
    const failure = {
      ready: false,
      checks: [],
      blockers: [{ code: device ? "PREFLIGHT_FAILED" : "CONFIG_CHECK_FAILED", message: error.message }],
    };
    if (device) state.task1.deviceValidation = failure;
    else state.task1.configValidation = failure;
    toast(error.message);
    throw error;
  } finally {
    state.task1.busyAction = "";
    renderTask1();
  }
}

function task1RunFromResponse(payload, mode) {
  if (payload?.run && typeof payload.run === "object") return payload.run;
  return {
    task_id: payload?.task_id || "",
    mode,
    config_name: task1ConfigRequestValue(),
    lifecycle: {
      phase: "starting",
      process: {
        alive: true,
        status: payload?.job?.status || "queued",
        started_at: payload?.job?.started_at || payload?.job?.created_at || Date.now() / 1000,
      },
      stop: { requested: false },
    },
    mission: { state: "created", outcome: "", error: null },
    progress: { planned_jobs: 0, delivered_jobs: 0, current_job_id: "", ratio: 0 },
    receipt: { status: "pending", lines: [] },
    navigation: { status: "idle", waypoint: "" },
    jobs: [],
    events: [],
    job: payload?.job || null,
    artifacts: [],
  };
}

async function startTask1Run(mode) {
  const real = mode === "execute";
  if (real) {
    const blocked = task1LiveBlockedReason();
    if (blocked) throw new Error(blocked);
  } else {
    clearTask1ExecutionApproval();
    state.task1.dryRunApprovedConfig = "";
  }
  state.task1.busyAction = real ? "live" : "dry";
  state.task1.stopRequestedLocally = false;
  renderTask1();
  try {
    const body = {
      ...task1ActionPayload(mode),
      confirm: real ? "RUN_TASK1" : "",
    };
    if (real) {
      body.validation_token = state.task1.validationToken;
      // The backend token is single-use. Consume the browser copy before the
      // request so any rejected or ambiguous response requires a fresh preflight.
      state.task1.validationToken = "";
      state.task1.validationExpiresAt = 0;
    }
    const payload = await startJob("/api/task1/runs", body);
    const run = task1RunFromResponse(payload, mode);
    normalizeTask1Current({ active: true, run });
    if (real) {
      state.task1.dryRunApprovedConfig = "";
    }
    toast(real ? "实机任务已启动，请持续观察现场" : "软件预演已启动，不会下发机器人运动");
    await refreshJobs().catch(() => {});
    scheduleTask1Poll(0);
    return payload;
  } catch (error) {
    toast(error.message);
    throw error;
  } finally {
    state.task1.busyAction = "";
    state.task1.liveRequestStarted = false;
    renderTask1();
  }
}

async function confirmAndStartTask1Live() {
  const blocked = task1LiveBlockedReason();
  if (blocked) {
    toast(blocked);
    return;
  }
  const confirmed = window.confirm(
    "确认开始 Task 1 实机任务？\n\n请确认人员已离开机器人运动范围、物理急停可触达、货架和桌面与预演一致。",
  );
  if (!confirmed) {
    toast("已取消实机启动");
    return;
  }
  state.task1.liveRequestStarted = true;
  await startTask1Run("execute");
}

function cancelTask1LiveHold() {
  state.task1.liveHoldStarted = false;
  if (state.task1.liveHoldTimer) {
    window.clearTimeout(state.task1.liveHoldTimer);
    state.task1.liveHoldTimer = null;
  }
  const button = $("#task1-start-live");
  if (button) button.classList.remove("is-holding");
  renderTask1Controls();
}

function beginTask1LiveHold(event) {
  const button = $("#task1-start-live");
  if (!button || button.disabled || state.task1.liveHoldStarted || state.task1.liveRequestStarted) return;
  if (event?.pointerType === "mouse" && event.button !== 0) return;
  state.task1.liveHoldStarted = true;
  button.classList.add("is-holding");
  button.textContent = "继续按住以确认实机启动…";
  state.task1.liveHoldTimer = window.setTimeout(() => {
    state.task1.liveHoldTimer = null;
    state.task1.liveHoldStarted = false;
    button.classList.remove("is-holding");
    confirmAndStartTask1Live().catch((error) => toast(error.message));
  }, TASK1_LIVE_HOLD_MS);
}

async function stopTask1() {
  const taskId = task1CurrentTaskId();
  if (!state.task1.active || !taskId) return;
  if (!window.confirm("确认请求 Task 1 软件停止？\n机器人将执行安全停机流程；如有紧急危险，请直接使用物理急停。")) return;
  state.task1.busyAction = "stop";
  state.task1.stopRequestedLocally = true;
  renderTask1();
  try {
    const payload = await api(`/api/task1/runs/${encodeURIComponent(taskId)}/stop`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    if (payload.run && typeof payload.run === "object") normalizeTask1Current({ active: true, run: payload.run });
    if (payload.stop_confirmed === false || payload.requires_physical_estop === true) {
      toast("Task 1 软件停止未获双通道确认：立即按物理急停；可再次点击“重试安全停止”");
      await refreshEstopStatus({ silent: true }).catch(() => {});
    } else {
      toast(payload.message || "停止请求已发送，正在等待进程安全退出");
    }
    scheduleTask1Poll(0);
  } catch (error) {
    state.task1.stopRequestedLocally = false;
    toast(error.message);
    throw error;
  } finally {
    state.task1.busyAction = "";
    renderTask1();
  }
}

function bindTask1Events() {
  $$('[data-agentic-tab]').forEach((button) => {
    button.addEventListener("click", () => switchAgenticTab(button.dataset.agenticTab));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const next = button.dataset.agenticTab === "task" ? "debug" : "task";
      switchAgenticTab(next);
      $(`[data-agentic-tab="${next}"]`)?.focus();
    });
  });
  $("#task1-config-select")?.addEventListener("change", (event) => {
    state.task1.selectedConfig = event.currentTarget.value;
    state.task1.config = task1SelectedConfig();
    state.task1.configValidation = null;
    state.task1.dryRunApprovedConfig = "";
    clearTask1ExecutionApproval();
    renderTask1();
    refreshTask1Config({ name: state.task1.selectedConfig, silent: true }).catch(() => {});
  });
  $("#task1-validate")?.addEventListener("click", () => {
    validateTask1("config").catch(() => {});
  });
  $("#task1-preflight")?.addEventListener("click", () => {
    validateTask1("device").catch(() => {});
  });
  $("#task1-dry-run")?.addEventListener("click", () => {
    startTask1Run("dry-run").catch(() => {});
  });
  $("#task1-software-stop")?.addEventListener("click", () => {
    stopTask1().catch(() => {});
  });
  const liveButton = $("#task1-start-live");
  liveButton?.addEventListener("pointerdown", beginTask1LiveHold);
  ["pointerup", "pointerleave", "pointercancel"].forEach((eventName) => {
    liveButton?.addEventListener(eventName, cancelTask1LiveHold);
  });
  liveButton?.addEventListener("click", (event) => {
    event.preventDefault();
  });
  liveButton?.addEventListener("keydown", (event) => {
    if (!["Enter", " "].includes(event.key) || event.repeat) return;
    event.preventDefault();
    beginTask1LiveHold(event);
  });
  liveButton?.addEventListener("keyup", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    cancelTask1LiveHold();
  });
  liveButton?.addEventListener("blur", cancelTask1LiveHold);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      refreshTask1Current({ silent: true }).catch(() => {});
      if (state.activeView === "agentic-view") refreshJobs().catch(() => {});
    }
  });
}

function idealStatusLabel(status) {
  const labels = {
    running: "执行中",
    success: "成功",
    failed: "失败",
    "dry-run": "预演",
    unknown: "未知",
  };
  return labels[status] || status || "未知";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalizeAngle(value) {
  let angle = Number.isFinite(value) ? Number(value) : 0;
  while (angle > Math.PI) angle -= Math.PI * 2;
  while (angle <= -Math.PI) angle += Math.PI * 2;
  return angle;
}

function robotHeadYaw(value) {
  return normalizeAngle(Math.PI - Number(value));
}

function robotHeadYawDeg(value) {
  return robotHeadYaw(value) * 180 / Math.PI;
}

function idealCanvasMetrics(canvas) {
  const pad = 54;
  const maxWidth = canvas.width - pad * 2;
  const maxHeight = canvas.height - pad * 2;
  const scale = Math.min(maxWidth / IDEAL_MAP.width, maxHeight / IDEAL_MAP.height);
  const mapWidth = IDEAL_MAP.width * scale;
  const mapHeight = IDEAL_MAP.height * scale;
  const originX = pad + (maxWidth - mapWidth) / 2;
  const originY = pad + (maxHeight - mapHeight) / 2;
  return { pad, scale, originX, originY, mapWidth, mapHeight, side: mapWidth };
}

function mapToCanvas(point, canvas) {
  const { scale, originX, originY, mapHeight } = idealCanvasMetrics(canvas);
  return {
    x: originX + Number(point.x) * scale,
    y: originY + mapHeight - Number(point.y) * scale,
  };
}

function mapRectToCanvas(rect, canvas) {
  const min = mapToCanvas({ x: rect.x, y: rect.y }, canvas);
  const max = mapToCanvas({ x: rect.x + rect.w, y: rect.y + rect.h }, canvas);
  return {
    x: min.x,
    y: max.y,
    w: max.x - min.x,
    h: min.y - max.y,
  };
}

function canvasToMap(clientX, clientY, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (clientX - rect.left) * scaleX;
  const py = (clientY - rect.top) * scaleY;
  const { scale, originX, originY, mapHeight } = idealCanvasMetrics(canvas);
  return {
    x: clamp((px - originX) / scale, IDEAL_MAP.robotHalfX, IDEAL_MAP.width - IDEAL_MAP.robotHalfX),
    y: clamp((originY + mapHeight - py) / scale, IDEAL_MAP.robotHalfY, IDEAL_MAP.height - IDEAL_MAP.robotHalfY),
  };
}

function setIdealTarget(point) {
  if (state.idealNav.isRunning) {
    toast("导航执行中，先等待当前任务结束");
    return;
  }
  window.clearInterval(state.idealNav.replayTimer);
  state.idealNav.target = {
    x: clamp(point.x, IDEAL_MAP.robotHalfX, IDEAL_MAP.width - IDEAL_MAP.robotHalfX),
    y: clamp(point.y, IDEAL_MAP.robotHalfY, IDEAL_MAP.height - IDEAL_MAP.robotHalfY),
  };
  state.idealNav.replayPoint = null;
  $("#ideal-goal-x").value = state.idealNav.target.x.toFixed(2);
  $("#ideal-goal-y").value = state.idealNav.target.y.toFixed(2);
  updateIdealPathSummary();
  renderMapPlanResult();
  renderIdealRouteList();
  drawIdealMap();
}

function selectedCompetitionWaypoint() {
  return state.idealNav.competitionWaypoints.find(
    (point) => point.id === state.idealNav.selectedCompetitionWaypointId,
  ) || null;
}

function competitionWaypointLiftHeight(point) {
  const raw = point?.lift_height_mm;
  if (raw === null || raw === undefined || raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? Math.round(value) : null;
}

function syncCompetitionWaypointForm(point = null) {
  if ($("#competition-waypoint-name")) {
    $("#competition-waypoint-name").value = point?.name || "";
  }
  if ($("#competition-waypoint-kind")) {
    $("#competition-waypoint-kind").value = point?.kind || "custom";
  }
  if ($("#competition-waypoint-yaw")) {
    const yaw = Number(point?.yaw ?? state.idealNav.finalYaw ?? 0);
    $("#competition-waypoint-yaw").value = (normalizeAngle(yaw) * 180 / Math.PI).toFixed(1);
  }
  const storedLiftHeight = competitionWaypointLiftHeight(point);
  const hasStoredLiftHeight = storedLiftHeight !== null;
  if ($("#competition-waypoint-auto-lift")) {
    $("#competition-waypoint-auto-lift").checked = point === null || hasStoredLiftHeight;
  }
  if ($("#competition-waypoint-lift-height")) {
    const height = hasStoredLiftHeight
      ? storedLiftHeight
      : COMPETITION_WAYPOINT_DEFAULT_LIFT_MM;
    $("#competition-waypoint-lift-height").value = String(Math.round(height));
  }
  if ($("#competition-waypoint-note")) {
    $("#competition-waypoint-note").value = point?.note || "";
  }
  updateCompetitionWaypointControls();
}

function updateCompetitionWaypointControls() {
  const selected = Boolean(selectedCompetitionWaypoint());
  const busy = state.idealNav.competitionWaypointBusy || state.idealNav.isRunning;
  [
    "#save-competition-waypoint",
    "#competition-waypoint-name",
    "#competition-waypoint-kind",
    "#competition-waypoint-yaw",
    "#competition-waypoint-auto-lift",
    "#competition-waypoint-note",
  ].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = busy;
  });
  const liftHeightInput = $("#competition-waypoint-lift-height");
  if (liftHeightInput) {
    liftHeightInput.disabled = busy || !$("#competition-waypoint-auto-lift")?.checked;
  }
  [
    "#update-competition-waypoint",
    "#delete-competition-waypoint",
    "#plan-to-competition-waypoint",
  ].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = busy || !selected;
  });
  const executeWaypointButton = $("#execute-competition-waypoint");
  if (executeWaypointButton) {
    executeWaypointButton.disabled = busy
      || !selected
      || state.estop.latched !== false;
  }
  const saveButton = $("#save-competition-waypoint");
  if (saveButton) {
    saveButton.textContent = state.idealNav.competitionWaypointBusy ? "保存中" : "保存新点";
  }
  const executeButton = $("#execute-competition-waypoint");
  if (executeButton) {
    executeButton.textContent = state.idealNav.isRunning
      && state.idealNav.executionMode === "competition-waypoint"
      ? "导航 / 调高中"
      : (state.idealNav.isRunning ? "执行中" : "执行到点");
  }
}

function renderCompetitionWaypoints() {
  const root = $("#competition-waypoint-list");
  if (!root) return;
  const points = state.idealNav.competitionWaypoints;
  if (!points.length) {
    root.innerHTML = '<div class="route-empty">点击地图选择位置，填写名称后保存。</div>';
    updateCompetitionWaypointControls();
    return;
  }
  root.innerHTML = points.map((point, index) => {
    const selected = point.id === state.idealNav.selectedCompetitionWaypointId;
    const kind = COMPETITION_WAYPOINT_KIND_LABELS[point.kind] || "自定义";
    const yawDeg = normalizeAngle(Number(point.yaw || 0)) * 180 / Math.PI;
    const liftHeight = competitionWaypointLiftHeight(point);
    const liftLabel = liftHeight !== null
      ? `升降 ${liftHeight}mm`
      : "不自动升降";
    return `
      <button type="button"
              class="competition-waypoint-row${selected ? " is-selected" : ""}"
              data-waypoint-id="${escapeHtml(point.id)}">
        <span class="competition-waypoint-index">${index + 1}</span>
        <span class="competition-waypoint-copy">
          <b>${escapeHtml(point.name)}</b>
          <small>${escapeHtml(kind)} · (${Number(point.x).toFixed(2)}, ${Number(point.y).toFixed(2)}) · ${yawDeg.toFixed(0)}° · ${liftLabel}</small>
        </span>
      </button>
    `;
  }).join("");
  updateCompetitionWaypointControls();
}

function applyCompetitionWaypointPayload(payload, preferredId = undefined) {
  state.idealNav.competitionWaypoints = Array.isArray(payload.points)
    ? payload.points
      .filter((point) => point && typeof point.id === "string")
      .map((point) => ({ ...point }))
    : [];
  state.idealNav.competitionWaypointRevision = Number(payload.revision || 0);
  const currentId = preferredId === undefined
    ? state.idealNav.selectedCompetitionWaypointId
    : preferredId;
  state.idealNav.selectedCompetitionWaypointId = state.idealNav.competitionWaypoints.some(
    (point) => point.id === currentId,
  ) ? currentId : null;
  if (payload.map_image && $("#map-plan-image")) {
    $("#map-plan-image").value = String(payload.map_image);
  }
  renderCompetitionWaypoints();
  drawIdealMap();
}

async function refreshCompetitionWaypoints() {
  const payload = await api("/api/map/waypoints");
  applyCompetitionWaypointPayload(payload);
  return payload;
}

function fallbackCompetitionScene3D() {
  const ids = ["replenish", "delivery", "shelf_a", "shelf_b"];
  return {
    revision: 0,
    map: { width_m: IDEAL_MAP.width, height_m: IDEAL_MAP.height, ceiling_m: 2.4 },
    fixtures: IDEAL_MAP.fixtures.map((fixture, index) => ({
      id: ids[index],
      type: fixture.type,
      label: fixture.label,
      x: fixture.x,
      y: fixture.y,
      size_x_m: fixture.w,
      size_y_m: fixture.h,
      base_z_m: 0,
      height_m: fixture.type === "shelf" ? 1.5 : 0.82,
      shelf_levels: fixture.type === "shelf" ? 5 : 0,
      level_pick_heights_m: fixture.type === "shelf" ? [0.15, 0.45, 0.75, 1.05, 1.35] : [],
      measurement_state: "estimated",
    })),
  };
}

function currentCompetitionScene3D() {
  return state.idealNav.scene3d || fallbackCompetitionScene3D();
}

function selectedScene3DFixture() {
  return (currentCompetitionScene3D().fixtures || []).find(
    (fixture) => fixture.id === state.idealNav.selectedScene3dFixtureId,
  ) || null;
}

function defaultShelfPickHeights(fixture, levelCount = Number(fixture?.shelf_levels || 0)) {
  const levels = Math.max(1, Math.round(Number(levelCount) || 1));
  const baseZ = Number.isFinite(Number(fixture?.base_z_m)) ? Number(fixture.base_z_m) : 0;
  const height = Number.isFinite(Number(fixture?.height_m)) ? Number(fixture.height_m) : 1.5;
  return Array.from({ length: levels }, (_, index) => (
    Number((baseZ + height * (index + 0.5) / levels).toFixed(3))
  ));
}

function shelfPickHeights(fixture, levelCount = Number(fixture?.shelf_levels || 0)) {
  const levels = Math.max(1, Math.round(Number(levelCount) || 1));
  const values = Array.isArray(fixture?.level_pick_heights_m)
    ? fixture.level_pick_heights_m.map(Number)
    : [];
  if (values.length === levels && values.every(Number.isFinite)) return values;
  return defaultShelfPickHeights(fixture, levels);
}

function scene3DPickColumn(point) {
  const match = String(point?.label || point?.id || "").match(/(?:^|-)0*(\d+)$/);
  return match ? Number(match[1]) : 0;
}

function scene3DPickSideName(point) {
  const id = String(point?.id || "");
  if (id.includes("-left-")) return "左面";
  if (id.includes("-other-")) return "另一面";
  return {
    left: "左面",
    right: "右面",
    front: "正面",
    back: "背面",
  }[point?.face] || "货架面";
}

function scene3DPickLabel(point) {
  const location = `${scene3DPickSideName(point)} 第 ${Math.round(Number(point?.row) || 0)} 排 第 ${scene3DPickColumn(point)} 列`;
  const productName = String(point?.product_name || "").trim();
  return productName ? `${productName} · ${location}` : location;
}

function pregraspRecordForPoint(fixtureId, pointId) {
  const records = state.idealNav.pregraspPoses?.records;
  if (!Array.isArray(records)) return null;
  return records.find((record) => (
    record?.fixture_id === fixtureId && record?.point_id === pointId
  )) || null;
}

function pregraspArmLabel(arm) {
  return arm === "right" ? "右臂" : "左臂";
}

function selectedScene3DPregraspRecord() {
  const selected = state.idealNav.selectedScene3dPickPoint;
  if (!selected?.fixtureId || !selected?.point?.id) return null;
  return pregraspRecordForPoint(selected.fixtureId, selected.point.id);
}

function prepareRecordedPregraspTarget() {
  const selected = state.idealNav.selectedScene3dPickPoint;
  const record = selectedScene3DPregraspRecord();
  const fixture = (currentCompetitionScene3D().fixtures || []).find(
    (item) => item.id === selected?.fixtureId,
  );
  if (!selected?.point || !fixture || !record) {
    throw new Error("\u8bf7\u5148\u9009\u62e9\u5df2\u8bb0\u5f55\u9884\u6293\u53d6\u4f4d\u59ff\u7684\u8d27\u4f4d");
  }
  const target = scene3DPickTarget(fixture, selected.point);
  if (!target.recorded || !Number.isFinite(Number(target.x)) || !Number.isFinite(Number(target.y))) {
    throw new Error("\u8be5\u8d27\u4f4d\u7684\u5e95\u76d8\u6807\u5b9a\u4f4d\u7f6e\u4e0d\u5b8c\u6574");
  }
  state.idealNav.finalYaw = Number(target.yaw);
  setIdealTarget({ x: Number(target.x), y: Number(target.y) });
  return { record, target };
}

async function previewRecordedPregrasp() {
  const { record } = prepareRecordedPregraspTarget();
  await previewMapPlan({ syncRoute: true });
  toast(`\u5df2\u9884\u6f14 ${scene3DPickLabel(state.idealNav.selectedScene3dPickPoint.point)} \u7684\u8bb0\u5f55\u505c\u8f66\u70b9`);
  return record;
}

async function executeRecordedPregrasp() {
  const { record } = prepareRecordedPregraspTarget();
  const restoreGripper = Boolean($("#scene3d-pregrasp-restore-gripper")?.checked);
  await executeMapPlanRoute({
    executionMode: "recorded-pregrasp",
    completionLiftHeightMm: Number(record.lift_height_mm),
    pregraspPose: {
      fixtureId: record.fixture_id,
      pointId: record.point_id,
      restoreGripper,
    },
  });
}

function scene3DPickTarget(fixture, point) {
  const recorded = pregraspRecordForPoint(fixture?.id, point?.id);
  const base = recorded?.base_pose;
  if (
    base
    && Number.isFinite(Number(base.x))
    && Number.isFinite(Number(base.y))
    && Number.isFinite(Number(base.yaw))
  ) {
    return {
      x: Number(base.x),
      y: Number(base.y),
      yaw: Number(base.yaw),
      recorded,
    };
  }
  return { ...scene3DPickApproach(fixture, point), recorded: null };
}

function scene3DPickApproach(fixture, point) {
  const sceneMap = currentCompetitionScene3D().map || {};
  const mapWidth = Number(sceneMap.width_m) || IDEAL_MAP.width;
  const mapHeight = Number(sceneMap.height_m) || IDEAL_MAP.height;
  const sizeX = Number(fixture.size_x_m) || 0;
  const sizeY = Number(fixture.size_y_m) || 0;
  const center = clamp(Number(point.u_center) || 0.5, 0, 1);
  // The chassis is 0.90m wide. A 0.65m center-to-shelf offset left only
  // 0.20m at the body edge, which is insufficient for a side arm to turn.
  const standoff = 1.00;
  let x = Number(fixture.x) || 0;
  let y = Number(fixture.y) || 0;
  let yaw = 0;
  if (point.face === "right") {
    x += sizeX + standoff;
    y += sizeY * center;
    // The base stays longitudinal in a narrow aisle; the arm supplies the side reach.
    yaw = Math.PI / 2;
  } else if (point.face === "front") {
    x += sizeX * center;
    y -= standoff;
    yaw = Math.PI / 2;
  } else if (point.face === "back") {
    x += sizeX * center;
    y += sizeY + standoff;
    yaw = -Math.PI / 2;
  } else {
    x -= standoff;
    y += sizeY * center;
    // The base stays longitudinal in a narrow aisle; the arm supplies the side reach.
    yaw = Math.PI / 2;
  }
  // Clamp with the rotated 1.00m x 0.90m footprint, plus one 5cm map-cell margin.
  const halfLong = Number(IDEAL_MAP.robotSizeX) / 2 + 0.05;
  const halfWide = Number(IDEAL_MAP.robotSizeY) / 2 + 0.05;
  const halfWorldX = Math.abs(Math.cos(yaw)) * halfLong + Math.abs(Math.sin(yaw)) * halfWide;
  const halfWorldY = Math.abs(Math.sin(yaw)) * halfLong + Math.abs(Math.cos(yaw)) * halfWide;
  return {
    x: clamp(x, halfWorldX, mapWidth - halfWorldX),
    y: clamp(y, halfWorldY, mapHeight - halfWorldY),
    yaw: normalizeAngle(yaw),
  };
}

function renderScene3DPickPoints(fixture) {
  const detail = $("#scene3d-pick-detail");
  const list = $("#scene3d-pick-list");
  if (!detail || !list) return;
  const points = Array.isArray(fixture?.pick_points) ? fixture.pick_points : [];
  if (!fixture || !points.length) {
    detail.innerHTML = '<div><span>选中货位</span><span>-</span></div>';
    list.innerHTML = "";
    return;
  }
  const selected = state.idealNav.selectedScene3dPickPoint;
  const selectedPoint = selected?.fixtureId === fixture.id
    ? points.find((point) => point.id === selected.point.id)
    : null;
  if (selectedPoint) {
    const approach = scene3DPickTarget(fixture, selectedPoint);
    const pickHeight = Number(selected.pickHeightM);
    const record = approach.recorded;
    detail.innerHTML = [
      ["选中货位", scene3DPickLabel(selectedPoint)],
      ["抓取 Z", `${pickHeight.toFixed(2)} m`],
      ["停车点", `${approach.x.toFixed(2)}, ${approach.y.toFixed(2)}${record ? " · 已标定" : " · 默认"}`],
      ["到点朝向", `${(approach.yaw * 180 / Math.PI).toFixed(0)} deg`],
      ["升降高度", record ? `${record.lift_height_mm} mm · 已标定` : "待实机标定"],
      ["预抓取手臂", record ? `${pregraspArmLabel(record.arm)} · 已记录` : "待记录"],
    ].map(([label, value]) => `<div><span>${label}</span><span>${value}</span></div>`).join("");
  } else {
    detail.innerHTML = '<div><span>选中货位</span><span>在三维图点击圆点</span></div>';
  }
  list.innerHTML = points.map((point) => {
    const selectedClass = selectedPoint?.id === point.id ? " is-selected" : "";
    return `<button type="button" class="scene3d-pick-button${selectedClass}" data-scene3d-pick-id="${escapeHtml(point.id)}">${escapeHtml(scene3DPickLabel(point))}</button>`;
  }).join("");
}

function selectScene3DPickPoint(point) {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能切换抓取货位");
    return;
  }
  const fixture = (currentCompetitionScene3D().fixtures || []).find(
    (item) => item.id === point?.fixture_id,
  );
  if (!fixture || !Array.isArray(fixture.pick_points)) return;
  const source = fixture.pick_points.find((item) => item.id === point.id);
  if (!source) return;
  const levels = Math.max(1, Math.round(Number(fixture.shelf_levels) || 1));
  const level = Math.max(1, Math.min(levels, Math.round(Number(source.level) || 1)));
  const pickHeightM = shelfPickHeights(fixture, levels)[level - 1];
  const approach = scene3DPickTarget(fixture, source);
  const record = approach.recorded;
  state.idealNav.selectedScene3dFixtureId = fixture.id;
  state.idealNav.selectedScene3dPickPoint = {
    fixtureId: fixture.id,
    point: { ...source },
    pickHeightM,
  };
  clearCompetitionWaypointSelection();
  state.idealNav.finalYaw = approach.yaw;
  setIdealTarget(approach);
  $("#competition-waypoint-name").value = `${fixture.label} ${scene3DPickLabel(source)}`;
  $("#competition-waypoint-kind").value = "shelf";
  $("#competition-waypoint-yaw").value = (approach.yaw * 180 / Math.PI).toFixed(0);
  $("#competition-waypoint-auto-lift").checked = Boolean(record);
  if (record) $("#competition-waypoint-lift-height").value = String(record.lift_height_mm);
  $("#competition-waypoint-note").value = record
    ? `${fixture.label} ${scene3DPickLabel(source)}; 已绑定标定停车位、${record.lift_height_mm}mm 升降高度和${pregraspArmLabel(record.arm)}预抓取姿态`
    : `${fixture.label} ${scene3DPickLabel(source)}; 抓取 Z=${pickHeightM.toFixed(2)}m; 升降高度待实机标定`;
  updateCompetitionWaypointControls();
  renderCompetitionScene3D();
  syncIdealMap3D();
  toast(`${fixture.label} ${scene3DPickLabel(source)} 已选中`);
}

function selectScene3DPickPointById(pointId) {
  const fixture = selectedScene3DFixture();
  const point = fixture?.pick_points?.find((item) => item.id === pointId);
  if (fixture && point) selectScene3DPickPoint({ ...point, fixture_id: fixture.id });
}

function renderScene3DLevelHeightInputs(fixture) {
  const container = $("#scene3d-level-heights");
  if (!container) return;
  if (!fixture || fixture.type !== "shelf") {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }
  const levels = Math.max(1, Math.round(Number(fixture.shelf_levels) || 1));
  const pickHeights = shelfPickHeights(fixture, levels);
  container.hidden = false;
  container.innerHTML = pickHeights.map((height, index) => `
    <label>第 ${index + 1} 层 m
      <input class="scene3d-level-height" type="number" min="0.00" max="2.40" step="0.01" value="${height.toFixed(2)}">
    </label>
  `).join("");
}

function renderScene3DReference(reference) {
  const container = $("#scene3d-reference");
  if (!container) return;
  if (!reference || typeof reference !== "object") {
    container.innerHTML = "";
    return;
  }
  const station = reference.station_nominal_dimensions_m || {};
  const expectedEquipment = Array.isArray(reference.expected_equipment)
    ? reference.expected_equipment
    : [];
  const mappedCount = expectedEquipment.filter((item) => item.mapping_status === "mapped").length;
  const unmappedCount = expectedEquipment.length - mappedCount;
  const dimensions = [station.size_x_m, station.size_y_m, station.height_m].every(Number.isFinite)
    ? `${Number(station.size_x_m).toFixed(2)} x ${Number(station.size_y_m).toFixed(2)} x ${Number(station.height_m).toFixed(2)}m`
    : "-";
  container.innerHTML = [
    ["标准台面参考", dimensions],
    ["赛项设备已建模", `${mappedCount} 类`],
    ["待定位建模", `${unmappedCount} 类`],
  ].map(([label, value]) => `<div><span>${label}</span><span>${value}</span></div>`).join("");
}

function updateScene3DControls() {
  const busy = state.idealNav.scene3dBusy || state.idealNav.pregraspCaptureBusy;
  const fixture = selectedScene3DFixture();
  [
    "#scene3d-fixture",
    "#scene3d-measurement-state",
    "#scene3d-size-x",
    "#scene3d-size-y",
    "#scene3d-base-z",
    "#scene3d-height",
    "#scene3d-shelf-levels",
  ].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = busy || !fixture;
  });
  document.querySelectorAll("#scene3d-level-heights input").forEach((control) => {
    control.disabled = busy || !fixture;
  });
  const save = $("#save-scene3d-fixture");
  if (save) {
    save.disabled = busy || !fixture;
    save.textContent = busy ? "保存中" : "保存结构尺寸";
  }
  const capture = $("#capture-scene3d-pregrasp");
  if (capture) {
    capture.disabled = busy || !state.idealNav.selectedScene3dPickPoint;
    capture.textContent = state.idealNav.pregraspCaptureBusy ? "记录中" : "记录当前位姿";
  }
  const arm = $("#scene3d-pregrasp-arm");
  if (arm) arm.disabled = busy || !state.idealNav.selectedScene3dPickPoint;
  const recorded = selectedScene3DPregraspRecord();
  const restoreGripper = $("#scene3d-pregrasp-restore-gripper");
  if (restoreGripper) restoreGripper.disabled = busy || !recorded;
  const previewRecorded = $("#preview-scene3d-pregrasp");
  if (previewRecorded) {
    previewRecorded.disabled = busy || !recorded;
  }
  const executeRecorded = $("#execute-scene3d-pregrasp");
  if (executeRecorded) {
    executeRecorded.disabled = busy
      || state.idealNav.isRunning
      || state.estop.latched !== false
      || !recorded;
  }
}

function renderCompetitionScene3D() {
  const select = $("#scene3d-fixture");
  const summary = $("#scene3d-summary");
  const semanticScene = currentCompetitionScene3D();
  const fixtures = Array.isArray(semanticScene.fixtures) ? semanticScene.fixtures : [];
  if (!fixtures.length) {
    if (select) select.innerHTML = '<option value="">暂无实体</option>';
    if (summary) summary.innerHTML = '<div><span>三维实体</span><span>0</span></div>';
    renderScene3DReference(semanticScene.reference);
    renderScene3DLevelHeightInputs(null);
    renderScene3DPickPoints(null);
    updateScene3DControls();
    return;
  }
  if (!fixtures.some((fixture) => fixture.id === state.idealNav.selectedScene3dFixtureId)) {
    state.idealNav.selectedScene3dFixtureId = fixtures[0].id;
  }
  if (select) {
    select.innerHTML = fixtures.map((fixture) => {
      const measured = fixture.measurement_state === "measured" ? "已实测" : "待实测";
      return `<option value="${escapeHtml(fixture.id)}">${escapeHtml(fixture.label)} · ${measured}</option>`;
    }).join("");
    select.value = state.idealNav.selectedScene3dFixtureId;
  }
  const fixture = selectedScene3DFixture();
  if (!fixture) return;
  $("#scene3d-measurement-state").value = fixture.measurement_state || "estimated";
  $("#scene3d-size-x").value = Number(fixture.size_x_m).toFixed(2);
  $("#scene3d-size-y").value = Number(fixture.size_y_m).toFixed(2);
  $("#scene3d-base-z").value = Number(fixture.base_z_m).toFixed(2);
  $("#scene3d-height").value = Number(fixture.height_m).toFixed(2);
  const shelfLevels = $("#scene3d-shelf-levels");
  if (shelfLevels) {
    shelfLevels.value = String(Number(fixture.shelf_levels || 0));
    shelfLevels.disabled = state.idealNav.scene3dBusy || fixture.type !== "shelf";
  }
  renderScene3DLevelHeightInputs(fixture);
  renderManipulationIntent();
  const measuredCount = fixtures.filter((item) => item.measurement_state === "measured").length;
  const maxHeight = Math.max(...fixtures.map((item) => Number(item.height_m) || 0));
  const pickPointCount = Array.isArray(fixture.pick_points) ? fixture.pick_points.length : 0;
  if (summary) {
    const rows = [
      ["三维实体", `${fixtures.length}`],
      ["已实测", `${measuredCount}`],
      ["待实测", `${fixtures.length - measuredCount}`],
      ["最高实体", `${maxHeight.toFixed(2)}m`],
    ];
    if (fixture.type === "shelf") rows.push(["照片标号点位", `${pickPointCount}`]);
    summary.innerHTML = rows.map(([label, value]) => `<div><span>${label}</span><span>${value}</span></div>`).join("");
  }
  renderScene3DReference(semanticScene.reference);
  renderScene3DPickPoints(fixture);
  updateScene3DControls();
}

function syncIdealMap3D() {
  const viewer = window.RucWoneMap3D;
  if (!viewer) return;
  const planned = state.idealNav.mapPlan?.waypoints;
  const route = Array.isArray(planned) && planned.length
    ? planned
    : [currentPlanningStart(), ...activeRouteWaypoints()];
  const start = currentPlanningStart();
  const first = route[0];
  const routeWithStart = first
    && Math.hypot(Number(first.x) - start.x, Number(first.y) - start.y) < 0.01
    ? route
    : [start, ...route];
  viewer.update({
    scene: currentCompetitionScene3D(),
    pose: state.idealNav.replayPoint || state.idealNav.pose,
    route: routeWithStart,
    waypoints: state.idealNav.competitionWaypoints,
    selectedWaypointId: state.idealNav.selectedCompetitionWaypointId,
    selectedScene3dPickPoint: state.idealNav.selectedScene3dPickPoint,
    onPickPoint: selectScene3DPickPoint,
  });
}

function setIdealMapMode(mode) {
  const nextMode = mode === "3d" ? "3d" : "2d";
  state.idealNav.mapMode = nextMode;
  const show3d = nextMode === "3d";
  const canvas = $("#ideal-map-canvas");
  const scene = $("#ideal-map-3d");
  if (canvas) canvas.hidden = show3d;
  if (scene) scene.hidden = !show3d;
  $("#ideal-map-mode-2d")?.classList.toggle("is-active", !show3d);
  $("#ideal-map-mode-3d")?.classList.toggle("is-active", show3d);
  $("#ideal-map-mode-2d")?.setAttribute("aria-pressed", String(!show3d));
  $("#ideal-map-mode-3d")?.setAttribute("aria-pressed", String(show3d));
  window.RucWoneMap3D?.setVisible(show3d);
  syncIdealMap3D();
  if (show3d && !state.idealNav.scene3d) {
    refreshCompetitionScene3D().catch((err) => toast(err.message));
  }
}

function applyCompetitionScene3DPayload(payload) {
  const scene = payload?.scene;
  if (!scene || !Array.isArray(scene.fixtures)) {
    throw new Error("三维地图数据无效");
  }
  state.idealNav.scene3d = {
    ...scene,
    fixtures: scene.fixtures.map((fixture) => ({ ...fixture })),
  };
  renderCompetitionScene3D();
  syncIdealMap3D();
}

async function refreshCompetitionScene3D() {
  const payload = await api("/api/map/scene-3d");
  applyCompetitionScene3DPayload(payload);
  return payload;
}

function applyPregraspPosePayload(payload) {
  const pregrasp = payload?.pregrasp || payload;
  if (!pregrasp || !Array.isArray(pregrasp.records)) {
    throw new Error("预抓取标定数据无效");
  }
  state.idealNav.pregraspPoses = {
    revision: Number(pregrasp.revision) || 0,
    records: pregrasp.records.map((record) => ({ ...record })),
  };
  renderCompetitionScene3D();
}

async function refreshPregraspPoses() {
  const payload = await api("/api/pregrasp-poses");
  applyPregraspPosePayload(payload);
  return payload;
}

async function captureScene3DPregraspPose() {
  const selected = state.idealNav.selectedScene3dPickPoint;
  if (!selected?.fixtureId || !selected?.point?.id) {
    throw new Error("请先在三维货架上选中一个商品点");
  }
  const arm = $("#scene3d-pregrasp-arm")?.value || "left";
  state.idealNav.pregraspCaptureBusy = true;
  updateScene3DControls();
  try {
    const payload = await api("/api/pregrasp-poses/capture", {
      method: "POST",
      body: JSON.stringify({
        fixture_id: selected.fixtureId,
        point_id: selected.point.id,
        arm,
      }),
    });
    applyPregraspPosePayload(payload);
    selectScene3DPickPoint({ ...selected.point, fixture_id: selected.fixtureId });
    toast(`${scene3DPickLabel(selected.point)} 的预抓取位姿已记录`);
  } finally {
    state.idealNav.pregraspCaptureBusy = false;
    updateScene3DControls();
  }
}

async function saveCompetitionScene3DFixture() {
  const fixture = selectedScene3DFixture();
  if (!fixture) throw new Error("请先选择三维实体");
  const sizeX = Number($("#scene3d-size-x")?.value);
  const sizeY = Number($("#scene3d-size-y")?.value);
  const baseZ = Number($("#scene3d-base-z")?.value);
  const height = Number($("#scene3d-height")?.value);
  const shelfLevels = Number($("#scene3d-shelf-levels")?.value);
  if (![sizeX, sizeY, baseZ, height, shelfLevels].every(Number.isFinite)) {
    throw new Error("三维尺寸必须是数字");
  }
  if (sizeX < 0.05 || sizeY < 0.05 || baseZ < 0 || height < 0.05 || baseZ + height > 2.4) {
    throw new Error("请检查实体尺寸和高度范围");
  }
  if (fixture.type === "shelf" && (!Number.isInteger(shelfLevels) || shelfLevels < 1 || shelfLevels > 8)) {
    throw new Error("货架层数必须在 1 到 8 之间");
  }
  const levelInputs = [...document.querySelectorAll("#scene3d-level-heights input")];
  const levelPickHeights = fixture.type === "shelf"
    ? (levelInputs.length === shelfLevels
      ? levelInputs.map((input) => Number(input.value))
      : defaultShelfPickHeights({ base_z_m: baseZ, height_m: height }, shelfLevels))
    : [];
  if (fixture.type === "shelf" && (
    !levelPickHeights.every(Number.isFinite)
    || levelPickHeights.some((item) => item < baseZ || item > baseZ + height)
    || levelPickHeights.some((item, index) => index > 0 && item <= levelPickHeights[index - 1])
  )) {
    throw new Error("层位抓取中心高度必须由下至上递增，并且位于货架高度内");
  }
  state.idealNav.scene3dBusy = true;
  updateScene3DControls();
  try {
    const payload = await api("/api/map/scene-3d", {
      method: "POST",
      body: JSON.stringify({
        action: "update_fixture",
        fixture: {
          ...fixture,
          size_x_m: sizeX,
          size_y_m: sizeY,
          base_z_m: baseZ,
          height_m: height,
          shelf_levels: fixture.type === "shelf" ? shelfLevels : 0,
          level_pick_heights_m: levelPickHeights,
          measurement_state: $("#scene3d-measurement-state")?.value || "estimated",
        },
      }),
    });
    applyCompetitionScene3DPayload(payload);
    toast("三维结构尺寸已保存");
  } finally {
    state.idealNav.scene3dBusy = false;
    updateScene3DControls();
  }
}

function selectCompetitionWaypoint(pointId) {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能切换比赛点位");
    return;
  }
  const point = state.idealNav.competitionWaypoints.find((item) => item.id === pointId);
  if (!point) {
    toast("点位不存在，请刷新列表");
    return;
  }
  state.idealNav.selectedCompetitionWaypointId = point.id;
  state.idealNav.finalYaw = normalizeAngle(Number(point.yaw || 0));
  syncCompetitionWaypointForm(point);
  setIdealTarget(point);
  renderCompetitionWaypoints();
  updateIdealPathSummary();
  drawIdealMap();
}

function clearCompetitionWaypointSelection() {
  state.idealNav.selectedCompetitionWaypointId = null;
  syncCompetitionWaypointForm(null);
  renderCompetitionWaypoints();
  drawIdealMap();
}

function competitionWaypointFormPoint(includeSelectedId) {
  const name = $("#competition-waypoint-name")?.value.trim() || "";
  if (!name) {
    throw new Error("请填写点位名称");
  }
  const yawDeg = Number($("#competition-waypoint-yaw")?.value);
  if (!Number.isFinite(yawDeg)) {
    throw new Error("点位朝向必须是数字");
  }
  let liftHeightMm = null;
  if ($("#competition-waypoint-auto-lift")?.checked) {
    liftHeightMm = Number($("#competition-waypoint-lift-height")?.value);
    if (
      !Number.isFinite(liftHeightMm)
      || liftHeightMm < COMPETITION_WAYPOINT_MIN_LIFT_MM
      || liftHeightMm > COMPETITION_WAYPOINT_MAX_LIFT_MM
    ) {
      throw new Error(
        `升降高度必须在 ${COMPETITION_WAYPOINT_MIN_LIFT_MM} 到 ${COMPETITION_WAYPOINT_MAX_LIFT_MM} mm 之间`,
      );
    }
    liftHeightMm = Math.round(liftHeightMm);
  }
  const selected = selectedCompetitionWaypoint();
  return {
    ...(includeSelectedId && selected ? { id: selected.id } : {}),
    name,
    kind: $("#competition-waypoint-kind")?.value || "custom",
    x: Number(state.idealNav.target.x),
    y: Number(state.idealNav.target.y),
    yaw_deg: yawDeg,
    lift_height_mm: liftHeightMm,
    note: $("#competition-waypoint-note")?.value.trim() || "",
  };
}

async function saveCompetitionWaypoint(updateSelected = false) {
  if (updateSelected && !selectedCompetitionWaypoint()) {
    toast("请先选择要更新的点位");
    return;
  }
  state.idealNav.competitionWaypointBusy = true;
  updateCompetitionWaypointControls();
  try {
    const payload = await api("/api/map/waypoints", {
      method: "POST",
      body: JSON.stringify({
        action: "upsert",
        map_image: $("#map-plan-image")?.value.trim(),
        point: competitionWaypointFormPoint(updateSelected),
      }),
    });
    applyCompetitionWaypointPayload(payload, payload.selected_id || null);
    const point = selectedCompetitionWaypoint();
    if (point) selectCompetitionWaypoint(point.id);
    toast(updateSelected ? "比赛点位已更新" : "比赛点位已保存");
  } finally {
    state.idealNav.competitionWaypointBusy = false;
    updateCompetitionWaypointControls();
  }
}

async function deleteCompetitionWaypoint() {
  const point = selectedCompetitionWaypoint();
  if (!point) {
    toast("请先选择要删除的点位");
    return;
  }
  if (!window.confirm(`删除比赛点位“${point.name}”？`)) return;
  state.idealNav.competitionWaypointBusy = true;
  updateCompetitionWaypointControls();
  try {
    const payload = await api("/api/map/waypoints", {
      method: "POST",
      body: JSON.stringify({ action: "delete", id: point.id }),
    });
    applyCompetitionWaypointPayload(payload, null);
    syncCompetitionWaypointForm(null);
    toast("比赛点位已删除");
  } finally {
    state.idealNav.competitionWaypointBusy = false;
    updateCompetitionWaypointControls();
  }
}

async function planToCompetitionWaypoint() {
  const point = selectedCompetitionWaypoint();
  if (!point) {
    toast("请先选择比赛点位");
    return;
  }
  selectCompetitionWaypoint(point.id);
  await previewMapPlan({ syncRoute: true });
}

async function executeCompetitionWaypoint() {
  const point = selectedCompetitionWaypoint();
  if (!point) {
    toast("请先选择比赛点位");
    return;
  }
  const liftHeight = competitionWaypointLiftHeight(point);
  const liftAction = liftHeight !== null
    ? `，到达并完成朝向后升降到 ${liftHeight} mm`
    : "，到达后不自动调整升降高度";
  if (!window.confirm(
    `执行“${point.name}”？机器人将导航到 (${Number(point.x).toFixed(2)}, ${Number(point.y).toFixed(2)})，最终朝向 ${(normalizeAngle(Number(point.yaw || 0)) * 180 / Math.PI).toFixed(0)}°${liftAction}。`,
  )) return;
  selectCompetitionWaypoint(point.id);
  state.idealNav.competitionWaypointExecution = {
    id: point.id,
    name: point.name,
    liftHeightMm: liftHeight,
  };
  try {
    await executeMapPlanRoute({
      executionMode: "competition-waypoint",
      completionLiftHeightMm: liftHeight,
    });
  } catch (err) {
    state.idealNav.competitionWaypointExecution = null;
    throw err;
  }
}

function clampPlanningPoint(point) {
  return {
    x: clamp(Number(point?.x ?? IDEAL_MAP.initial.x), 0, IDEAL_MAP.width),
    y: clamp(Number(point?.y ?? IDEAL_MAP.initial.y), 0, IDEAL_MAP.height),
  };
}

function currentPlanningStart() {
  // Always begin planning from the freshest localized chassis pose.
  return clampPlanningPoint(state.idealNav.pose || IDEAL_MAP.initial);
}

function syncPlanningStartInputs() {
  const start = currentPlanningStart();
  if ($("#ideal-start-x")) { $("#ideal-start-x").value = start.x.toFixed(2); $("#ideal-start-x").readOnly = true; }
  if ($("#ideal-start-y")) { $("#ideal-start-y").value = start.y.toFixed(2); $("#ideal-start-y").readOnly = true; }
}

function syncIdealTargetInputs() {
  const target = state.idealNav.target;
  if (!target) return;
  if ($("#ideal-goal-x")) $("#ideal-goal-x").value = Number(target.x).toFixed(2);
  if ($("#ideal-goal-y")) $("#ideal-goal-y").value = Number(target.y).toFixed(2);
}

function currentManipulationIntent() {
  const goal = state.idealNav.target || IDEAL_MAP.initial;
  const scene = currentCompetitionScene3D();
  const selected = state.idealNav.selectedScene3dPickPoint;
  const fixture = selected
    ? (scene.fixtures || []).find((item) => item.id === selected.fixtureId)
    : null;
  const point = selected?.point;
  const armLabel = (arm) => arm === "left" ? "\u5de6\u81c2" : arm === "right" ? "\u53f3\u81c2" : arm === "both" ? "\u53cc\u81c2" : "-";
  if (fixture?.type === "shelf" && point) {
    // A right shelf face is on the robot's left after it parks outside that face.
    const arm = point.face === "right" ? "left" : point.face === "left" ? "right" : "both";
    const centralAisle = Number(goal.x) >= 2.5 && Number(goal.x) <= 3.5 && Number(goal.y) >= 1.85;
    const shelfSide = point.face === "right" ? "\u5de6" : point.face === "left" ? "\u53f3" : "\u6b63\u524d";
    return {
      kind: "shelf", fixtureId: fixture.id, fixtureLabel: fixture.label, arm,
      targetSide: point.face, rotateDeg: 90, centralAisle,
      phase: centralAisle ? "arrive_then_turn" : "arrive_then_face",
      text: centralAisle
        ? `${armLabel(arm)}\u4fdd\u6301\u7eb5\u5411\u7a7f\u8fc7\u4e2d\u592e\u7a84\u9053\uff0c\u5230\u4f4d\u540e\u5411${shelfSide}\u4fa7\u8d27\u67b6\u8f6c 90\u00b0`
        : `${armLabel(arm)}\u5728\u5e95\u76d8\u5230\u4f4d\u540e\u5411\u76ee\u6807\u8d27\u67b6\u8f6c 90\u00b0`,
    };
  }
  const stations = (scene.fixtures || []).filter((item) => item.type === "station");
  let nearest = null;
  for (const station of stations) {
    const cx = Number(station.x) + Number(station.size_x_m) / 2;
    const cy = Number(station.y) + Number(station.size_y_m);
    const distance = Math.hypot(Number(goal.x) - cx, Number(goal.y) - cy);
    if (!nearest || distance < nearest.distance) nearest = { fixture: station, distance };
  }
  if (nearest && nearest.distance <= 1.0) {
    return {
      kind: "station", fixtureId: nearest.fixture.id, fixtureLabel: nearest.fixture.label,
      arm: "both", targetSide: "table", rotateDeg: 90, centralAisle: false,
      phase: "arrive_then_face",
      text: `\u53cc\u81c2\u5728\u5e95\u76d8\u5230\u4f4d\u540e\u671d${nearest.fixture.label}\u8f6c\u5411`,
    };
  }
  return {
    kind: "unknown", fixtureId: null, fixtureLabel: "\u672a\u8bc6\u522b\u76ee\u6807",
    arm: "none", targetSide: "none", rotateDeg: 0, centralAisle: false,
    phase: "none", text: "\u672a\u7ed1\u5b9a 3D \u8d27\u4f4d\u6216\u684c\u524d\u505c\u8f66\u70b9\uff1b\u4ec5\u6267\u884c\u5e95\u76d8\u5bfc\u822a",
  };
}

function renderManipulationIntent() {
  const root = $("#goal-manipulation-summary");
  if (!root) return;
  const intent = currentManipulationIntent();
  const arm = intent.arm === "left" ? "\u5de6\u81c2" : intent.arm === "right" ? "\u53f3\u81c2" : intent.arm === "both" ? "\u53cc\u81c2" : "-";
  root.innerHTML = [
    ["\u76ee\u6807\u5b9e\u4f53", intent.fixtureLabel],
    ["\u673a\u68b0\u81c2", arm],
    ["\u5230\u4f4d\u52a8\u4f5c", intent.text],
    ["\u7a84\u9053\u89c4\u5219", intent.centralAisle ? "\u4ece\u4e2d\u592e\u5165\u53e3\u7eb5\u5411\u8fdb\u5165\uff0c\u5230\u4f4d\u540e\u518d\u8f6c\u81c2" : "\u5e95\u76d8\u5230\u4f4d\u540e\u518d\u8f6c\u81c2"],
  ].map(([key, value]) => `<div><span>${escapeHtml(key)}</span><span>${escapeHtml(value)}</span></div>`).join("");
}

function setPlanningStart(point, options = {}) {
  if (state.idealNav.isRunning && !options.force) {
    toast("导航执行中，不能修改规划起点");
    return;
  }
  state.idealNav.planningStart = clampPlanningPoint(point);
  syncPlanningStartInputs();
  updateIdealPathSummary();
  renderMapPlanResult();
  renderManipulationIntent();
  updateIdealMemorySummary();
  renderIdealRouteList();
  drawIdealMap();
}

function activeRouteWaypoints() {
  return state.idealNav.route;
}

function idealPathDistance(points = activeRouteWaypoints()) {
  let cursor = currentPlanningStart();
  return points.reduce((total, point) => {
    const distance = Math.hypot(point.x - cursor.x, point.y - cursor.y);
    cursor = point;
    return total + distance;
  }, 0);
}

function addIdealRouteWaypoint(point = state.idealNav.target, options = {}) {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能修改路线");
    return;
  }
  if (!state.idealNav.routeActive && !options.replace) {
    toast("请先点击新建路");
    return;
  }
  const waypoint = {
    x: clamp(point.x, 0, IDEAL_MAP.size),
    y: clamp(point.y, 0, IDEAL_MAP.size),
  };
  if (Number.isFinite(point.yaw)) {
    waypoint.yaw = point.yaw;
  }
  if (options.replace) {
    state.idealNav.route = [waypoint];
    state.idealNav.routeActive = true;
  } else {
    state.idealNav.route.push(waypoint);
  }
  state.idealNav.routeMode = options.mode || "normal";
  state.idealNav.finalYaw = Number.isFinite(options.finalYaw) ? options.finalYaw : IDEAL_MAP.initial.yaw;
  state.idealNav.target = { x: waypoint.x, y: waypoint.y };
  $("#ideal-goal-x").value = waypoint.x.toFixed(2);
  $("#ideal-goal-y").value = waypoint.y.toFixed(2);
  state.idealNav.replayPoint = null;
  updateIdealPathSummary();
  renderMapPlanResult();
  renderIdealRouteList();
  drawIdealMap();
}


/* Removed obsolete overridden function. */


function removeLastIdealRouteWaypoint() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能修改路线");
    return;
  }
  state.idealNav.route.pop();
  state.idealNav.routeMode = "normal";
  state.idealNav.routeActive = true;
  updateIdealPathSummary();
  renderManipulationIntent();
  renderIdealRouteList();
  drawIdealMap();
}


/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */


function renderIdealRouteList() {
  const root = $("#ideal-route-list");
  if (!root) return;
  const route = state.idealNav.route;
  if (!route.length) {
    root.innerHTML = state.idealNav.routeActive
      ? `<div class="route-empty">已新建路。点击地图选择候选点，再点新建点加入路径。</div>`
      : `<div class="route-empty">尚未新建路。先点击新建路。</div>`;
    return;
  }
  root.innerHTML = route.map((point, index) => `<div class="route-item">
    <strong>${index + 1}</strong>
    <span>${point.x.toFixed(2)}, ${point.y.toFixed(2)}</span>
  </div>`).join("");
}

function updateIdealPathSummary() {
  const root = $("#ideal-path-summary");
  if (!root) return;
  const waypoints = activeRouteWaypoints();
  const target = waypoints[waypoints.length - 1] || state.idealNav.target;
  const avoidObstacles = currentIdealAvoidObstaclesEnabled();
  const planningStart = currentPlanningStart();
  root.innerHTML = [
    ["起点", `${planningStart.x.toFixed(2)}, ${planningStart.y.toFixed(2)}`],
    ["路状态", state.idealNav.routeActive ? "编辑中" : "未新建"],
    ["路径点数", `${waypoints.length}`],
    ["候选点", `${state.idealNav.target.x.toFixed(2)}, ${state.idealNav.target.y.toFixed(2)}`],
    ["终点", waypoints.length ? `${target.x.toFixed(2)}, ${target.y.toFixed(2)}` : "未设置"],
    ["总距离", `${idealPathDistance(waypoints).toFixed(2)}m`],
    ["最终朝向", `${(state.idealNav.finalYaw * 180 / Math.PI).toFixed(1)}deg`],
    ["前方避障", avoidObstacles ? "开启" : "关闭"],
    ["半径", `${IDEAL_MAP.robotRadius.toFixed(2)}m`],
  ].map(([key, value]) => `<div><span>${key}</span><span>${value}</span></div>`).join("");
}

function updateIdealMemorySummary() {
  const root = $("#ideal-memory-summary");
  if (!root) return;
  const pose = state.idealNav.pose;
  const planningStart = currentPlanningStart();
  const localization = state.idealNav.localization || {};
  const raw = localization.raw_pose;
  root.innerHTML = [
    ["地图 X", pose.x.toFixed(2)],
    ["地图 Y", pose.y.toFixed(2)],
    ["车头朝向", `${robotHeadYawDeg(pose.yaw).toFixed(1)}deg`],
    ["原始里程计映射", raw ? `${Number(raw.x).toFixed(2)}, ${Number(raw.y).toFixed(2)}` : "-"],
    ["定位模式", localization.mode || "未初始化"],
    ["规划起点", `${planningStart.x.toFixed(2)}, ${planningStart.y.toFixed(2)}`],
  ].map(([key, value]) => `<div><span>${key}</span><span>${value}</span></div>`).join("");
}


/* Removed obsolete overridden function. */


function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}


/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */


function resetIdealRecoveryState() {
  state.idealNav.routeJobId = null;
  state.idealNav.executionGoal = null;
  state.idealNav.executionWaypoints = [];
  state.idealNav.encounterPoints = [];
  state.idealNav.recoveryBusy = false;
  state.idealNav.recoveryCount = 0;
  state.idealNav.lastObstacleSampleKey = "";
}

function readMapPlanNumber(selector, label) {
  const value = Number($(selector).value);
  if (!Number.isFinite(value)) {
    throw new Error(`${label} 需要是数字`);
  }
  return value;
}

function currentMapPlanEndpoints() {
  const start = currentPlanningStart();
  const inputX = Number($("#ideal-goal-x")?.value);
  const inputY = Number($("#ideal-goal-y")?.value);
  // Goal inputs are the operator-facing source of truth.  Do not plan using a
  // stale canvas candidate left by an earlier route.
  const goal = Number.isFinite(inputX) && Number.isFinite(inputY)
    ? {
      x: clamp(inputX, IDEAL_MAP.robotHalfX, IDEAL_MAP.width - IDEAL_MAP.robotHalfX),
      y: clamp(inputY, IDEAL_MAP.robotHalfY, IDEAL_MAP.height - IDEAL_MAP.robotHalfY),
    }
    : {
      x: Number(state.idealNav.target.x),
      y: Number(state.idealNav.target.y),
    };
  return {
    start: { x: Number(start.x), y: Number(start.y) },
    goal,
  };
}

function currentIdealAvoidObstaclesEnabled() {
  if ($("#map-plan-avoid-obstacles")) return Boolean($("#map-plan-avoid-obstacles").checked);
  if ($("#ideal-avoid-obstacles")) return Boolean($("#ideal-avoid-obstacles").checked);
  return Boolean(state.idealNav.avoidObstacles);
}

function setIdealAvoidObstaclesEnabled(enabled) {
  const checked = Boolean(enabled);
  state.idealNav.avoidObstacles = checked;
  if ($("#ideal-avoid-obstacles")) $("#ideal-avoid-obstacles").checked = checked;
  if ($("#map-plan-avoid-obstacles")) $("#map-plan-avoid-obstacles").checked = checked;
}

function optionalMapPlanNumber(selector, digits = 2, suffix = "") {
  const value = Number($(selector)?.value);
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "-";
}

function formatMapPlanPoint(point, fallback = null) {
  const x = Number(point?.x ?? fallback?.x);
  const y = Number(point?.y ?? fallback?.y);
  return Number.isFinite(x) && Number.isFinite(y) ? `${x.toFixed(2)}, ${y.toFixed(2)}` : "-";
}

function mapPlanPayload(overrides = {}) {
  const mapImage = $("#map-plan-image").value.trim();
  if (mapImage === "locomotion/maps/rucwone_top_half_6x4.pgm") {
    const resolutionInput = $("#map-plan-resolution");
    if (resolutionInput && Number(resolutionInput.value) !== 0.01) resolutionInput.value = "0.01";
  }
  if (!mapImage) {
    throw new Error("请填写地图文件");
  }
  const resolution = readMapPlanNumber("#map-plan-resolution", "分辨率");
  const robotSizeX = readMapPlanNumber("#map-plan-size-x", "机器人 x 方向长度");
  const robotSizeY = readMapPlanNumber("#map-plan-size-y", "机器人 y 方向宽度");
  const clearance = readMapPlanNumber("#map-plan-clearance", "额外间隙");
  if (resolution <= 0) throw new Error("分辨率必须大于 0");
  if (robotSizeX <= 0) throw new Error("机器人 x 方向长度必须大于 0");
  if (robotSizeY <= 0) throw new Error("机器人 y 方向宽度必须大于 0");
  if (clearance < 0) throw new Error("额外间隙不能小于 0");
  const endpoints = currentMapPlanEndpoints();
  const requestedStart = overrides.start || endpoints.start;
  // The lattice can select a reverse escape heading when the chassis is pinned near a wall.
  const plannerStart = {
    x: Number(requestedStart.x),
    y: Number(requestedStart.y),
  };
  const payload = {
    map_image: mapImage,
    resolution,
    route_policy: "aisle-turn-zones-v2",
    actual_start_yaw: Number(requestedStart?.yaw ?? state.idealNav.pose?.yaw ?? Math.PI / 2),
    start: plannerStart,
    goal: overrides.goal || endpoints.goal,
    robot_radius: Math.max(robotSizeX, robotSizeY) / 2,
    robot_size_x: robotSizeX,
    robot_size_y: robotSizeY,
    navigation_profile: "runtime_stowed",
    arm_motion: IDEAL_MAP.armMotion,
    manipulation_intent: currentManipulationIntent(),
    clearance,
    include_occupancy: overrides.includeOccupancy ?? false,
    snap_to_free: overrides.snapToFree ?? true,
  };
  if (Array.isArray(overrides.temporaryObstacles) && overrides.temporaryObstacles.length) {
    payload.temporary_obstacles = overrides.temporaryObstacles.map((item) => ({
      x: Number(item.x),
      y: Number(item.y),
      radius_m: Number(item.radiusM ?? item.radius_m ?? 0),
    }));
  }
  return payload;
}

function mapPlanBoundaryIssue() {
  const start = currentPlanningStart();
  const robotSizeX = Number($("#map-plan-size-x")?.value);
  const robotSizeY = Number($("#map-plan-size-y")?.value);
  const clearance = Number($("#map-plan-clearance")?.value);
  if (
    !Number.isFinite(start.x)
    || !Number.isFinite(start.y)
    || !Number.isFinite(robotSizeX)
    || !Number.isFinite(robotSizeY)
    || !Number.isFinite(clearance)
  ) {
    return "";
  }

  // Below this margin the rectangular footprint cannot fit at any cardinal heading.
  const minimumHalfExtent = Math.min(robotSizeX, robotSizeY) / 2 + clearance;
  const edges = [];
  if (start.x < minimumHalfExtent) edges.push("左");
  if (start.x > IDEAL_MAP.width - minimumHalfExtent) edges.push("右");
  if (start.y < minimumHalfExtent) edges.push("下");
  if (start.y > IDEAL_MAP.height - minimumHalfExtent) edges.push("上");
  if (!edges.length) return "";

  const confidence = Number(state.idealNav.localization?.confidence);
  const confidenceText = Number.isFinite(confidence)
    ? `；当前定位置信度 ${confidence.toFixed(3)}`
    : "";
  return `当前位置 (${start.x.toFixed(2)}, ${start.y.toFixed(2)}) 进入地图${edges.join("/")}边界的车体中心不可达区`
    + `，${robotSizeX.toFixed(2)}m x ${robotSizeY.toFixed(2)}m 车体中心至少要离边界 ${minimumHalfExtent.toFixed(2)}m`
    + `${confidenceText}。已阻止底盘启动，请先校准地图位置或把机器人移回安全中心区。`;
}

function mapPlanLocalizationIssue() {
  const localization = state.idealNav.localization;
  if (!localization?.initialized) {
    return "小地图定位尚未初始化，已阻止底盘启动。请把机器人对准起点后重置起点。";
  }
  if (localization.running !== true) {
    return "小地图定位进程未运行，已阻止底盘启动。";
  }

  const lidar = localization.sources?.["/scan"];
  const staleSources = [
    ["/scan", lidar],
  ].filter(([, source]) => !source || source.fresh !== true || source.usable === false);
  if (localization.error || staleSources.length) {
    const sourceText = staleSources.map(([name, source]) => {
      const age = Number(source?.age_s);
      return `${name}${Number.isFinite(age) ? ` 已 ${age.toFixed(1)}s 未更新` : " 无新数据"}`;
    }).join("，");
    const detail = sourceText || String(localization.error || "定位数据不可用");
    return `底盘/定位数据已中断：${detail}。当前地图坐标是冻结的旧值，已阻止底盘启动；请先恢复底盘 ROS 网络。`;
  }
  return "";
}

function mapPlanFailureMessage(error) {
  const raw = String(error?.message || error || "路线规划失败");
  if (raw.includes("no obstacle-free path found for the rectangular footprint")) {
    return mapPlanBoundaryIssue()
      || "当前位置、车头朝向与障碍物安全区之间没有可执行路线，已阻止底盘启动。请先核对地图中的车体位置和橙色车头方向。";
  }
  if (raw.includes("start heading is not axis-aligned")) {
    return "当前位置没有足够空间安全调整车头，已阻止底盘启动。请先把机器人移到可转向区域。";
  }
  if (raw.includes("start_yaw is required")) {
    return "当前位置无法安全原地转向，且没有读到可靠车头朝向，已阻止底盘启动。";
  }
  return raw;
}

function renderMapPlanFailure(message) {
  state.idealNav.mapPlan = null;
  renderMapPlanResult(null);
  const result = $("#map-plan-result");
  if (result) {
    result.innerHTML = `<div class="route-empty">${escapeHtml(message)}</div>`;
  }
  drawIdealMap();
}

function fmtMapPlanNumber(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "-";
}

function mapPlanDistance(points = []) {
  if (!Array.isArray(points) || points.length < 2) return 0;
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    const prev = points[index - 1];
    const next = points[index];
    total += Math.hypot(Number(next.x) - Number(prev.x), Number(next.y) - Number(prev.y));
  }
  return total;
}

function renderMapPlanResult(plan = state.idealNav.mapPlan) {
  const summary = $("#map-plan-summary");
  const result = $("#map-plan-result");
  if (!summary || !result) return;
  const endpoints = currentMapPlanEndpoints();
  if (!plan) {
    summary.innerHTML = [
      ["起点", formatMapPlanPoint(endpoints.start)],
      ["终点", formatMapPlanPoint(endpoints.goal)],
      ["地图", escapeHtml($("#map-plan-image")?.value?.trim() || "-")],
      ["分辨率", optionalMapPlanNumber("#map-plan-resolution", 3, "m/px")],
      ["机械臂策略", "运行时收臂"],
      ["收臂通行外形", `${optionalMapPlanNumber("#map-plan-size-x", 2, "m")} x ${optionalMapPlanNumber("#map-plan-size-y", 2, "m")}`],
      ["额外间隙", optionalMapPlanNumber("#map-plan-clearance", 2, "m")],
    ].map(([key, value]) => `<div><span>${key}</span><span>${value}</span></div>`).join("");
    result.innerHTML = `<div class="route-empty">点击规划路线生成关键点</div>`;
    updateMapPlanControls();
    return;
  }
  const map = plan.map || {};
  const waypoints = plan.waypoints || [];
  const mapWidthM = Number(map.width) * Number(map.resolution);
  const mapHeightM = Number(map.height) * Number(map.resolution);
  const robotSizeX = Number($("#map-plan-size-x")?.value);
  const robotSizeY = Number($("#map-plan-size-y")?.value);
  const clearance = Number($("#map-plan-clearance")?.value);
  const armMotion = String(plan.planner?.arm_motion || IDEAL_MAP.armMotion);
  const halfX = (Number.isFinite(robotSizeX) ? robotSizeX : IDEAL_MAP.robotSizeX) / 2;
  const halfY = (Number.isFinite(robotSizeY) ? robotSizeY : IDEAL_MAP.robotSizeY) / 2;
  const turnSafetyRadius = Math.hypot(halfX, halfY) + (Number.isFinite(clearance) ? clearance : 0);
  const startLabel = formatMapPlanPoint(plan.start, endpoints.start);
  const goalLabel = formatMapPlanPoint(plan.goal, endpoints.goal);
  const navigationGoalLabel = formatMapPlanPoint(
    plan.navigation_goal,
    waypoints[waypoints.length - 1],
  );
  const taskTargetLabel = formatMapPlanPoint(plan.task_target, plan.goal || endpoints.goal);
  summary.innerHTML = [
    ["起点", startLabel],
    ["任务目标", taskTargetLabel || goalLabel],
    ["底盘停车点", navigationGoalLabel],
    ["关键点", `${plan.waypoint_count ?? waypoints.length}`],
    ["路线长度", `${fmtMapPlanNumber(mapPlanDistance(waypoints), 2)}m`],
    ["地图范围", `${fmtMapPlanNumber(mapWidthM, 2)}m x ${fmtMapPlanNumber(mapHeightM, 2)}m`],
    ["分辨率", `${fmtMapPlanNumber(map.resolution)}m/px`],
    ["机械臂策略", "与底盘导航独立"],
    ["收臂通行外形", `${fmtMapPlanNumber(robotSizeX, 2)}m x ${fmtMapPlanNumber(robotSizeY, 2)}m`],
    ["转弯安全半径", `${fmtMapPlanNumber(turnSafetyRadius, 2)}m`],
    ["中心不可达", `${plan.occupancy?.center_blocked_cells?.length ?? plan.blocked_cells ?? 0} 格`],
  ].map(([key, value]) => `<div><span>${key}</span><span>${escapeHtml(value)}</span></div>`).join("");

  result.innerHTML = waypoints.length
    ? waypoints.map((point, index) => {
      const yawDeg = Number.isFinite(Number(point.yaw)) ? fmtMapPlanNumber((Number(point.yaw) * 180) / Math.PI, 1) : "-";
      return `<div class="route-item map-plan-point">
        <strong>M${index + 1}</strong>
        <span>
          <b>x ${fmtMapPlanNumber(point.x)}, y ${fmtMapPlanNumber(point.y)}</b>
          <small>朝向 ${yawDeg}deg</small>
        </span>
      </div>`;
    }).join("")
    : `<div class="route-empty">未返回关键点</div>`;
  updateMapPlanControls();
}

function mapPlanRouteState(plan = state.idealNav.mapPlan) {
  const rawPoints = Array.isArray(plan?.waypoints) ? plan.waypoints : [];
  const planStart = plan?.start;
  const firstPoint = rawPoints[0];
  const firstIsStart = Boolean(
    firstPoint
    && Number.isFinite(Number(planStart?.x))
    && Number.isFinite(Number(planStart?.y))
    && Math.hypot(Number(firstPoint.x) - Number(planStart.x), Number(firstPoint.y) - Number(planStart.y)) < 0.02,
  );
  // The planner contract is now explicit: remove a first point only when it is the start pose.
  const plannedPoints = firstIsStart ? rawPoints.slice(1) : rawPoints.slice();
  if (!plannedPoints.length) {
    throw new Error("规划结果里没有可执行的关键点");
  }
  const lastPoint = plannedPoints[plannedPoints.length - 1];
  const finalYaw = Number.isFinite(Number(lastPoint?.yaw))
    ? Number(lastPoint.yaw)
    : Number(state.idealNav.finalYaw);
  return {
    finalYaw,
    waypoints: plannedPoints.map((point, index) => ({
      x: Number(point.x),
      y: Number(point.y),
      ...(typeof point.motion === "string" ? { motion: point.motion } : {}),
      ...(typeof point.turn_zone === "string" ? { turn_zone: point.turn_zone } : {}),
      ...(point.task_target && Number.isFinite(Number(point.task_target.x)) && Number.isFinite(Number(point.task_target.y))
        ? { task_target: { x: Number(point.task_target.x), y: Number(point.task_target.y) } }
        : {}),
      ...(index === plannedPoints.length - 1
        ? { yaw: finalYaw }
        : (Number.isFinite(Number(point.yaw)) ? { yaw: Number(point.yaw) } : {})),
    })),
  };
}

function loadMapPlanIntoIdealRoute(plan = state.idealNav.mapPlan) {
  const routeState = mapPlanRouteState(plan);
  state.idealNav.route = routeState.waypoints.map((point) => ({ ...point }));
  state.idealNav.routeActive = true;
  state.idealNav.routeMode = "normal";
  state.idealNav.finalYaw = routeState.finalYaw;
  const finalPoint = routeState.waypoints[routeState.waypoints.length - 1];
  state.idealNav.target = {
    x: Number(finalPoint.x),
    y: Number(finalPoint.y),
  };
  $("#ideal-goal-x").value = state.idealNav.target.x.toFixed(2);
  $("#ideal-goal-y").value = state.idealNav.target.y.toFixed(2);
  updateIdealPathSummary();
  renderIdealRouteList();
  drawIdealMap();
  return {
    waypoints: routeState.waypoints.map((point) => ({ ...point })),
    finalYaw: routeState.finalYaw,
    finalGoal: {
      x: Number(finalPoint.x),
      y: Number(finalPoint.y),
      yaw: routeState.finalYaw,
    },
  };
}

function drawPlanningStartMarker(ctx, canvas, point) {
  const marker = mapToCanvas(point, canvas);
  ctx.save();
  ctx.fillStyle = "#0f5d8f";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(marker.x, marker.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function updateMapPlanControls() {
  const disableInputs = state.idealNav.mapPlanBusy || state.idealNav.isRunning;
  [
    "#ideal-start-x",
    "#ideal-start-y",
    "#map-plan-image",
    "#map-plan-resolution",
    "#map-plan-size-x",
    "#map-plan-size-y",
    "#map-plan-clearance",
    "#map-plan-avoid-obstacles",
  ].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = disableInputs;
  });
  const runButton = $("#run-map-plan");
  if (runButton) {
    runButton.disabled = disableInputs;
    runButton.textContent = state.idealNav.mapPlanBusy ? "规划中" : "规划路线";
  }
  const executeButton = $("#execute-map-plan");
  if (executeButton) {
    executeButton.disabled = disableInputs;
    executeButton.textContent = state.idealNav.isRunning ? "执行中" : "执行路线";
  }
  const clearButton = $("#clear-map-plan");
  if (clearButton) {
    clearButton.disabled = disableInputs || !state.idealNav.mapPlan;
  }
}

function setMapPlanBusy(isBusy) {
  state.idealNav.mapPlanBusy = isBusy;
  updateMapPlanControls();
}

async function previewMapPlan(options = {}) {
  await refreshIdealState({ force: true, syncPlanningStart: true });
  const localizationIssue = mapPlanLocalizationIssue();
  if (localizationIssue) {
    renderMapPlanFailure(localizationIssue);
    throw new Error(localizationIssue);
  }
  const displayedGoal = currentMapPlanEndpoints().goal;
  state.idealNav.target = { ...displayedGoal };
  setMapPlanBusy(true);
  try {
    const payload = await api("/api/map/plan", {
      method: "POST",
      body: JSON.stringify(mapPlanPayload()),
    });
    state.idealNav.mapLayer = {
      map: payload.map,
      occupancy: payload.occupancy || state.idealNav.mapLayer?.occupancy || null,
    };
    state.idealNav.mapPlan = payload;
    renderMapPlanResult(payload);
    // The 3D renderer has its own scene; update it immediately after a plan.
    syncIdealMap3D();
    if (options.syncRoute) {
      loadMapPlanIntoIdealRoute(payload);
    }
    drawIdealMap();
    if (!options.silent) {
      const snapped = [payload.start?.source, payload.goal?.source].some((source) => String(source || "").includes("snapped"));
      const count = payload.waypoint_count ?? (payload.waypoints || []).length;
      const suffix = options.syncRoute ? "，已同步到共享路线" : "";
      toast(snapped ? `规划完成，已自动吸附到最近自由点：${count} 个关键点${suffix}` : `规划完成：${count} 个关键点${suffix}`);
    }
    return payload;
  } catch (error) {
    const message = mapPlanFailureMessage(error);
    renderMapPlanFailure(message);
    throw new Error(message);
  } finally {
    setMapPlanBusy(false);
  }
}

function clearMapPlanPreview() {
  state.idealNav.mapPlan = null;
  renderMapPlanResult(null);
  drawIdealMap();
}

async function executeMapPlanRoute(options = {}) {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能重复执行");
    return;
  }
  resetIdealRecoveryState();
  renderIdealAvoidanceSummary();
  const plan = await previewMapPlan({ silent: true });
  const routeState = loadMapPlanIntoIdealRoute(plan);
  await startIdealRouteExecution(routeState.waypoints, {
    avoidObstacles: currentIdealAvoidObstaclesEnabled(),
    finalGoal: routeState.finalGoal,
    finalYaw: routeState.finalYaw,
    executionMode: options.executionMode,
    routeOverrides: {
      completionLiftHeightMm: options.completionLiftHeightMm,
      pregraspPose: options.pregraspPose,
    },
  });
}


/* Removed obsolete overridden function. */




function drawTaskZone(ctx, canvas) {
  const zone = IDEAL_MAP.taskZone;
  const center = mapToCanvas(zone, canvas);
  const edge = mapToCanvas({ x: zone.x + zone.radiusM, y: zone.y }, canvas);
  const radius = Math.abs(edge.x - center.x);
  ctx.save();
  ctx.fillStyle = "rgba(250, 204, 21, 0.20)";
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#1f2937";
  ctx.font = "15px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("\u8d77\u70b9/\u7ec8\u70b9", center.x, center.y - 7);
  ctx.fillText("\u4efb\u52a1\u7ed3\u675f\u5224\u5b9a\u533a", center.x, center.y + 14);
  ctx.restore();
}

function drawIdealMapFixtures(ctx, canvas) {
  ctx.save();

  IDEAL_MAP.fixtures.forEach((fixture) => {
    const margin = IDEAL_MAP.robotTurnRadius;
    const unavailable = {
      x: Math.max(0, fixture.x - margin),
      y: Math.max(0, fixture.y - margin),
      w: Math.min(IDEAL_MAP.width, fixture.x + fixture.w + margin) - Math.max(0, fixture.x - margin),
      h: Math.min(IDEAL_MAP.height, fixture.y + fixture.h + margin) - Math.max(0, fixture.y - margin),
    };
    const blockedBox = mapRectToCanvas(unavailable, canvas);
    ctx.fillStyle = "rgba(185, 56, 34, 0.14)";
    ctx.fillRect(blockedBox.x, blockedBox.y, blockedBox.w, blockedBox.h);
    ctx.strokeStyle = "rgba(185, 56, 34, 0.55)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(blockedBox.x, blockedBox.y, blockedBox.w, blockedBox.h);
    ctx.setLineDash([]);

    const box = mapRectToCanvas(fixture, canvas);
    if (fixture.type === "shelf") {
      ctx.fillStyle = "#7f1d1d";
      ctx.fillRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = "#1f2937";
      ctx.fillRect(box.x + box.w * 0.45, box.y, box.w * 0.10, box.h);
      const colors = ["#eab308", "#14b8a6", "#3b82f6", "#ef4444", "#f97316", "#22c55e"];
      const columns = 4;
      const rows = 10;
      const cellW = box.w * 0.16;
      const cellH = box.h * 0.07;
      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < columns; col += 1) {
          const x = box.x + box.w * 0.08 + col * box.w * 0.21;
          const y = box.y + box.h * 0.05 + row * box.h * 0.092;
          ctx.fillStyle = colors[(row + col) % colors.length];
          ctx.fillRect(x, y, cellW, cellH);
        }
      }
    } else {
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(box.x, box.y, box.w, box.h);
      ctx.strokeStyle = "#c7a17a";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(box.x + box.w * 0.10, box.y + box.h * 0.28);
      ctx.lineTo(box.x + box.w * 0.38, box.y + box.h * 0.42);
      ctx.lineTo(box.x + box.w * 0.10, box.y + box.h * 0.64);
      ctx.stroke();
    }
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = "#1f2937";
    ctx.font = "13px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(fixture.label, box.x + box.w / 2, box.y + box.h / 2 + 5);
  });

  const dividerStart = mapToCanvas({ x: 0, y: IDEAL_MAP.height }, canvas);
  const dividerEnd = mapToCanvas({ x: IDEAL_MAP.width, y: IDEAL_MAP.height }, canvas);
  ctx.strokeStyle = "#f2c94c";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(dividerStart.x, dividerStart.y);
  ctx.lineTo(dividerEnd.x, dividerEnd.y);
  ctx.stroke();
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(dividerStart.x, dividerStart.y);
  ctx.lineTo(dividerEnd.x, dividerEnd.y);
  ctx.stroke();
  ctx.restore();
}

function drawSmallMapLocalization(ctx, canvas) {
  const localization = state.idealNav.localization;
  if (!localization) return;
  const corrected = localization.pose;
  const raw = localization.raw_pose;
  const history = Array.isArray(localization.history) ? localization.history : [];

  ctx.save();
  if (history.length > 1) {
    ctx.strokeStyle = "rgba(20, 122, 76, 0.65)";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    history.forEach((item, index) => {
      const pose = item?.pose;
      if (!pose) return;
      const point = mapToCanvas(pose, canvas);
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  if (raw && corrected) {
    const rawPoint = mapToCanvas(raw, canvas);
    const correctedPoint = mapToCanvas(corrected, canvas);
    const distance = Math.hypot(
      Number(corrected.x) - Number(raw.x),
      Number(corrected.y) - Number(raw.y),
    );
    ctx.fillStyle = "#667085";
    ctx.beginPath();
    ctx.arc(rawPoint.x, rawPoint.y, 5, 0, Math.PI * 2);
    ctx.fill();
    if (distance > 0.002) {
      ctx.strokeStyle = "#147a4c";
      ctx.fillStyle = "#147a4c";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(rawPoint.x, rawPoint.y);
      ctx.lineTo(correctedPoint.x, correctedPoint.y);
      ctx.stroke();
      const angle = Math.atan2(
        correctedPoint.y - rawPoint.y,
        correctedPoint.x - rawPoint.x,
      );
      ctx.beginPath();
      ctx.moveTo(correctedPoint.x, correctedPoint.y);
      ctx.lineTo(
        correctedPoint.x - 10 * Math.cos(angle - Math.PI / 6),
        correctedPoint.y - 10 * Math.sin(angle - Math.PI / 6),
      );
      ctx.lineTo(
        correctedPoint.x - 10 * Math.cos(angle + Math.PI / 6),
        correctedPoint.y - 10 * Math.sin(angle + Math.PI / 6),
      );
      ctx.closePath();
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawCompetitionWaypoints(ctx, canvas) {
  const selectedId = state.idealNav.selectedCompetitionWaypointId;
  ctx.save();
  state.idealNav.competitionWaypoints.forEach((point, index) => {
    const center = mapToCanvas(point, canvas);
    const color = COMPETITION_WAYPOINT_KIND_COLORS[point.kind]
      || COMPETITION_WAYPOINT_KIND_COLORS.custom;
    const selected = point.id === selectedId;
    const yaw = normalizeAngle(Number(point.yaw || 0));
    const arrowLength = selected ? 28 : 22;

    if (selected) {
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(center.x, center.y, 13, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), center.x, center.y + 0.5);

    const arrowX = center.x + Math.cos(yaw) * arrowLength;
    const arrowY = center.y - Math.sin(yaw) * arrowLength;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(arrowX, arrowY);
    ctx.stroke();
    const arrowAngle = Math.atan2(arrowY - center.y, arrowX - center.x);
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - 8 * Math.cos(arrowAngle - Math.PI / 6),
      arrowY - 8 * Math.sin(arrowAngle - Math.PI / 6),
    );
    ctx.lineTo(
      arrowX - 8 * Math.cos(arrowAngle + Math.PI / 6),
      arrowY - 8 * Math.sin(arrowAngle + Math.PI / 6),
    );
    ctx.closePath();
    ctx.fill();

    const label = String(point.name || `点位 ${index + 1}`).slice(0, 16);
    const liftHeight = competitionWaypointLiftHeight(point);
    const heightLabel = liftHeight === null ? "H --" : `H ${liftHeight}mm`;
    ctx.font = "12px system-ui";
    ctx.textBaseline = "alphabetic";
    const nameWidth = ctx.measureText(label).width;
    ctx.font = "11px system-ui";
    const heightWidth = ctx.measureText(heightLabel).width;
    const labelWidth = Math.ceil(Math.max(nameWidth, heightWidth)) + 12;
    let labelX = center.x + 14;
    if (labelX + labelWidth > canvas.width - 8) {
      labelX = center.x - labelWidth - 14;
    }
    const labelY = center.y - 22;
    ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
    ctx.fillRect(labelX, labelY - 15, labelWidth, 34);
    ctx.strokeStyle = selected ? "#111827" : color;
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeRect(labelX, labelY - 15, labelWidth, 34);
    ctx.fillStyle = "#111827";
    ctx.textAlign = "left";
    ctx.font = "12px system-ui";
    ctx.fillText(label, labelX + 6, labelY - 1);
    ctx.fillStyle = "#667085";
    ctx.font = "11px system-ui";
    ctx.fillText(heightLabel, labelX + 6, labelY + 13);
  });
  ctx.restore();
}

function competitionWaypointAtCanvasPosition(clientX, clientY, canvas) {
  const rect = canvas.getBoundingClientRect();
  const canvasX = (clientX - rect.left) * canvas.width / rect.width;
  const canvasY = (clientY - rect.top) * canvas.height / rect.height;
  let closest = null;
  let closestDistance = 20;
  state.idealNav.competitionWaypoints.forEach((point) => {
    const marker = mapToCanvas(point, canvas);
    const distance = Math.hypot(marker.x - canvasX, marker.y - canvasY);
    if (distance < closestDistance) {
      closest = point;
      closestDistance = distance;
    }
  });
  return closest;
}

function handleIdealMapClick(event) {
  const marker = competitionWaypointAtCanvasPosition(
    event.clientX,
    event.clientY,
    event.currentTarget,
  );
  if (marker) {
    selectCompetitionWaypoint(marker.id);
    return;
  }
  clearCompetitionWaypointSelection();
  setIdealTarget(canvasToMap(event.clientX, event.clientY, event.currentTarget));
}

function drawIdealMap(pointOverride = null) {
  const canvas = $("#ideal-map-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { pad, originX, originY, mapWidth, mapHeight } = idealCanvasMetrics(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(originX, originY, mapWidth, mapHeight);
  ctx.strokeStyle = "#1f2937";
  ctx.lineWidth = 2;
  ctx.strokeRect(originX, originY, mapWidth, mapHeight);

  const safeMin = mapToCanvas({ x: IDEAL_MAP.robotHalfX, y: IDEAL_MAP.robotHalfY }, canvas);
  const safeMax = mapToCanvas({ x: IDEAL_MAP.width - IDEAL_MAP.robotHalfX, y: IDEAL_MAP.height - IDEAL_MAP.robotHalfY }, canvas);
  ctx.fillStyle = "rgba(38, 131, 79, 0.08)";
  ctx.fillRect(safeMin.x, safeMax.y, safeMax.x - safeMin.x, safeMin.y - safeMax.y);
  ctx.setLineDash([8, 6]);
  ctx.strokeStyle = "#8ac9a5";
  ctx.lineWidth = 2;
  ctx.strokeRect(safeMin.x, safeMax.y, safeMax.x - safeMin.x, safeMin.y - safeMax.y);
  ctx.setLineDash([]);

  ctx.strokeStyle = "#d7dce2";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#667085";
  ctx.font = "12px system-ui";
  for (let i = 0; i <= IDEAL_MAP.width; i += 1) {
    const a = mapToCanvas({ x: i, y: 0 }, canvas);
    const b = mapToCanvas({ x: i, y: IDEAL_MAP.height }, canvas);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.fillText(`${i}`, a.x - 4, originY + mapHeight + 22);
  }
  for (let i = 0; i <= IDEAL_MAP.height; i += 1) {
    const c = mapToCanvas({ x: 0, y: i }, canvas);
    const d = mapToCanvas({ x: IDEAL_MAP.width, y: i }, canvas);
    ctx.beginPath();
    ctx.moveTo(c.x, c.y);
    ctx.lineTo(d.x, d.y);
    ctx.stroke();
    ctx.fillText(`${i}`, originX - 26, c.y + 4);
  }

  drawIdealMapFixtures(ctx, canvas);
  drawTaskZone(ctx, canvas);
  drawCompetitionWaypoints(ctx, canvas);

  const route = activeRouteWaypoints();
  const planningStart = currentPlanningStart();
  const start = mapToCanvas(planningStart, canvas);
  const showSharedRoute = route.length && !(state.idealNav.mapPlan?.waypoints?.length);
  if (showSharedRoute) {
    ctx.strokeStyle = "#1769aa";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    route.forEach((point) => {
      const next = mapToCanvas(point, canvas);
      ctx.lineTo(next.x, next.y);
    });
    ctx.stroke();
  }

  drawMapPlanPreview(ctx, canvas, state.idealNav.mapLayer, state.idealNav.mapPlan);
  state.idealNav.encounterPoints.forEach((point, index) => {
    drawEncounterPoint(ctx, canvas, point, index + 1);
  });

  drawSmallMapLocalization(ctx, canvas);
  drawPlanningStartMarker(ctx, canvas, planningStart);
  drawRobot(ctx, canvas, state.idealNav.pose, "#1769aa", "实时定位");
  if (showSharedRoute) {
    route.forEach((point, index) => {
      drawWaypoint(ctx, mapToCanvas(point, canvas), index + 1, state.idealNav.routeMode === "reset" ? "#b7791f" : "#1769aa");
    });
    const goal = mapToCanvas(route[route.length - 1], canvas);
    drawTarget(ctx, goal, state.idealNav.routeMode === "reset" ? "#b7791f" : "#b93822");
  }
  drawCandidate(ctx, mapToCanvas(state.idealNav.target, canvas));

  const replayPoint = pointOverride || state.idealNav.replayPoint;
  if (replayPoint) {
    drawRobot(ctx, canvas, replayPoint, "#b7791f", "执行轨迹");
  }
  drawPerceptionCorrection(ctx, canvas, replayPoint || state.idealNav.pose, state.idealNav.latestCorrection);
  syncIdealMap3D();
}

function drawMapPlanPreview(ctx, canvas, mapLayer, plan) {
  if (!mapLayer && !plan) return;
  ctx.save();
  const layer = mapLayer || plan;
  const map = layer.map || {};
  const origin = map.origin || {};
  const mapWidthM = Number(map.width) * Number(map.resolution);
  const mapHeightM = Number(map.height) * Number(map.resolution);
  if (
    Number.isFinite(mapWidthM)
    && Number.isFinite(mapHeightM)
    && Number.isFinite(Number(origin.x ?? 0))
    && Number.isFinite(Number(origin.y ?? 0))
    && Math.abs(Number(origin.yaw ?? 0)) < 1e-6
  ) {
    const min = mapToCanvas({ x: Number(origin.x ?? 0), y: Number(origin.y ?? 0) }, canvas);
    const max = mapToCanvas({ x: Number(origin.x ?? 0) + mapWidthM, y: Number(origin.y ?? 0) + mapHeightM }, canvas);
    ctx.fillStyle = "rgba(38, 131, 79, 0.06)";
    ctx.strokeStyle = "rgba(38, 131, 79, 0.55)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    ctx.fillRect(min.x, max.y, max.x - min.x, min.y - max.y);
    ctx.strokeRect(min.x, max.y, max.x - min.x, min.y - max.y);
    ctx.setLineDash([]);
  }

  drawMapPlanObstacles(ctx, canvas, layer);

  const points = (plan?.waypoints || []).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!points.length) {
    ctx.restore();
    return;
  }

  ctx.strokeStyle = "#26834f";
  ctx.lineWidth = 4;
  ctx.setLineDash([10, 5]);
  ctx.beginPath();
  points.forEach((point, index) => {
    const canvasPoint = mapToCanvas(point, canvas);
    if (index === 0) ctx.moveTo(canvasPoint.x, canvasPoint.y);
    else ctx.lineTo(canvasPoint.x, canvasPoint.y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  points.forEach((point, index) => {
    const canvasPoint = mapToCanvas(point, canvas);
    const isGoal = index === points.length - 1;
    ctx.fillStyle = isGoal ? "#b93822" : "#26834f";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(canvasPoint.x, canvasPoint.y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1f2937";
    ctx.font = "12px system-ui";
    ctx.fillText(`M${index + 1}`, canvasPoint.x + 11, canvasPoint.y - 8);
  });
  ctx.restore();
}

function drawMapPlanObstacles(ctx, canvas, plan) {
  const map = plan?.map || {};
  const origin = map.origin || {};
  const cells = plan?.occupancy?.cells || [];
  const centerCells = plan?.occupancy?.center_unreachable_cells || [];
  const resolution = Number(map.resolution);
  const height = Number(map.height);
  const originX = Number(origin.x ?? 0);
  const originY = Number(origin.y ?? 0);
  const yaw = Number(origin.yaw ?? 0);
  if (
    (!cells.length && !centerCells.length)
    || !Number.isFinite(resolution)
    || !Number.isFinite(height)
    || !Number.isFinite(originX)
    || !Number.isFinite(originY)
    || Math.abs(yaw) >= 1e-6
  ) {
    return;
  }

  const fillCells = (items, color) => {
    ctx.fillStyle = color;
    items.forEach((cell) => {
      const x = Number(cell.x);
      const y = Number(cell.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const low = mapToCanvas({
        x: originX + x * resolution,
        y: originY + (height - y - 1) * resolution,
      }, canvas);
      const high = mapToCanvas({
        x: originX + (x + 1) * resolution,
        y: originY + (height - y) * resolution,
      }, canvas);
      ctx.fillRect(low.x, high.y, high.x - low.x, low.y - high.y);
    });
  };
  fillCells(centerCells, "rgba(185, 56, 34, 0.18)");
  fillCells(cells, "rgba(31, 35, 40, 0.72)");
}

function drawRobot(ctx, canvas, point, color, label) {
  const center = mapToCanvas(point, canvas);
  const xEdge = mapToCanvas({ x: point.x + IDEAL_MAP.robotHalfX, y: point.y }, canvas);
  const yEdge = mapToCanvas({ x: point.x, y: point.y + IDEAL_MAP.robotHalfY }, canvas);
  const halfW = Math.abs(xEdge.x - center.x);
  const halfH = Math.abs(yEdge.y - center.y);
  const noseDepth = Math.min(20, Math.max(10, halfW * 0.30));
  const noseHalfHeight = Math.min(15, Math.max(8, halfH * 0.32));
  const yaw = Number.isFinite(point.yaw) ? point.yaw : ((point.yawDeg || 0) * Math.PI) / 180;
  const headYaw = robotHeadYaw(yaw);
  ctx.save();
  ctx.translate(center.x, center.y);
  // The chassis localization +x axis points toward the camera-facing rear.
  ctx.rotate(-headYaw);
  ctx.fillStyle = `${color}22`;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.fillRect(-halfW, -halfH, halfW * 2, halfH * 2);
  ctx.strokeRect(-halfW, -halfH, halfW * 2, halfH * 2);

  // The orange edge and arrow mark the physical camera-facing front.
  ctx.strokeStyle = "#f59e0b";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(halfW, -halfH + 3);
  ctx.lineTo(halfW, halfH - 3);
  ctx.stroke();

  ctx.fillStyle = "#f59e0b";
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(halfW + 7, 0);
  ctx.lineTo(halfW - noseDepth, -noseHalfHeight);
  ctx.lineTo(halfW - noseDepth, noseHalfHeight);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = color;
  ctx.font = "13px system-ui";
  ctx.fillText(label, center.x + 10, center.y - 10);
}

function drawTarget(ctx, point, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(point.x - 14, point.y);
  ctx.lineTo(point.x + 14, point.y);
  ctx.moveTo(point.x, point.y - 14);
  ctx.lineTo(point.x, point.y + 14);
  ctx.stroke();
}

function drawCandidate(ctx, point) {
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = "#b93822";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#b93822";
  ctx.font = "12px system-ui";
  ctx.fillText("候选点", point.x + 12, point.y + 4);
}


function formatSigned(value, digits = 3, suffix = "") {
  if (!Number.isFinite(Number(value))) return "-";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}${suffix}`;
}

function correctionFromSample(record, sample) {
  if (!sample || sample.perception_confidence === undefined) return null;
  const pose = sampleToMap(record, sample);
  return {
    pose,
    source: sample.perception_source || sample.obstacle_scan_topic || "-",
    confidence: Number(sample.perception_confidence),
    lateralError: Number(sample.perception_lateral_error_m),
    yawError: Number(sample.perception_yaw_error_rad),
    cmdYaw: Number(sample.correction_cmd_yaw || 0),
    pairedWalls: sample.perception_paired_walls === true,
    leftDistance: Number(sample.perception_left_distance_m),
    rightDistance: Number(sample.perception_right_distance_m),
    ageS: Number(sample.perception_age_s),
    t: Number(sample.t || 0),
  };
}

function correctionFromReplayPoint(point) {
  if (!point || point.perception_confidence === undefined) return null;
  return {
    pose: point,
    source: point.perception_source || "-",
    confidence: Number(point.perception_confidence),
    lateralError: Number(point.perception_lateral_error_m),
    yawError: Number(point.perception_yaw_error_rad),
    cmdYaw: Number(point.correction_cmd_yaw || 0),
    pairedWalls: point.perception_paired_walls === true,
  };
}

function drawArrowHead(ctx, x, y, angle, size = 8) {
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - Math.cos(angle - 0.55) * size, y + Math.sin(angle - 0.55) * size);
  ctx.lineTo(x - Math.cos(angle + 0.55) * size, y + Math.sin(angle + 0.55) * size);
  ctx.closePath();
  ctx.fill();
}

function drawPerceptionCorrection(ctx, canvas, pose, correction) {
  if (!correction || !pose) return;
  const confidence = Number(correction.confidence);
  const cmdYaw = Number(correction.cmdYaw);
  const yawError = Number(correction.yawError);
  const lateral = Number(correction.lateralError);
  const pairedWalls = correction.pairedWalls === true;
  const active = pairedWalls && Number.isFinite(confidence) && confidence >= 0.45 && Math.abs(cmdYaw) > 1e-4;

  ctx.save();
  const panelX = canvas.width - 246;
  const panelY = 18;
  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  ctx.strokeStyle = active ? "#7c3aed" : "#98a2b3";
  ctx.lineWidth = 2;
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") ctx.roundRect(panelX, panelY, 224, 130, 8);
  else ctx.rect(panelX, panelY, 224, 130);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = active ? "#5b21b6" : "#667085";
  ctx.font = "13px system-ui";
  [
    `感知源 ${correction.source || "-"}`,
    `置信度 ${Number.isFinite(confidence) ? confidence.toFixed(2) : "-"}`,
    `横向偏差 ${formatSigned(lateral, 3, "m")}`,
    `航向偏差 ${Number.isFinite(yawError) ? formatSigned(yawError * 180 / Math.PI, 1, "deg") : "-"}`,
    `角速度修正 ${formatSigned(cmdYaw, 3, "rad/s")}`,
  ].forEach((row, index) => ctx.fillText(row, panelX + 12, panelY + 22 + index * 18));
  ctx.fillStyle = pairedWalls ? "#147a4c" : "#b54708";
  ctx.fillText(
    pairedWalls ? "双侧墙已确认，可参与纠偏" : "单侧墙观测：仅诊断，不转向",
    panelX + 12,
    panelY + 112,
  );

  const center = mapToCanvas(pose, canvas);
  if (active) {
    const yaw = Number.isFinite(pose.yaw) ? pose.yaw : 0;
    const sign = cmdYaw >= 0 ? 1 : -1;
    const radius = 34;
    const start = yaw - sign * 0.35;
    const end = yaw + sign * 0.85;
    ctx.strokeStyle = "#7c3aed";
    ctx.fillStyle = "#7c3aed";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(center.x, center.y, radius, -start, -end, sign < 0);
    ctx.stroke();
    const arrowX = center.x + Math.cos(end) * radius;
    const arrowY = center.y - Math.sin(end) * radius;
    drawArrowHead(ctx, arrowX, arrowY, -end + (sign > 0 ? 0.8 : -0.8), 9);
  }
  if (Number.isFinite(lateral) && Math.abs(lateral) > 0.005) {
    const yaw = Number.isFinite(pose.yaw) ? pose.yaw : 0;
    const normal = yaw + Math.PI / 2;
    const length = Math.min(0.35, Math.abs(lateral) * 4);
    const endMap = {
      x: pose.x - Math.sign(lateral) * Math.cos(normal) * length,
      y: pose.y - Math.sign(lateral) * Math.sin(normal) * length,
    };
    const end = mapToCanvas(endMap, canvas);
    ctx.strokeStyle = "#a855f7";
    ctx.fillStyle = "#a855f7";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    drawArrowHead(ctx, end.x, end.y, Math.atan2(center.y - end.y, end.x - center.x), 8);
  }
  ctx.restore();
}

function drawWaypoint(ctx, point, index, color) {
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(index), point.x, point.y);
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function drawEncounterPoint(ctx, canvas, point, index) {
  if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) return;
  const center = mapToCanvas(point, canvas);
  const radiusPoint = mapToCanvas({ x: point.x + (point.radiusM || 0.1), y: point.y }, canvas);
  const radius = Math.max(8, Math.abs(radiusPoint.x - center.x));
  ctx.save();
  ctx.strokeStyle = "#b93822";
  ctx.fillStyle = "rgba(185, 56, 34, 0.12)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(center.x - 8, center.y - 8);
  ctx.lineTo(center.x + 8, center.y + 8);
  ctx.moveTo(center.x - 8, center.y + 8);
  ctx.lineTo(center.x + 8, center.y - 8);
  ctx.stroke();
  ctx.fillStyle = "#b93822";
  ctx.font = "12px system-ui";
  ctx.fillText(`O${index}`, center.x + 10, center.y + 4);
  ctx.restore();
}

function sampleToMap(record, sample) {
  const startOdom = record.start_odom || {};
  const initial = record.request?.initial || IDEAL_MAP.initial;
  const yawOffset = (startOdom.yaw || 0) - (initial.yaw || 0);
  const dx = sample.x - (startOdom.x || 0);
  const dy = sample.y - (startOdom.y || 0);
  const cosYaw = Math.cos(-yawOffset);
  const sinYaw = Math.sin(-yawOffset);
  return {
    x: initial.x + cosYaw * dx - sinYaw * dy,
    y: initial.y + sinYaw * dx + cosYaw * dy,
    yaw: (sample.yaw || 0) - yawOffset,
    perception_source: sample.perception_source,
    perception_confidence: sample.perception_confidence,
    perception_lateral_error_m: sample.perception_lateral_error_m,
    perception_yaw_error_rad: sample.perception_yaw_error_rad,
    correction_cmd_yaw: sample.correction_cmd_yaw,
  };
}

function replayPoints(points, intervalMs = 90) {
  window.clearInterval(state.idealNav.replayTimer);
  let index = 0;
  state.idealNav.replayPoint = points[0] || null;
  state.idealNav.latestCorrection = correctionFromReplayPoint(state.idealNav.replayPoint) || state.idealNav.latestCorrection;
  drawIdealMap();
  state.idealNav.replayTimer = window.setInterval(() => {
    if (index >= points.length) {
      window.clearInterval(state.idealNav.replayTimer);
      state.idealNav.replayTimer = null;
      return;
    }
    state.idealNav.replayPoint = points[index];
    state.idealNav.latestCorrection = correctionFromReplayPoint(state.idealNav.replayPoint) || state.idealNav.latestCorrection;
    drawIdealMap();
    index += 1;
  }, intervalMs);
}

function replayCurrentPath() {
  if (!state.idealNav.route.length) {
    toast("请先新建点加入路径");
    return;
  }
  const points = [];
  let cursor = state.idealNav.pose;
  activeRouteWaypoints().forEach((target) => {
    const steps = Math.max(8, Math.ceil(Math.hypot(target.x - cursor.x, target.y - cursor.y) / 0.02));
    for (let i = 0; i <= steps; i += 1) {
      const ratio = i / steps;
      points.push({
        x: cursor.x + (target.x - cursor.x) * ratio,
        y: cursor.y + (target.y - cursor.y) * ratio,
        yaw: Number.isFinite(target.yaw) ? target.yaw : state.idealNav.finalYaw,
      });
    }
    cursor = target;
  });
  replayPoints(points, 40);
}

async function refreshIdealRuns() {
  const select = $("#ideal-run-select");
  if (!select) return;
  const payload = await api("/api/ideal-nav/runs");
  state.idealNav.runs = payload.runs;
  select.innerHTML = payload.runs.length
    ? payload.runs.map((run) => `<option value="${escapeHtml(run.id)}">${escapeHtml(fmtTime(run.created_at))} ${escapeHtml(idealStatusLabel(run.status))} (${escapeHtml(run.sample_count)}点)</option>`).join("")
    : `<option value="">暂无记录</option>`;
  if (state.idealNav.pendingRunId && payload.runs.some((run) => run.id === state.idealNav.pendingRunId)) {
    select.value = state.idealNav.pendingRunId;
    state.idealNav.pendingRunId = null;
    await loadIdealRun(select.value);
  }
}

async function refreshIdealState(options = {}) {
  const payload = await api("/api/ideal-nav/state");
  const pose = payload.state?.pose || {};
  state.idealNav.localization = payload.state?.localization || null;
  state.idealNav.pose = {
    x: Number(pose.x ?? IDEAL_MAP.initial.x),
    y: Number(pose.y ?? IDEAL_MAP.initial.y),
    yaw: Number(pose.yaw ?? IDEAL_MAP.initial.yaw),
  };
  if (
    !state.idealNav.mapPlanBusy
    && (options.syncPlanningStart === true || !state.idealNav.routeActive)
  ) {
    state.idealNav.planningStart = clampPlanningPoint(state.idealNav.pose);
  }
  syncPlanningStartInputs();
  updateIdealPathSummary();
  renderMapPlanResult();
  updateIdealMemorySummary();
  renderIdealAbsoluteSummary();
  updateIdealExecutionControls();
  renderIdealRouteList();
  drawIdealMap();
}

async function resetIdealPose() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能生成复位路线");
    return;
  }
  buildResetRoute();
}

function shiftPoint(point, dx, dy) {
  return {
    ...point,
    x: clamp(point.x + dx, 0, IDEAL_MAP.size),
    y: clamp(point.y + dy, 0, IDEAL_MAP.size),
  };
}


/* Removed obsolete overridden function. */


async function autoLocateIdealStart() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能自动定位");
    return;
  }
  const button = $("#auto-ideal-localize");
  if (button) button.disabled = true;
  toast("正在用激光雷达扫描全图定位；机器人不会移动");
  try {
    const payload = await api("/api/ideal-nav/auto-localize", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const pose = payload.state?.pose || {};
    const newPose = {
      x: Number(pose.x ?? IDEAL_MAP.initial.x),
      y: Number(pose.y ?? IDEAL_MAP.initial.y),
      yaw: Number(pose.yaw ?? IDEAL_MAP.initial.yaw),
    };
    state.idealNav.localization = payload.state?.localization || null;
    state.idealNav.pose = newPose;
    state.idealNav.planningStart = clampPlanningPoint(newPose);
    resetIdealRecoveryState();
    state.idealNav.route = [];
    state.idealNav.routeActive = false;
    state.idealNav.routeMode = "normal";
    state.idealNav.mapPlan = null;
    syncPlanningStartInputs();
    state.idealNav.replayPoint = null;
    state.idealNav.finalYaw = newPose.yaw;
    updateIdealPathSummary();
    renderMapPlanResult();
    updateIdealMemorySummary();
    updateIdealExecutionControls();
    renderIdealAvoidanceSummary();
    renderIdealAbsoluteSummary();
    renderIdealRouteList();
    drawIdealMap();
    toast("自动雷达定位完成；机器人没有移动");
  } finally {
    if (button) button.disabled = state.idealNav.isRunning;
  }
}

async function loadIdealRun(runId) {
  if (!runId) return;
  const payload = await api(`/api/ideal-nav/runs/${runId}`);
  state.idealNav.loadedRun = payload.run.record;
  const goal = state.idealNav.loadedRun.request?.goal;
  if (goal) setIdealTarget({ x: goal.x, y: goal.y });
  renderIdealRunSummary();
}

function renderIdealRunSummary() {
  const root = $("#ideal-run-summary");
  if (!root) return;
  const record = state.idealNav.loadedRun;
  if (!record) {
    root.innerHTML = "";
    return;
  }
  const end = record.end_odom || {};
  const avoidance = record.request?.avoidance || {};
  const samples = Array.isArray(record.samples) ? record.samples : [];
  const lastObstacleSample = [...samples].reverse().find((sample) => sample.obstacle_front_min_m !== undefined);
  const lastCorrectionSample = [...samples].reverse().find((sample) => sample.perception_confidence !== undefined);
  root.innerHTML = [
    ["状态", idealStatusLabel(record.status)],
    ["采样点", `${samples.length}`],
    ["前方避障", avoidance.enabled ? "开启" : "关闭"],
    ["感知源", lastCorrectionSample?.perception_source || "-"],
    ["置信度", lastCorrectionSample?.perception_confidence == null ? "-" : Number(lastCorrectionSample.perception_confidence).toFixed(2)],
    ["横向偏差", lastCorrectionSample?.perception_lateral_error_m == null ? "-" : `${formatSigned(lastCorrectionSample.perception_lateral_error_m, 3)}m`],
    ["航向偏差", lastCorrectionSample?.perception_yaw_error_rad == null ? "-" : `${formatSigned(Number(lastCorrectionSample.perception_yaw_error_rad) * 180 / Math.PI, 1)}deg`],
    ["角速度修正", lastCorrectionSample?.correction_cmd_yaw == null ? "-" : `${formatSigned(lastCorrectionSample.correction_cmd_yaw, 3)}rad/s`],
    ["前方参考", lastObstacleSample?.obstacle_front_min_m == null ? "-" : `${Number(lastObstacleSample.obstacle_front_min_m).toFixed(2)}m`],
    ["终点里程", `${Number(end.x || 0).toFixed(3)}, ${Number(end.y || 0).toFixed(3)}`],
    ["朝向", `${Number(end.yaw || 0).toFixed(3)}rad`],
  ].map(([key, value]) => `<div><span>${key}</span><span>${escapeHtml(value)}</span></div>`).join("");
}

function replayLoadedRun() {
  const record = state.idealNav.loadedRun;
  if (!record || !Array.isArray(record.samples) || !record.samples.length) {
    toast("还没有加载轨迹");
    return;
  }
  replayPoints(record.samples.map((sample) => sampleToMap(record, sample)), 90);
}

function startLiveIdealRun(runId) {
  window.clearInterval(state.idealNav.liveTimer);
  state.idealNav.isRunning = true;
  updateIdealExecutionControls();
  state.idealNav.liveTimer = window.setInterval(async () => {
    try {
      const payload = await api(`/api/ideal-nav/runs/${runId}`);
      const record = payload.run.record;
      state.idealNav.loadedRun = record;
      const samples = Array.isArray(record.samples) ? record.samples : [];
      if (samples.length) {
        const latestSample = samples[samples.length - 1];
        state.idealNav.replayPoint = sampleToMap(record, latestSample);
        state.idealNav.latestCorrection = correctionFromSample(record, latestSample) || state.idealNav.latestCorrection;
        drawIdealMap();
      }
      renderIdealRunSummary();
      if (record.status && record.status !== "running") {
        window.clearInterval(state.idealNav.liveTimer);
        state.idealNav.liveTimer = null;
        state.idealNav.isRunning = false;
        if (record.status === "success") {
          const goal = record.request?.goal;
          if (goal) {
            state.idealNav.pose = {
              x: Number(goal.x),
              y: Number(goal.y),
              yaw: Number(goal.yaw ?? state.idealNav.pose.yaw),
            };
          }
        }
        updateIdealPathSummary();
        updateIdealMemorySummary();
        updateIdealExecutionControls();
        drawIdealMap();
        await refreshIdealRuns();
        await refreshIdealState({ force: true });
      }
    } catch (err) {
      // The record may not exist for the first few hundred milliseconds.
    }
  }, 300);
}


/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */


function latestObstacleStopSample(record) {
  const samples = Array.isArray(record?.samples) ? record.samples : [];
  return [...samples].reverse().find((sample) => sample?.phase === "obstacle-stop" && sample?.obstacle_blocked);
}

function obstacleSampleKey(runId, sample) {
  return `${runId}:${Number(sample?.t || 0).toFixed(3)}:${Number(sample?.x || 0).toFixed(3)}:${Number(sample?.y || 0).toFixed(3)}`;
}

function buildObstacleEncounter(record, sample) {
  const recovery = currentIdealRecoveryConfig();
  const robotPose = sampleToMap(record, sample);
  const frontMin = Number(sample?.obstacle_front_min_m);
  const travel = Number.isFinite(frontMin) && frontMin > 0
    ? frontMin
    : IDEAL_MAP.robotRadius + recovery.markRadius + 0.05;
  return {
    x: clamp(robotPose.x + Math.cos(robotPose.yaw || 0) * travel, 0, IDEAL_MAP.width),
    y: clamp(robotPose.y + Math.sin(robotPose.yaw || 0) * travel, 0, IDEAL_MAP.height),
    radiusM: recovery.markRadius,
    distanceM: Number.isFinite(frontMin) ? frontMin : null,
    robotPose,
    t: Number(sample?.t || 0),
  };
}

async function waitForJobToSettle(jobId, timeoutMs = 6000) {
  const deadline = Date.now() + timeoutMs;
  let lastJob = null;
  while (Date.now() < deadline) {
    try {
      const payload = await api(`/api/jobs/${jobId}`);
      lastJob = payload.job;
      if (payload.job?.status && payload.job.status !== "running") {
        return payload.job;
      }
    } catch (_err) {
      // Ignore transient polling failures while the job shuts down.
    }
    await delay(250);
  }
  return lastJob;
}


/* Removed obsolete overridden function. */


function startLiveIdealRoute(runIds, jobId, context) {
  window.clearInterval(state.idealNav.liveTimer);
  state.idealNav.isRunning = true;
  state.idealNav.pendingRouteRunIds = runIds;
  state.idealNav.activeRouteIndex = 0;
  state.idealNav.routeJobId = jobId;
  updateIdealExecutionControls();
  state.idealNav.liveTimer = window.setInterval(async () => {
    try {
      if (!runIds.length) return;
      const index = Math.min(state.idealNav.activeRouteIndex, runIds.length - 1);
      const runId = runIds[index];
      const payload = await api(`/api/ideal-nav/runs/${runId}`);
      const record = payload.run.record;
      state.idealNav.loadedRun = record;
      const samples = Array.isArray(record.samples) ? record.samples : [];
      if (samples.length) {
        state.idealNav.replayPoint = sampleToMap(record, samples[samples.length - 1]);
        drawIdealMap();
      }
      renderIdealRunSummary();
      if (state.idealNav.avoidObstacles && currentIdealRecoveryConfig().autoReplan) {
        const obstacleSample = latestObstacleStopSample(record);
        if (obstacleSample) {
          await recoverIdealRouteFromObstacle(runId, jobId, record);
          return;
        }
      }
      if (record.status === "success" && index < runIds.length - 1) {
        state.idealNav.activeRouteIndex = index + 1;
        return;
      }
      if (record.status === "failed") {
        await finishLiveIdealRoute(false);
        return;
      }
      if (record.status === "success" && index === runIds.length - 1) {
        const job = await api(`/api/jobs/${jobId}`);
        if (job.job.status === "success") {
          const finalPoint = context.finalGoal || context.waypoints[context.waypoints.length - 1];
          state.idealNav.pose = {
            x: Number(finalPoint.x),
            y: Number(finalPoint.y),
            yaw: Number(finalPoint.yaw ?? context.finalYaw ?? state.idealNav.finalYaw),
          };
          await finishLiveIdealRoute(true);
        } else if (job.job.status === "failed") {
          await finishLiveIdealRoute(false);
        }
      }
    } catch (_err) {
      // A later segment record may not exist until the previous segment exits.
    }
  }, 300);
}


/* Removed obsolete overridden function. */



/* Removed obsolete overridden function. */


function readNumberInputValue(selector, fallback) {
  const value = Number($(selector)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function currentIdealAvoidanceConfig() {
  const scanTopic = ($("#ideal-scan-topic")?.value || IDEAL_AVOIDANCE.scanTopic).trim() || IDEAL_AVOIDANCE.scanTopic;
  return {
    scanTopic,
    stopDistance: clamp(readNumberInputValue("#ideal-stop-distance", IDEAL_AVOIDANCE.stopDistance), 0.05, 5.0),
    frontDeg: clamp(readNumberInputValue("#ideal-front-deg", IDEAL_AVOIDANCE.frontDeg), 5, 180),
    staleS: clamp(readNumberInputValue("#ideal-stale-s", IDEAL_AVOIDANCE.staleS), 0.1, 10.0),
  };
}

function currentIdealRecoveryConfig() {
  return {
    autoReplan: Boolean($("#ideal-auto-replan")?.checked ?? IDEAL_RECOVERY.autoReplan),
    markRadius: clamp(readNumberInputValue("#ideal-replan-radius", IDEAL_RECOVERY.markRadius), 0.0, 2.0),
    maxReplans: Math.min(10, Math.max(0, Math.round(readNumberInputValue("#ideal-max-replans", IDEAL_RECOVERY.maxReplans)))),
  };
}

function currentIdealAbsoluteTrimConfig() {
  return {
    ...IDEAL_ABSOLUTE_TRIM,
    enabled: Boolean($("#ideal-absolute-trim-enabled")?.checked ?? IDEAL_ABSOLUTE_TRIM.enabled),
  };
}

function currentIdealFinalGoal() {
  const point = state.idealNav.executionGoal
    || state.idealNav.route[state.idealNav.route.length - 1]
    || state.idealNav.target;
  if (!point) return null;
  return {
    x: Number(point.x),
    y: Number(point.y),
    yaw: Number.isFinite(point.yaw) ? Number(point.yaw) : Number(state.idealNav.finalYaw),
  };
}

function formatAbsoluteReason(reason) {
  const labels = {
    absolute_pose_unavailable: "绝对位姿不可用",
    localization_state_unavailable: "定位状态不可用",
    initialed_unknown: "initialed 未知",
    localization_uninitialized: "定位未初始化",
    confidence_unknown: "confidence 未知",
    localization_confidence_low: "定位置信度低",
    move_access_false: "moveAccess 关闭",
    laser_warn_unavailable: "激光告警不可用",
    no_laser_received: "未收到激光",
    laser_hz_low: "激光频率低",
    nav_laser_low_pub_hz: "导航激光发布频率低",
    collision_shutdown: "碰撞停机",
    collision_warning: "碰撞告警",
    request_failed: "状态读取失败",
  };
  return labels[reason] || reason || "-";
}

function formatAbsoluteValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(3)}${suffix}`;
  }
  return `${value}${suffix}`;
}

function renderIdealAbsoluteSummary() {
  const root = $("#ideal-absolute-summary");
  if (!root) return;
  const trim = currentIdealAbsoluteTrimConfig();
  const absolute = state.idealNav.absoluteState;
  const localization = state.idealNav.localization || absolute?.localization;
  if (!localization) {
    root.innerHTML = [
      ["小地图定位", "等待读取"],
      ["激光雷达 2", "-"],
      ["深度相机 1", "-"],
    ].map(([key, value]) => `<div><span>${key}</span><span>${escapeHtml(value)}</span></div>`).join("");
    return;
  }
  const pose = localization.pose;
  const raw = localization.raw_pose;
  const correction = localization.correction;
  const initialOffset = localization.anchor?.initial_offset;
  const lidar = localization.sources?.["/scan"] || {};
  const depth = localization.sources?.["/camera_scan"] || {};
  const ready = Boolean(
    localization.initialized
    && localization.running
    && lidar.fresh
    && pose,
  );
  const modeLabels = {
    tracking: "地图匹配纠偏",
    "odom-anchor": "起点锚定 + 里程计",
    uninitialized: "等待重置起点",
    unavailable: "定位器不可用",
    error: "定位异常",
  };
  const sensorLabel = (source) => {
    if (!source || !source.samples) return "未收到";
    const age = Number(source.age_s);
    const validPoints = Number(source.valid_points);
    if (source.fresh && source.usable === false) {
      return `在线 · 0有效点/${source.samples}原始点 · 未参与定位`;
    }
    return source.fresh
      ? `在线 · ${Number.isFinite(validPoints) ? validPoints : source.samples}有效点 · ${Number.isFinite(age) ? age.toFixed(2) : "-"}s`
      : `数据过期 · ${Number.isFinite(age) ? age.toFixed(1) : "-"}s`;
  };
  const correctionLabel = correction
    ? `${Number(correction.distance_m || 0).toFixed(3)}m / ${Number(correction.yaw_deg || 0).toFixed(1)}deg`
    : "0.000m / 0.0deg";
  const lastApplied = localization.last_applied;
  const lastAppliedLabel = lastApplied
    ? `${Number(lastApplied.distance_m || 0).toFixed(3)}m / ${Number(lastApplied.yaw_deg || 0).toFixed(1)}deg`
    : "-";
  const initialOffsetLabel = initialOffset
    ? `${Number(initialOffset.distance_m || 0).toFixed(3)}m / ${Number(initialOffset.yaw_deg || 0).toFixed(1)}deg`
    : "-";
  const decision = localization.last_match?.decision || "";
  const decisionLabels = {
    within_threshold: "偏差在 0.05m / 5° 内，保持当前坐标",
    collecting_consensus: "正在收集连续 5 帧一致结果",
    insufficient_evidence: "匹配改善不足，拒绝写入坐标",
    depth_disagrees: "深度相机不支持本次修正，已拒绝",
    stationary_hold: "静止保护：只评估，不写入坐标",
    motion_hold: "运动中保护：等待停车后再做地图匹配纠偏",
    suggestion_only: "货架轮廓存在对称歧义：仅显示建议，不自动写入坐标",
    rate_limited: "平滑修正间隔保护",
    corrected: "已执行一次平滑修正",
    rejected: "传感器结果未通过质量检查",
  };
  const reasonText = ready
    ? (decisionLabels[decision] || (localization.tracking ? "持续纠偏中" : "定位可用，等待稳定地图匹配"))
    : (localization.error || (!localization.initialized ? "请把机器人放在标定起点后点击重置起点" : "激光雷达数据未就绪"));
  root.innerHTML = [
    ["小地图定位", ready ? "可用" : "未就绪"],
    ["当前模式", modeLabels[localization.mode] || localization.mode || "-"],
    ["方向基准", "地图上方 = 现场地理南"],
    ["起点锚定", "固定标记点；雷达/深度仅校验"],
    ["底盘坐标", Number(localization.anchor?.odom_y_sign) === -1 ? "左右与转角方向已校正" : "标准方向"],
    ["融合定位位置", pose ? `${Number(pose.x).toFixed(2)}, ${Number(pose.y).toFixed(2)}, ${robotHeadYawDeg(pose.yaw).toFixed(1)}deg` : "-"],
    ["原始锚定位姿", raw ? `${Number(raw.x).toFixed(2)}, ${Number(raw.y).toFixed(2)}, ${robotHeadYawDeg(raw.yaw).toFixed(1)}deg` : "-"],
    ["起点摆放修正", initialOffsetLabel],
    ["累计纠偏", correctionLabel],
    ["最近一次修正", lastAppliedLabel],
    ["匹配置信度", Number(localization.confidence || 0).toFixed(3)],
    ["修正次数", `${Number(localization.anchor?.correction_count || 0)}`],
    ["纠偏决策", decisionLabels[decision] || "-"],
    ["激光雷达 2", sensorLabel(lidar)],
    ["深度相机 1（辅助）", sensorLabel(depth)],
    ["终点二次校验", trim.enabled ? "开启" : "关闭"],
    ["状态", reasonText],
  ].map(([key, value]) => `<div><span>${key}</span><span>${escapeHtml(value)}</span></div>`).join("");
}


/* Removed obsolete overridden function. */


async function refreshIdealAbsoluteState(options = {}) {
  try {
    const payload = await api("/api/ideal-nav/absolute-state");
    state.idealNav.absoluteState = payload.absolute_state || null;
  } catch (err) {
    state.idealNav.absoluteState = {
      available: false,
      pose: null,
      gate: {
        ok: false,
        reasons: ["request_failed"],
        checks: {},
      },
      errors: [err.message],
      updated_at: Date.now() / 1000,
    };
    if (!options.silent) {
      toast(`绝对位姿读取失败：${err.message}`);
    }
    renderIdealAbsoluteSummary();
    if (options.throwOnError) throw err;
    return state.idealNav.absoluteState;
  }
  renderIdealAbsoluteSummary();
  if (options.successToast) {
    toast("已刷新绝对位姿与定位健康门");
  }
  return state.idealNav.absoluteState;
}

function absoluteTrimDelta(pose, goal) {
  const yaw = Number.isFinite(goal?.yaw) ? Number(goal.yaw) : Number(state.idealNav.finalYaw);
  return {
    distance: Math.hypot(Number(goal.x) - Number(pose.x), Number(goal.y) - Number(pose.y)),
    yawError: Math.abs(normalizeAngle(yaw - Number(pose.yaw))),
  };
}

async function startAbsoluteTrimToGoal(goal, options = {}) {
  const absolute = options.absoluteState || state.idealNav.absoluteState || await refreshIdealAbsoluteState({ silent: true });
  if (!goal) {
    throw new Error("没有可用于绝对精修的终点");
  }
  if (!absolute?.pose) {
    throw new Error("没有读到绝对位姿");
  }
  if (!absolute?.gate?.ok) {
    throw new Error("定位健康门未通过，不能做终点绝对精修");
  }
  const trim = currentIdealAbsoluteTrimConfig();
  const targetYaw = Number.isFinite(goal.yaw) ? Number(goal.yaw) : Number(state.idealNav.finalYaw);
  state.idealNav.absoluteTrimActive = true;
  state.idealNav.routeMode = "absolute-trim";
  renderIdealAbsoluteSummary();
  return startIdealRouteExecution(
    [{ x: Number(goal.x), y: Number(goal.y), yaw: targetYaw }],
    {
      finalGoal: { x: Number(goal.x), y: Number(goal.y), yaw: targetYaw },
      finalYaw: targetYaw,
      avoidObstacles: false,
      useMemory: false,
      initialPose: absolute.pose,
      executionMode: "absolute-trim",
      routeOverrides: {
        maxSpeed: trim.maxSpeed,
        maxYawSpeed: trim.maxYawSpeed,
        timeout: trim.timeout,
        posTol: trim.posTol,
        yawTol: trim.yawTol,
      },
      silentToast: options.silentToast,
    },
  );
}

function syncIdealAvoidanceInputs() {
  const avoidance = currentIdealAvoidanceConfig();
  const recovery = currentIdealRecoveryConfig();
  if ($("#ideal-scan-topic")) $("#ideal-scan-topic").value = avoidance.scanTopic;
  if ($("#ideal-stop-distance")) $("#ideal-stop-distance").value = avoidance.stopDistance.toFixed(2);
  if ($("#ideal-front-deg")) $("#ideal-front-deg").value = avoidance.frontDeg.toFixed(0);
  if ($("#ideal-stale-s")) $("#ideal-stale-s").value = avoidance.staleS.toFixed(1);
  setIdealAvoidObstaclesEnabled(state.idealNav.avoidObstacles);
  if ($("#ideal-auto-replan")) $("#ideal-auto-replan").checked = recovery.autoReplan;
  if ($("#ideal-replan-radius")) $("#ideal-replan-radius").value = recovery.markRadius.toFixed(2);
  if ($("#ideal-max-replans")) $("#ideal-max-replans").value = String(recovery.maxReplans);
  if ($("#ideal-absolute-trim-enabled")) $("#ideal-absolute-trim-enabled").checked = currentIdealAbsoluteTrimConfig().enabled;
}

function renderIdealAvoidanceSummary() {
  const root = $("#ideal-avoidance-summary");
  if (!root) return;
  const avoidance = currentIdealAvoidanceConfig();
  const recovery = currentIdealRecoveryConfig();
  const lastEncounter = state.idealNav.encounterPoints[state.idealNav.encounterPoints.length - 1];
  root.innerHTML = [
    ["感知话题", avoidance.scanTopic],
    ["前向参考", `${avoidance.stopDistance.toFixed(2)}m`],
    ["感知源", state.idealNav.latestCorrection?.source || "-"],
    ["置信度", Number.isFinite(state.idealNav.latestCorrection?.confidence) ? state.idealNav.latestCorrection.confidence.toFixed(2) : "-"],
    ["横向偏差", Number.isFinite(state.idealNav.latestCorrection?.lateralError) ? `${formatSigned(state.idealNav.latestCorrection.lateralError, 3)}m` : "-"],
    ["航向偏差", Number.isFinite(state.idealNav.latestCorrection?.yawError) ? `${formatSigned(state.idealNav.latestCorrection.yawError * 180 / Math.PI, 1)}deg` : "-"],
    ["角速度修正", Number.isFinite(state.idealNav.latestCorrection?.cmdYaw) ? `${formatSigned(state.idealNav.latestCorrection.cmdYaw, 3)}rad/s` : "-"],
    ["前向扇区", `${avoidance.frontDeg.toFixed(0)}deg`],
    ["数据超时", `${avoidance.staleS.toFixed(1)}s`],
    ["自动重规划", recovery.autoReplan ? "开启" : "关闭"],
    ["标记半径", `${recovery.markRadius.toFixed(2)}m`],
    ["最大重规划", `${recovery.maxReplans}`],
    ["已重规划", `${state.idealNav.recoveryCount}`],
    ["遇障点数", `${state.idealNav.encounterPoints.length}`],
    ["恢复状态", state.idealNav.recoveryBusy ? "重规划中" : "空闲"],
    ["最近遇障点", lastEncounter ? `${lastEncounter.x.toFixed(2)}, ${lastEncounter.y.toFixed(2)}` : "-"],
  ].map(([key, value]) => `<div><span>${key}</span><span>${escapeHtml(value)}</span></div>`).join("");
}


/* Removed obsolete overridden function. */


function newIdealRoute() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能新建路径");
    return;
  }
  resetIdealRecoveryState();
  state.idealNav.route = [];
  state.idealNav.routeActive = true;
  state.idealNav.routeMode = "normal";
  state.idealNav.replayPoint = null;
  updateIdealPathSummary();
  renderIdealAvoidanceSummary();
  renderIdealRouteList();
  drawIdealMap();
  toast("已新建路径，点击地图选择候选点后再加入路径。");
}

function clearIdealRoute() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能修改路径");
    return;
  }
  resetIdealRecoveryState();
  state.idealNav.route = [];
  state.idealNav.routeActive = false;
  state.idealNav.routeMode = "normal";
  state.idealNav.replayPoint = null;
  updateIdealPathSummary();
  renderIdealAvoidanceSummary();
  renderIdealRouteList();
  drawIdealMap();
}

function buildResetRoute() {
  resetIdealRecoveryState();
  const planningStart = currentPlanningStart();
  addIdealRouteWaypoint(
    { x: planningStart.x, y: planningStart.y, yaw: IDEAL_MAP.initial.yaw },
    { replace: true, finalYaw: IDEAL_MAP.initial.yaw, mode: "reset" },
  );
  toast("已生成复位路线，请确认预演后再执行。");
}

async function resetIdealStart() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能重置起点");
    return;
  }
  const payload = await api("/api/ideal-nav/reset", {
    method: "POST",
    // Clicking this control is the operator's confirmation that the base is
    // physically on the marked start, even when ROS inputs are reconnecting.
    body: JSON.stringify({ operator_confirmed_start: true }),
  });
  const pose = payload.state?.pose || {};
  const newPose = {
    x: Number(pose.x ?? IDEAL_MAP.initial.x),
    y: Number(pose.y ?? IDEAL_MAP.initial.y),
    yaw: Number(pose.yaw ?? IDEAL_MAP.initial.yaw),
  };
  state.idealNav.localization = payload.state?.localization || null;
  state.idealNav.pose = newPose;
  state.idealNav.planningStart = clampPlanningPoint(newPose);
  resetIdealRecoveryState();
  state.idealNav.route = [];
  state.idealNav.routeActive = false;
  state.idealNav.routeMode = "normal";
  state.idealNav.mapPlan = null;
  syncPlanningStartInputs();
  state.idealNav.replayPoint = null;
  state.idealNav.finalYaw = IDEAL_MAP.initial.yaw;
  updateIdealPathSummary();
  updateIdealMemorySummary();
  updateIdealExecutionControls();
  renderIdealAvoidanceSummary();
  renderIdealAbsoluteSummary();
  renderIdealRouteList();
  drawIdealMap();
  toast("已用底盘当前位置建立小地图起点锚点；机器人没有移动");
}


/* Removed obsolete overridden function. */


async function recoverIdealRouteFromObstacle(runId, jobId, record) {
  const recovery = currentIdealRecoveryConfig();
  const finalGoal = state.idealNav.executionGoal;
  const sample = latestObstacleStopSample(record);
  if (!finalGoal || !sample || !recovery.autoReplan || state.idealNav.recoveryBusy) return;
  if (state.idealNav.recoveryCount >= recovery.maxReplans) return;
  const sampleKey = obstacleSampleKey(runId, sample);
  if (sampleKey === state.idealNav.lastObstacleSampleKey) return;
  state.idealNav.lastObstacleSampleKey = sampleKey;
  state.idealNav.recoveryBusy = true;
  updateIdealExecutionControls();
  const encounter = buildObstacleEncounter(record, sample);
  state.idealNav.encounterPoints.push(encounter);
  state.idealNav.recoveryCount += 1;
  state.idealNav.pose = { ...encounter.robotPose };
  state.idealNav.replayPoint = { ...encounter.robotPose };
  state.idealNav.routeMode = "replan";
  renderIdealAvoidanceSummary();
  updateIdealPathSummary();
  updateIdealMemorySummary();
  renderIdealRouteList();
  drawIdealMap();
  window.clearInterval(state.idealNav.liveTimer);
  state.idealNav.liveTimer = null;
  try {
    await api(`/api/jobs/${jobId}/terminate`, {
      method: "POST",
      body: JSON.stringify({}),
    });
  } catch (_err) {
    // Best effort: the route job may already be winding down.
  }
  try {
    await waitForJobToSettle(jobId);
    await refreshIdealState({ force: true });
    const plan = await api("/api/map/plan", {
      method: "POST",
      body: JSON.stringify(mapPlanPayload({
        start: { x: encounter.robotPose.x, y: encounter.robotPose.y },
        goal: { x: finalGoal.x, y: finalGoal.y },
        includeOccupancy: true,
        snapToFree: true,
        temporaryObstacles: state.idealNav.encounterPoints,
      })),
    });
    state.idealNav.mapLayer = {
      map: plan.map,
      occupancy: plan.occupancy,
    };
    state.idealNav.mapPlan = plan;
    renderMapPlanResult(plan);
    const replannedWaypoints = (plan.waypoints || []).slice(1).map((point, index, items) => ({
      x: Number(point.x),
      y: Number(point.y),
      ...(index === items.length - 1
        ? { yaw: Number(finalGoal.yaw ?? state.idealNav.finalYaw) }
        : (Number.isFinite(point.yaw) ? { yaw: Number(point.yaw) } : {})),
    }));
    if (!replannedWaypoints.length) {
      throw new Error("重规划没有生成可执行的关键点。");
    }
    state.idealNav.route = replannedWaypoints.map((point) => ({ ...point }));
    state.idealNav.routeActive = true;
    state.idealNav.finalYaw = Number(finalGoal.yaw ?? state.idealNav.finalYaw);
    updateIdealPathSummary();
    renderIdealRouteList();
    drawIdealMap();
    await startIdealRouteExecution(replannedWaypoints, {
      finalGoal,
      finalYaw: state.idealNav.finalYaw,
      silentToast: true,
      replayPoint: encounter.robotPose,
    });
    toast(`检测到障碍，已重规划 ${state.idealNav.recoveryCount}/${recovery.maxReplans} 次。`);
  } catch (err) {
    toast(`遇障后重规划失败：${err.message}`);
    await finishLiveIdealRoute(false);
  } finally {
    state.idealNav.recoveryBusy = false;
    updateIdealExecutionControls();
    renderIdealAvoidanceSummary();
    drawIdealMap();
  }
}

async function executeIdealPath() {
  if (state.idealNav.isRunning) {
    toast("导航执行中，不能重复执行");
    return;
  }
  if (!state.idealNav.routeActive) {
    toast("请先新建路径");
    return;
  }
  if (!state.idealNav.route.length) {
    toast("请先把关键点加入路径");
    return;
  }
  resetIdealRecoveryState();
  renderIdealAvoidanceSummary();
  const waypoints = activeRouteWaypoints().map((point) => ({ ...point }));
  await startIdealRouteExecution(waypoints, {
    finalGoal: {
      x: Number(waypoints[waypoints.length - 1].x),
      y: Number(waypoints[waypoints.length - 1].y),
      yaw: state.idealNav.finalYaw,
    },
  });
}


/* Removed obsolete overridden function. */


async function refreshStatus() {
  const payload = await api("/api/status");
  const status = payload.status;
  state.status = status;
  state.config = status.config;

  const profileLabel = $("#profile-label");
  const roleLabel = $("#role-label");
  if (profileLabel) profileLabel.textContent = status.config.profile;
  if (roleLabel) roleLabel.textContent = `role: ${status.config.role}`;

  const zeroDot = $("#zero-dot");
  const zeroSummary = $("#zero-summary");
  const zeroUrl = $("#zero-url");
  if (zeroDot) zeroDot.className = statusClass(status.zerograsp.ok);
  if (status.zerograsp.ok) {
    const zeroPayload = status.zerograsp.payload || {};
    if (zeroSummary) {
      zeroSummary.textContent = zeroPayload.stub ? "服务在线 · Stub 模式" : "服务在线 · Real 模式";
    }
  } else {
    if (zeroSummary) zeroSummary.textContent = status.zerograsp.error || "健康检查失败";
  }
  if (zeroUrl) zeroUrl.textContent = status.config.perception_url;

  const networkDot = $("#network-dot");
  const networkSummary = $("#network-summary");
  const robotUrl = $("#robot-url");
  if (networkDot) networkDot.className = "status-dot ok";
  if (networkSummary) {
    networkSummary.textContent = status.config.role === "console"
      ? "控制端模式 · API 转发到机器人"
      : status.config.bind === "0.0.0.0"
        ? "本机执行 · 已开放局域网访问"
        : "本机执行 · 仅本机访问";
  }
  if (robotUrl) robotUrl.textContent = window.location.origin;

  const legacyAirbotDot = $("#airbot-dot");
  const legacyAirbotSummary = $("#airbot-summary");
  const legacyAirbotPorts = $("#airbot-ports");
  if (legacyAirbotDot) legacyAirbotDot.className = statusClass(status.airbot.ok);
  if (legacyAirbotSummary) {
    legacyAirbotSummary.textContent = status.airbot.tmux ? "tmux session exists" : "tmux session not found";
  }
  if (legacyAirbotPorts) {
    legacyAirbotPorts.innerHTML = status.airbot.ports
      .map((item) => {
        const label = item.listening ? "listening" : "closed";
        return `<div><span>${escapeHtml(item.label)}</span><span>${escapeHtml(item.host)}:${escapeHtml(item.port)} ${label}</span></div>`;
      })
      .join("");
  }

  const legacySafetyDot = $("#safety-dot");
  const legacySafetySummary = $("#safety-summary");
  const legacyRosUrl = $("#ros-url");
  if (legacySafetyDot) {
    legacySafetyDot.className = status.config.enable_hardware_actions ? "status-dot ok" : "status-dot bad";
  }
  if (legacySafetySummary) {
    legacySafetySummary.textContent = status.config.enable_hardware_actions
      ? "Hardware actions enabled"
      : "Hardware actions disabled";
  }
  if (legacyRosUrl) legacyRosUrl.textContent = status.config.ros_master_uri;

  const baseSpeed = $("#base-speed");
  const baseYaw = $("#base-yaw");
  const baseDuration = $("#base-duration");
  if (baseSpeed) baseSpeed.max = status.config.max_linear_mps;
  if (baseYaw) baseYaw.max = status.config.max_angular_radps;
  if (baseDuration) baseDuration.max = status.config.max_base_duration_s;
  renderGraspFlow();
}

async function startJob(path, body = {}) {
  const payload = await api(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
  if (payload.job) {
    state.selectedJobId = payload.job.id;
    window.dispatchEvent(new CustomEvent("ruc:job-started", { detail: payload.job }));
  }
  return payload;
}

async function postJob(path, body = {}) {
  const payload = await startJob(path, body);
  if (payload.job) {
    toast(`Started: ${payload.job.label}`);
    await refreshJobs();
  } else {
    toast("Done");
  }
  return payload;
}

async function refreshJobs() {
  const payload = await api("/api/jobs");
  const list = $("#job-list");
  if (list) {
    list.innerHTML = payload.jobs
      .map((job) => {
        const active = job.id === state.selectedJobId ? " data-active=\"true\"" : "";
        return `<button class="job-item" data-job-id="${escapeHtml(job.id)}"${active}>
          <strong>${escapeHtml(job.label)}</strong>
          <span>${escapeHtml(job.status)} ${escapeHtml(fmtTime(job.created_at))}</span>
        </button>`;
      })
      .join("");
  }
  updateRecoveryControls(payload.jobs);
  if (!state.selectedJobId && payload.jobs[0]) {
    state.selectedJobId = payload.jobs[0].id;
  }
  if ($("#job-log")) await refreshSelectedJob();
  await syncGraspFlow(payload.jobs);
}

function updateRecoveryControls(jobs) {
  const resetButton = $("[data-action=\"arms-reset\"]");
  if (!resetButton) return;
  const runningRecovery = jobs.some((job) => job.kind === "recovery" && job.status === "running");
  resetButton.disabled = runningRecovery;
  resetButton.textContent = runningRecovery ? "Resetting..." : "Reset Arms";
}

async function refreshSelectedJob() {
  const log = $("#job-log");
  if (!state.selectedJobId || !log) return;
  const payload = await api(`/api/jobs/${state.selectedJobId}`);
  log.textContent = payload.job.log || "No log yet.";
}

async function refreshArtifacts() {
  const payload = await api("/api/debug/artifacts");
  const root = $("#artifacts");
  if (!payload.artifacts.length) {
    root.innerHTML = `<div class="tool-panel full">No debug artifacts yet.</div>`;
    return;
  }
  root.innerHTML = payload.artifacts
    .map((item) => {
      const name = escapeHtml(item.name);
      const url = typeof item.url === "string" && item.url.startsWith("/artifacts/")
        ? escapeHtml(item.url)
        : "#";
      if ([".png", ".jpg", ".jpeg"].includes(item.suffix)) {
        return `<article class="artifact"><a href="${url}" target="_blank" rel="noopener noreferrer"><img src="${url}" alt="${name}">${name}</a></article>`;
      }
      if (item.suffix === ".mp4") {
        return `<article class="artifact"><video src="${url}" controls></video><a href="${url}" target="_blank" rel="noopener noreferrer">${name}</a></article>`;
      }
      return `<article class="artifact"><a href="${url}" target="_blank" rel="noopener noreferrer">${name}</a></article>`;
    })
    .join("");
}

function graspFormPayload() {
  return {
    prompt: $("#grasp-prompt").value,
    arm: $("#grasp-arm").value,
    strategy: $("#grasp-strategy").value,
    speed: $("#grasp-speed").value,
    endpoint: $("#grasp-endpoint").value,
  };
}

function samePreparedGrasp(a, b) {
  if (!a || !b) return false;
  return ["prompt", "arm", "strategy", "speed", "endpoint"].every((key) => a[key] === b[key]);
}

function invalidateRealityApproval(reason) {
  const flow = state.graspFlow;
  if (!flow.readyForReality && flow.phase !== "real-success") return;
  resetGraspFlow({
    phase: "idle",
    payload: null,
    lastNoticeKey: "",
  });
  renderGraspFlow(reason || "抓取参数已变化，请重新检查并预演。 ");
}

function graspStepState(phase) {
  const steps = {
    validate: { text: "等待", className: "" },
    dry: { text: "等待", className: "" },
    reality: { text: "未解锁", className: "" },
  };
  switch (phase) {
    case "validating":
      steps.validate = { text: "检查中", className: "is-active" };
      break;
    case "validation-failed":
      steps.validate = { text: "失败", className: "is-failed" };
      break;
    case "dry-run-starting":
    case "dry-running":
      steps.validate = { text: "通过", className: "is-success" };
      steps.dry = { text: "预演中", className: "is-active" };
      break;
    case "dry-run-failed":
      steps.validate = { text: "通过", className: "is-success" };
      steps.dry = { text: "失败", className: "is-failed" };
      break;
    case "ready":
      steps.validate = { text: "通过", className: "is-success" };
      steps.dry = { text: "完成", className: "is-ready" };
      steps.reality = { text: "已解锁", className: "is-ready" };
      break;
    case "executing-real":
      steps.validate = { text: "通过", className: "is-success" };
      steps.dry = { text: "完成", className: "is-ready" };
      steps.reality = { text: "执行中", className: "is-active" };
      break;
    case "real-failed":
      steps.validate = { text: "通过", className: "is-success" };
      steps.dry = { text: "完成", className: "is-ready" };
      steps.reality = { text: "失败", className: "is-failed" };
      break;
    case "real-success":
      steps.validate = { text: "通过", className: "is-success" };
      steps.dry = { text: "完成", className: "is-ready" };
      steps.reality = { text: "完成", className: "is-success" };
      break;
    default:
      break;
  }
  return steps;
}

function graspNoteForPhase(phase) {
  switch (phase) {
    case "validating":
      return "正在检查服务、相机与目标分割结果。";
    case "validation-failed":
      return "环境检查失败，请查看顶部统一日志后重试。";
    case "dry-run-starting":
      return "环境检查通过，正在启动软件预演。";
    case "dry-running":
      return "正在生成抓取方案，不会运动机器人硬件。";
    case "dry-run-failed":
      return "软件预演失败，请检查统一日志和下方调试产物。";
    case "ready":
      return "软件预演完成，请检查下方结果后再执行实机抓取。";
    case "executing-real":
      return "正在实机执行已检查的抓取方案。";
    case "real-failed":
      return "实机抓取失败，请先检查日志，不要直接重复执行。";
    case "real-success":
      return "实机抓取完成；下一次抓取必须重新检查并预演。";
    default:
      return "点击“检查并预演”完成环境检查和软件预演，确认结果后才能执行实机抓取。";
  }
}

function zeroGraspPayload() {
  return state.status?.zerograsp?.payload || {};
}

function realityBlockedReason(flow = state.graspFlow) {
  if (state.estop.latched !== false) {
    return "Web Console STOP 状态尚未解除。";
  }
  if (!state.config?.enable_hardware_actions) {
    return "Web Console 尚未启用硬件动作。";
  }
  if (!flow.payload) {
    return "请先完成环境检查和软件预演。";
  }
  if (flow.payload.endpoint !== "/grasp") {
    return "实机抓取必须使用实际 ZeroGrasp 接口，模拟接口只用于预演。";
  }
  const zerograsp = state.status?.zerograsp;
  if (!zerograsp?.ok) {
    return zerograsp?.error ? `ZeroGrasp 未就绪：${zerograsp.error}` : "ZeroGrasp 未就绪。";
  }
  const payload = zeroGraspPayload();
  if (payload.stub) {
    return "当前感知服务仍是 Stub 模式，请启动实际 ZeroGrasp 服务。";
  }
  const modelLoaded = payload.model_loaded || {};
  if (!modelLoaded.segment || !modelLoaded.grasp) {
    return "ZeroGrasp 的分割或抓取模型尚未完成加载。";
  }
  return "";
}

function renderGraspFlow(noteOverride = "") {
  const flow = state.graspFlow;
  const steps = graspStepState(flow.phase);
  [
    ["validate", $("#grasp-step-validate"), $("#grasp-step-validate-state")],
    ["dry", $("#grasp-step-dry-run"), $("#grasp-step-dry-run-state")],
    ["reality", $("#grasp-step-reality"), $("#grasp-step-reality-state")],
  ].forEach(([key, card, label]) => {
    card.className = `grasp-step ${steps[key].className}`.trim();
    label.textContent = steps[key].text;
  });

  const realityReason = realityBlockedReason(flow);
  const note = noteOverride || (flow.readyForReality && realityReason ? realityReason : graspNoteForPhase(flow.phase));
  $("#grasp-flow-note").textContent = note;

  const busy = ["validating", "dry-run-starting", "dry-running", "executing-real"].includes(flow.phase);
  $("#start-grasp").disabled = busy;

  const hardwareEnabled = Boolean(state.config?.enable_hardware_actions);
  const canRunReality = flow.readyForReality && !realityReason && !busy;
  $("#run-grasp-reality").disabled = !canRunReality;
  if (!hardwareEnabled) {
    $("#run-grasp-reality").textContent = "实机抓取未启用";
  } else if (flow.payload?.endpoint !== "/grasp") {
    $("#run-grasp-reality").textContent = "需要实际 ZeroGrasp 接口";
  } else if (zeroGraspPayload().stub) {
    $("#run-grasp-reality").textContent = "需要实际 ZeroGrasp 服务";
  } else {
    $("#run-grasp-reality").textContent = "执行实机抓取";
  }
  if (flow.readyForReality && realityReason) {
    $("#grasp-step-reality").className = "grasp-step is-failed";
    $("#grasp-step-reality-state").textContent = "受阻";
  }
}

async function maybeStartDryRun() {
  const flow = state.graspFlow;
  if (!flow.payload || flow.autoStartingDryRun || flow.dryRunJobId) return;
  flow.autoStartingDryRun = true;
  flow.phase = "dry-run-starting";
  renderGraspFlow();
  try {
    const payload = await startJob("/api/grasp/run", {
      ...flow.payload,
      dry_run: true,
      record_video: false,
      confirm: "",
    });
    flow.dryRunJobId = payload.job.id;
    flow.phase = "dry-running";
    toast("环境检查通过，正在启动软件预演。 ");
    await refreshJobs();
  } catch (err) {
    flow.phase = "dry-run-failed";
    flow.readyForReality = false;
    renderGraspFlow(err.message);
    toast(err.message);
  } finally {
    flow.autoStartingDryRun = false;
  }
}

async function syncGraspFlow(jobs) {
  const flow = state.graspFlow;
  const jobsById = new Map(jobs.map((job) => [job.id, job]));
  const validateJob = flow.validateJobId ? jobsById.get(flow.validateJobId) : null;
  const dryRunJob = flow.dryRunJobId ? jobsById.get(flow.dryRunJobId) : null;
  const realJob = flow.realJobId ? jobsById.get(flow.realJobId) : null;

  if (flow.phase === "validating" && validateJob) {
    if (validateJob.status === "failed") {
      flow.phase = "validation-failed";
      flow.readyForReality = false;
      renderGraspFlow();
      await refreshArtifacts();
      if (flow.lastNoticeKey !== `validation-failed:${validateJob.id}`) {
        flow.lastNoticeKey = `validation-failed:${validateJob.id}`;
        toast("环境检查失败，请查看顶部统一日志。 ");
      }
      return;
    }
    if (validateJob.status === "success") {
      await refreshArtifacts();
      await maybeStartDryRun();
      return;
    }
  }

  if (["dry-run-starting", "dry-running"].includes(flow.phase) && dryRunJob) {
    if (dryRunJob.status === "failed") {
      flow.phase = "dry-run-failed";
      flow.readyForReality = false;
      renderGraspFlow();
      await refreshArtifacts();
      if (flow.lastNoticeKey !== `dry-run-failed:${dryRunJob.id}`) {
        flow.lastNoticeKey = `dry-run-failed:${dryRunJob.id}`;
        toast("软件预演失败，请检查日志与调试产物。 ");
      }
      return;
    }
    if (dryRunJob.status === "success") {
      flow.phase = "ready";
      flow.readyForReality = true;
      renderGraspFlow();
      await refreshArtifacts();
      if (flow.lastNoticeKey !== `dry-run-ready:${dryRunJob.id}`) {
        flow.lastNoticeKey = `dry-run-ready:${dryRunJob.id}`;
        toast("软件预演完成，请确认产物后再执行实机抓取。 ");
      }
      return;
    }
  }

  if (flow.phase === "executing-real" && realJob) {
    if (realJob.status === "failed") {
      flow.phase = "real-failed";
      flow.readyForReality = false;
      renderGraspFlow();
      await refreshArtifacts();
      if (flow.lastNoticeKey !== `real-failed:${realJob.id}`) {
        flow.lastNoticeKey = `real-failed:${realJob.id}`;
        toast("实机抓取失败，请先检查日志。 ");
      }
      return;
    }
    if (realJob.status === "success") {
      flow.phase = "real-success";
      flow.readyForReality = false;
      renderGraspFlow();
      await refreshArtifacts();
      if (flow.lastNoticeKey !== `real-success:${realJob.id}`) {
        flow.lastNoticeKey = `real-success:${realJob.id}`;
        toast("实机抓取完成。 ");
      }
      return;
    }
  }

  renderGraspFlow();
}

function activeCameraGrid() {
  const activePanel = document.getElementById(state.activeView);
  if (!activePanel) return null;
  return activePanel.querySelector("[data-camera-grid]");
}

function clearInactiveCameraGrids() {
  $$("[data-camera-grid]").forEach((grid) => {
    if (!grid.closest(`#${state.activeView}`)) {
      grid.querySelectorAll("img").forEach((img) => {
        img.removeAttribute("src");
      });
      grid.innerHTML = "";
    }
  });
}

function clearActiveCameraGrid() {
  const grid = activeCameraGrid();
  if (!grid) return;
  grid.querySelectorAll("img").forEach((img) => {
    img.removeAttribute("src");
  });
  grid.innerHTML = "";
}

function orderedCameras(cameras) {
  const byName = new Map(cameras.map((camera) => [camera.name, camera]));
  const priority = ["left", "front", "right"];
  const preferred = priority
    .map((name) => byName.get(name) || DEFAULT_CAMERAS.find((camera) => camera.name === name))
    .filter(Boolean);
  const extras = cameras.filter((camera) => !priority.includes(camera.name));
  return [...preferred, ...extras];
}

function cameraSupportsDepth(camera) {
  if (!camera) return false;
  if (typeof camera.depth_capable === "boolean") return camera.depth_capable;
  return /^(realsense|rs):/i.test(String(camera.source || "").trim());
}

function camerasForGrid(root, cameras) {
  const ordered = orderedCameras(cameras);
  if (root && ordered.length) {
    const grasp = state.graspAssist;
    if (
      root.id === "teleop-camera-grid"
      && !ordered.some((camera) => camera.name === grasp.mainCamera)
    ) {
      grasp.mainCamera = (
        ordered.find((camera) => camera.name === "front")?.name
        || ordered[0].name
      );
    }
    root.dataset.cameraCount = String(ordered.length);
    root.style.setProperty("--camera-count", String(ordered.length));
    root.style.setProperty("--thumb-count", String(Math.max(1, ordered.length - 1)));
  }
  return ordered;
}

function gripperAxisGuideMarkup(camera) {
  const points = Array.isArray(camera?.gripper_axis_guide)
    ? camera.gripper_axis_guide
      .map((point) => ({
        x: Number(point?.x),
        y: Number(point?.y),
        distanceCm: Number(point?.distance_cm),
      }))
      .filter((point) => (
        Number.isFinite(point.x)
        && Number.isFinite(point.y)
        && Number.isFinite(point.distanceCm)
        && point.x >= 0
        && point.x <= 1
        && point.y >= 0
        && point.y <= 1
      ))
    : [];
  if (points.length < 2) return "";
  const coordinates = points
    .map((point) => `${(point.x * 960).toFixed(1)},${(point.y * 540).toFixed(1)}`)
    .join(" ");
  const markers = points
    .filter((point) => ![25, 35].includes(point.distanceCm))
    .map((point) => {
      const x = (point.x * 960).toFixed(1);
      const y = (point.y * 540).toFixed(1);
      return `<g>
        <circle cx="${x}" cy="${y}" r="6" fill="#ffd44d" stroke="#111827" stroke-width="2"></circle>
        <text x="${(Number(x) + 11).toFixed(1)}" y="${(Number(y) - 8).toFixed(1)}"
          fill="#fff7c2" stroke="#111827" stroke-width="3" paint-order="stroke"
          font-size="18" font-weight="800">${point.distanceCm} cm</text>
      </g>`;
    })
    .join("");
  return `<svg class="gripper-axis-guide grasp-detail-only" viewBox="0 0 960 540"
    preserveAspectRatio="none" aria-label="夹爪轴深度引导线"
    style="position:absolute;inset:0;width:100%;height:100%;z-index:3;pointer-events:none">
    <polyline points="${coordinates}" fill="none" stroke="#ffd44d" stroke-width="3"
      stroke-dasharray="10 8" vector-effect="non-scaling-stroke"></polyline>
    ${markers}
    <text x="22" y="35" fill="#fff7c2" stroke="#111827" stroke-width="4"
      paint-order="stroke" font-size="20" font-weight="900">夹爪轴 · 软件校正</text>
  </svg>`;
}

function defaultGraspTarget(camera) {
  const points = Array.isArray(camera?.gripper_axis_guide)
    ? camera.gripper_axis_guide
    : [];
  const reference = points.reduce((best, point) => {
    const x = Number(point?.x);
    const y = Number(point?.y);
    const distance = Number(point?.distance_cm);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(distance)) {
      return best;
    }
    const delta = Math.abs(distance - 40);
    return !best || delta < best.delta ? { x, y, delta } : best;
  }, null);
  return reference || { x: 0.5, y: 0.55 };
}

function graspOverlayMarkup(camera) {
  return `<div class="grasp-overlay" data-grasp-overlay="${escapeHtml(camera.name)}">
    ${gripperAxisGuideMarkup(camera)}
    <div class="grasp-camera-envelope grasp-detail-only"></div>
    <div class="grasp-jaw is-left grasp-detail-only"></div>
    <div class="grasp-jaw is-right grasp-detail-only"></div>
    <div class="grasp-pinch-zone grasp-detail-only"></div>
    <div class="grasp-reticle"></div>
    <div class="grasp-live-readout grasp-detail-only">
      相机到目标
      <strong data-grasp-live-distance>-- mm</strong>
      <span data-grasp-live-tip>尖端余量 -- mm</span>
    </div>
    <span class="grasp-zone-chip is-left grasp-detail-only" data-grasp-zone="left">左 --</span>
    <span class="grasp-zone-chip is-right grasp-detail-only" data-grasp-zone="right">右 --</span>
    <span class="grasp-zone-chip is-top grasp-detail-only" data-grasp-zone="top">上 --</span>
  </div>`;
}

function armTemperatureFullscreenAlertMarkup() {
  return `<section
    id="arm-temperature-fullscreen-alert"
    class="arm-temperature-fullscreen-alert"
    role="alert"
    aria-live="assertive"
    hidden
  >
    <span class="arm-temperature-fullscreen-icon" aria-hidden="true">⚠</span>
    <div>
      <strong>机械臂高温：立即停止操作</strong>
      <span id="arm-temperature-fullscreen-alert-detail"></span>
    </div>
  </section>`;
}

function armJointLimitFullscreenAlertMarkup() {
  return `<section
    id="arm-joint-limit-fullscreen-alert"
    class="arm-joint-limit-fullscreen-alert"
    role="alert"
    aria-live="assertive"
    hidden
  >
    <span class="arm-joint-limit-fullscreen-icon" aria-hidden="true">⚠</span>
    <div>
      <strong id="arm-joint-limit-fullscreen-alert-title">机械臂接近限位：请减速回退</strong>
      <span id="arm-joint-limit-fullscreen-alert-detail"></span>
    </div>
  </section>`;
}

function renderActiveCameraGrid(cameras) {
  const root = activeCameraGrid();
  if (!root) return;
  const displayLabels = { left: "Left", front: "Center", right: "Right" };
  const isTeleop = state.activeView === "teleop-view";
  root.innerHTML = camerasForGrid(root, cameras)
    .map((camera) => {
      const source = camera.source || "not configured";
      const frameUrl = camera.frame_url || `/api/cameras/${encodeURIComponent(camera.name)}/frame.jpg`;
      const hasDepth = cameraSupportsDepth(camera);
      const mainClass = isTeleop && camera.name === state.graspAssist.mainCamera
        ? " is-main"
        : "";
      return `<article class="camera-card${mainClass}" data-camera-name="${escapeHtml(camera.name)}">
        <header>
          <h3>${escapeHtml(displayLabels[camera.name] || camera.label)}</h3>
          <span>${escapeHtml(source)}</span>
        </header>
        <div class="camera-visual">
          <img
            data-low-latency-camera="${escapeHtml(camera.name)}"
            data-frame-url="${escapeHtml(frameUrl)}"
            data-render-token="${state.cameraRenderToken}"
            alt="${escapeHtml(camera.label)} camera stream"
          >
          ${isTeleop && hasDepth ? graspOverlayMarkup(camera) : ""}
        </div>
      </article>`;
    })
    .join("")
    + (isTeleop
      ? armTemperatureFullscreenAlertMarkup() + armJointLimitFullscreenAlertMarkup()
      : "");
  startLowLatencyCameraStreams(root);
  applyGraspAssistStateToDom();
  if (window.rucSyncArmSafetyAlerts) {
    window.rucSyncArmSafetyAlerts();
  } else {
    window.rucSyncArmTemperatureAlert?.();
  }
}

function sleepCameraRetry(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function loadLatestCameraFrames(image) {
  const frameUrl = image.dataset.frameUrl;
  const renderToken = image.dataset.renderToken;
  let version = -1;
  let activeObjectUrl = "";

  try {
    while (
      image.isConnected
      && image.dataset.renderToken === renderToken
      && Number(renderToken) === state.cameraRenderToken
    ) {
      if (document.hidden) {
        await sleepCameraRetry(150);
        continue;
      }
      try {
        const separator = frameUrl.includes("?") ? "&" : "?";
        const response = await fetch(
          `${frameUrl}${separator}after=${version}&_=${Date.now()}`,
          {
            cache: "no-store",
            credentials: "same-origin",
          },
        );
        if (!response.ok) {
          throw new Error(`camera frame HTTP ${response.status}`);
        }
        const nextVersion = Number(response.headers.get("X-Camera-Version"));
        const sourceFrameTime = Number(response.headers.get("X-Camera-Frame-Time"));
        const blob = await response.blob();
        if (
          !image.isConnected
          || image.dataset.renderToken !== renderToken
          || Number(renderToken) !== state.cameraRenderToken
        ) {
          break;
        }

        const nextObjectUrl = URL.createObjectURL(blob);
        image.src = nextObjectUrl;
        try {
          await image.decode();
        } catch (_err) {
          // Retry on the next frame if a transient decode was interrupted.
        }
        if (Number.isFinite(sourceFrameTime) && sourceFrameTime > 0) {
          image.dataset.frameAgeMs = String(
            Math.max(0, Math.round(Date.now() - sourceFrameTime * 1000)),
          );
        }
        if (Number.isFinite(nextVersion)) {
          image.dataset.cameraVersion = String(nextVersion);
        }
        if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
        activeObjectUrl = nextObjectUrl;
        if (Number.isFinite(nextVersion)) version = nextVersion;
      } catch (_err) {
        await sleepCameraRetry(120);
      }
    }
  } finally {
    if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
  }
}

function startLowLatencyCameraStreams(root) {
  root.querySelectorAll("img[data-low-latency-camera]").forEach((image) => {
    loadLatestCameraFrames(image).catch(() => {});
  });
}

function renderCameraPlaceholders(cameras) {
  const root = activeCameraGrid();
  if (!root) return;
  const displayLabels = { left: "Left", front: "Center", right: "Right" };
  const isTeleop = state.activeView === "teleop-view";
  root.innerHTML = camerasForGrid(root, cameras)
    .map((camera) => {
      const hasDepth = cameraSupportsDepth(camera);
      const mainClass = isTeleop && camera.name === state.graspAssist.mainCamera
        ? " is-main"
        : "";
      return `<article class="camera-card${mainClass}" data-camera-name="${escapeHtml(camera.name)}">
        <header>
          <h3>${escapeHtml(displayLabels[camera.name] || camera.label)}</h3>
          <span>loading</span>
        </header>
        <div class="camera-visual">
          <div class="camera-placeholder">Loading camera source</div>
          ${isTeleop && hasDepth ? graspOverlayMarkup(camera) : ""}
        </div>
      </article>`;
    })
    .join("")
    + (isTeleop
      ? armTemperatureFullscreenAlertMarkup() + armJointLimitFullscreenAlertMarkup()
      : "");
  applyGraspAssistStateToDom();
  if (window.rucSyncArmSafetyAlerts) {
    window.rucSyncArmSafetyAlerts();
  } else {
    window.rucSyncArmTemperatureAlert?.();
  }
}

function formatGraspDistance(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? `${Math.round(numeric)} mm` : "-- mm";
}

function setGraspText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function applyGraspAssistStateToDom() {
  const grasp = state.graspAssist;
  const grid = $("#teleop-camera-grid");
  const toolbar = $("#teleop-grasp-toolbar");
  const button = $("#teleop-grasp-mode");
  if (!grid) return;
  const mainCamera = (state.cameras || DEFAULT_CAMERAS).find(
    (camera) => camera.name === grasp.mainCamera,
  );
  const mainHasDepth = cameraSupportsDepth(mainCamera);
  grid.classList.toggle("is-grasp-mode", grasp.enabled);
  grid.classList.toggle("is-color-only-main", grasp.enabled && !mainHasDepth);
  if (!mainHasDepth) {
    grid.classList.remove("is-grasp-warning", "is-grasp-danger");
  }
  const cards = Array.from(grid.querySelectorAll(".camera-card"));
  grid.dataset.cameraCount = String(cards.length);
  grid.style.setProperty("--camera-count", String(cards.length));
  grid.style.setProperty("--thumb-count", String(Math.max(1, cards.length - 1)));
  grid.querySelectorAll(".camera-card").forEach((card) => {
    card.classList.toggle("is-main", card.dataset.cameraName === grasp.mainCamera);
  });
  grid.querySelectorAll(".grasp-overlay").forEach((overlay) => {
    const isMain = overlay.dataset.graspOverlay === grasp.mainCamera;
    overlay.style.setProperty("--target-x", isMain ? `${grasp.targetX * 100}%` : "50%");
    overlay.style.setProperty("--target-y", isMain ? `${grasp.targetY * 100}%` : "55%");
  });
  grid.querySelectorAll(".camera-card img").forEach((image) => {
    if (image.dataset.graspGeometryBound === "1") return;
    image.dataset.graspGeometryBound = "1";
    image.addEventListener("load", scheduleGraspOverlayGeometrySync, { once: true });
  });
  syncGraspOverlayGeometry();
  if (toolbar) {
    toolbar.hidden = !grasp.enabled;
    toolbar.classList.toggle("is-color-only-main", !mainHasDepth);
  }
  const hint = $("#grasp-toolbar-hint");
  if (hint) {
    hint.textContent = mainHasDepth
      ? "点击主画面选择目标；点击小画面可切换主相机"
      : "当前为普通彩色主画面；点击小画面可切换主相机";
  }
  if (button) {
    button.setAttribute("aria-pressed", String(grasp.enabled));
    button.textContent = grasp.enabled ? "G 退出抓取辅助" : "G 抓取辅助";
  }
  syncTeleopCameraExpandButton();
}

function graspDepthSeverity(depth) {
  if (!depth?.available || depth.stale) return "waiting";
  const grasp = state.graspAssist;
  const targetTipMm = Number(depth.target_mm) - grasp.tipOffsetMm;
  const distances = [
    targetTipMm,
    depth.zones?.left?.distance_mm,
    depth.zones?.right?.distance_mm,
    depth.zones?.top?.distance_mm,
  ].map(Number).filter((value) => Number.isFinite(value) && value > 0);
  if (!distances.length) return "waiting";
  const nearest = Math.min(...distances);
  if (nearest <= grasp.dangerDistanceMm) return "danger";
  if (nearest <= grasp.warningDistanceMm) return "warning";
  return "clear";
}

function renderGraspDepth(depth) {
  const grasp = state.graspAssist;
  grasp.latestDepth = depth || null;
  const cameraDistanceMm = depth?.available ? Number(depth.target_mm) : NaN;
  const tipDistanceMm = Number.isFinite(cameraDistanceMm)
    ? Math.max(0, cameraDistanceMm - grasp.tipOffsetMm)
    : NaN;
  const zoneDistance = (name) => depth?.zones?.[name]?.distance_mm;
  setGraspText("#grasp-camera-distance", formatGraspDistance(cameraDistanceMm));
  setGraspText("#grasp-tip-distance", formatGraspDistance(tipDistanceMm));
  setGraspText("#grasp-left-distance", formatGraspDistance(zoneDistance("left")));
  setGraspText("#grasp-right-distance", formatGraspDistance(zoneDistance("right")));
  setGraspText("#grasp-top-distance", formatGraspDistance(zoneDistance("top")));

  const grid = $("#teleop-camera-grid");
  const stateChip = $("#grasp-depth-state");
  const severity = graspDepthSeverity(depth);
  grid?.classList.toggle("is-grasp-warning", severity === "warning");
  grid?.classList.toggle("is-grasp-danger", severity === "danger");
  if (stateChip) {
    stateChip.className = `grasp-depth-state${severity === "clear" ? "" : ` is-${severity}`}`;
    stateChip.textContent = severity === "danger"
      ? "危险距离"
      : severity === "warning"
        ? "接近障碍"
        : severity === "clear"
          ? "深度正常"
          : depth?.reason || "等待深度";
  }

  const mainCard = grid?.querySelector(`.camera-card[data-camera-name="${grasp.mainCamera}"]`);
  if (!mainCard) return;
  const liveDistance = mainCard.querySelector("[data-grasp-live-distance]");
  const liveTip = mainCard.querySelector("[data-grasp-live-tip]");
  if (liveDistance) liveDistance.textContent = formatGraspDistance(cameraDistanceMm);
  if (liveTip) liveTip.textContent = `尖端余量 ${formatGraspDistance(tipDistanceMm)}`;
  ["left", "right", "top"].forEach((name) => {
    const chip = mainCard.querySelector(`[data-grasp-zone="${name}"]`);
    if (!chip) return;
    const labels = { left: "左", right: "右", top: "上" };
    chip.textContent = `${labels[name]} ${formatGraspDistance(zoneDistance(name))}`;
  });
}

async function pollGraspDepth() {
  const grasp = state.graspAssist;
  if (!grasp.enabled || state.activeView !== "teleop-view" || grasp.pollBusy) return;
  const camera = (state.cameras || DEFAULT_CAMERAS).find(
    (item) => item.name === grasp.mainCamera,
  );
  if (!cameraSupportsDepth(camera)) return;
  grasp.pollBusy = true;
  try {
    const query = new URLSearchParams({
      x: String(grasp.targetX),
      y: String(grasp.targetY),
    });
    const payload = await api(`/api/cameras/${encodeURIComponent(grasp.mainCamera)}/depth?${query}`);
    renderGraspDepth(payload.depth);
  } catch (error) {
    renderGraspDepth({ available: false, reason: error.message || "深度读取失败" });
  } finally {
    grasp.pollBusy = false;
  }
}

function syncGraspDepthPolling() {
  const grasp = state.graspAssist;
  if (grasp.pollTimer) {
    window.clearInterval(grasp.pollTimer);
    grasp.pollTimer = null;
  }
  const camera = (state.cameras || DEFAULT_CAMERAS).find(
    (item) => item.name === grasp.mainCamera,
  );
  if (
    !grasp.enabled
    || state.activeView !== "teleop-view"
    || !cameraSupportsDepth(camera)
  ) return;
  pollGraspDepth();
  grasp.pollTimer = window.setInterval(pollGraspDepth, 400);
}

function renderedImageContentRect(img) {
  const rect = img.getBoundingClientRect();
  const sourceWidth = img.naturalWidth || 16;
  const sourceHeight = img.naturalHeight || 9;
  const sourceRatio = sourceWidth / sourceHeight;
  const boxRatio = rect.width / Math.max(1, rect.height);
  const objectFit = window.getComputedStyle(img).objectFit;
  let displayWidth = rect.width;
  let displayHeight = rect.height;
  let offsetX = 0;
  let offsetY = 0;
  if (objectFit === "cover") {
    if (boxRatio > sourceRatio) {
      displayHeight = rect.width / sourceRatio;
      offsetY = (rect.height - displayHeight) / 2;
    } else {
      displayWidth = rect.height * sourceRatio;
      offsetX = (rect.width - displayWidth) / 2;
    }
  } else if (objectFit !== "fill") {
    if (boxRatio > sourceRatio) {
      displayWidth = rect.height * sourceRatio;
      offsetX = (rect.width - displayWidth) / 2;
    } else {
      displayHeight = rect.width / sourceRatio;
      offsetY = (rect.height - displayHeight) / 2;
    }
  }
  return {
    left: rect.left + offsetX,
    top: rect.top + offsetY,
    width: displayWidth,
    height: displayHeight,
  };
}

let graspOverlayGeometryFrame = 0;

function syncGraspOverlayGeometry() {
  const grid = $("#teleop-camera-grid");
  if (!grid) return;
  grid.querySelectorAll(".grasp-overlay").forEach((overlay) => {
    const visual = overlay.parentElement;
    const image = visual?.querySelector("img");
    if (!visual || !image) return;
    const visualRect = visual.getBoundingClientRect();
    const contentRect = renderedImageContentRect(image);
    if (
      visualRect.width <= 0
      || visualRect.height <= 0
      || contentRect.width <= 0
      || contentRect.height <= 0
    ) return;
    overlay.style.inset = "auto";
    overlay.style.left = `${contentRect.left - visualRect.left}px`;
    overlay.style.top = `${contentRect.top - visualRect.top}px`;
    overlay.style.width = `${contentRect.width}px`;
    overlay.style.height = `${contentRect.height}px`;
  });
}

function scheduleGraspOverlayGeometrySync() {
  if (graspOverlayGeometryFrame) return;
  graspOverlayGeometryFrame = window.requestAnimationFrame(() => {
    graspOverlayGeometryFrame = 0;
    syncGraspOverlayGeometry();
  });
}

function normalizedImagePoint(img, clientX, clientY) {
  const contentRect = renderedImageContentRect(img);
  const x = (clientX - contentRect.left) / Math.max(1, contentRect.width);
  const y = (clientY - contentRect.top) / Math.max(1, contentRect.height);
  return {
    x: Math.min(1, Math.max(0, x)),
    y: Math.min(1, Math.max(0, y)),
  };
}

function handleGraspCameraClick(event) {
  const grasp = state.graspAssist;
  if (!grasp.enabled) return;
  const card = event.target.closest(".camera-card[data-camera-name]");
  if (!card) return;
  const cameraName = card.dataset.cameraName;
  const camera = (state.cameras || DEFAULT_CAMERAS).find(
    (item) => item.name === cameraName,
  );
  if (!camera) return;
  if (cameraName !== grasp.mainCamera) {
    grasp.mainCamera = cameraName;
    const target = defaultGraspTarget(camera);
    grasp.targetX = target.x;
    grasp.targetY = target.y;
    renderActiveCameraGrid(state.cameras || DEFAULT_CAMERAS);
    renderGraspDepth(null);
    syncGraspDepthPolling();
    return;
  }
  if (!cameraSupportsDepth(camera)) return;
  const image = card.querySelector("img");
  if (!image) return;
  const point = normalizedImagePoint(image, event.clientX, event.clientY);
  grasp.targetX = point.x;
  grasp.targetY = point.y;
  applyGraspAssistStateToDom();
  pollGraspDepth();
}

function readGraspSetting(inputId, fallback) {
  const value = Number($(`#${inputId}`)?.value);
  return Number.isFinite(value) ? value : fallback;
}

function updateGraspSettings() {
  const grasp = state.graspAssist;
  grasp.tipOffsetMm = Math.max(0, readGraspSetting("grasp-tip-offset", grasp.tipOffsetMm));
  grasp.warningDistanceMm = Math.max(20, readGraspSetting("grasp-warning-distance", grasp.warningDistanceMm));
  grasp.dangerDistanceMm = Math.max(10, readGraspSetting("grasp-danger-distance", grasp.dangerDistanceMm));
  if (grasp.dangerDistanceMm >= grasp.warningDistanceMm) {
    grasp.dangerDistanceMm = Math.max(10, grasp.warningDistanceMm - 10);
    const dangerInput = $("#grasp-danger-distance");
    if (dangerInput) dangerInput.value = String(grasp.dangerDistanceMm);
  }
  try {
    window.localStorage.setItem("ruc-grasp-settings", JSON.stringify({
      tipOffsetMm: grasp.tipOffsetMm,
      warningDistanceMm: grasp.warningDistanceMm,
      dangerDistanceMm: grasp.dangerDistanceMm,
    }));
  } catch (_error) {
    // Settings persistence is optional.
  }
  renderGraspDepth(grasp.latestDepth);
}

function loadGraspSettings() {
  try {
    const saved = JSON.parse(window.localStorage.getItem("ruc-grasp-settings") || "{}");
    if (Number.isFinite(Number(saved.tipOffsetMm))) state.graspAssist.tipOffsetMm = Math.max(0, Number(saved.tipOffsetMm));
    if (Number.isFinite(Number(saved.warningDistanceMm))) state.graspAssist.warningDistanceMm = Math.max(20, Number(saved.warningDistanceMm));
    if (Number.isFinite(Number(saved.dangerDistanceMm))) state.graspAssist.dangerDistanceMm = Math.max(10, Number(saved.dangerDistanceMm));
  } catch (_error) {
    // Ignore malformed or unavailable local settings.
  }
  const values = {
    "grasp-tip-offset": state.graspAssist.tipOffsetMm,
    "grasp-warning-distance": state.graspAssist.warningDistanceMm,
    "grasp-danger-distance": state.graspAssist.dangerDistanceMm,
  };
  Object.entries(values).forEach(([id, value]) => {
    const input = $(`#${id}`);
    if (input) input.value = String(value);
  });
}

function toggleGraspAssist(forceEnabled) {
  const grasp = state.graspAssist;
  grasp.enabled = typeof forceEnabled === "boolean" ? forceEnabled : !grasp.enabled;
  if (grasp.enabled) {
    const cameras = state.cameras || DEFAULT_CAMERAS;
    if (!cameras.some((camera) => camera.name === grasp.mainCamera)) {
      grasp.mainCamera = (
        cameras.find((camera) => camera.name === "front")?.name
        || cameras[0]?.name
        || ""
      );
    }
    const camera = cameras.find((item) => item.name === grasp.mainCamera);
    const target = defaultGraspTarget(camera);
    grasp.targetX = target.x;
    grasp.targetY = target.y;
  } else {
    renderGraspDepth(null);
  }
  applyGraspAssistStateToDom();
  syncGraspDepthPolling();
}

async function refreshCameras(options = {}) {
  if (!activeCameraGrid()) return;
  if (options.forceReconnect) {
    state.cameraRenderToken += 1;
    clearActiveCameraGrid();
    renderCameraPlaceholders(state.cameras || DEFAULT_CAMERAS);
  }
  const token = state.cameraRenderToken;
  const payload = await api("/api/cameras");
  if (token !== state.cameraRenderToken) return;
  state.cameras = payload.cameras;
  renderActiveCameraGrid(state.cameras);
  renderTask1CameraStatus();
}

function syncTeleopCameraExpandButton() {
  const button = $("#teleop-camera-expand");
  if (!button) return;
  const expanded = document.fullscreenElement === $("#teleop-camera-grid");
  const modeLabel = state.graspAssist.enabled ? "\u6293\u53d6\u753b\u9762" : "\u753b\u9762";
  button.textContent = expanded ? `F11 \u9000\u51fa${modeLabel}\u653e\u5927` : `F11 \u653e\u5927${modeLabel}`;
  button.setAttribute("aria-pressed", String(expanded));
}
async function toggleTeleopCameraFullscreen() {
  const grid = $("#teleop-camera-grid");
  if (!grid) return;
  if (document.fullscreenElement === grid) {
    await document.exitFullscreen();
    return;
  }
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  }
  if (!grid.requestFullscreen) {
    throw new Error("当前浏览器不支持三画面放大");
  }
  await grid.requestFullscreen();
}

function bindEvents() {
  loadGraspSettings();
  bindTask1Events();
  $$("[data-view]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });

  $("#refresh-status")?.addEventListener("click", () => refreshStatus().catch((err) => toast(err.message)));
  $("#refresh-jobs")?.addEventListener("click", () => refreshJobs().catch((err) => toast(err.message)));
  $("#refresh-artifacts").addEventListener("click", () => refreshArtifacts().catch((err) => toast(err.message)));
  $("#refresh-ideal-runs").addEventListener("click", () => refreshIdealRuns().catch((err) => toast(err.message)));
  $$("[data-refresh-cameras]").forEach((button) => {
    button.addEventListener("click", () => refreshCameras({ forceReconnect: true }).catch((err) => toast(err.message)));
  });
  $("#teleop-camera-expand")?.addEventListener("click", () => {
    toggleTeleopCameraFullscreen().catch((err) => toast(err.message));
  });
  $("#teleop-grasp-mode")?.addEventListener("click", () => toggleGraspAssist());
  $("#teleop-camera-grid")?.addEventListener("click", handleGraspCameraClick);
  ["#grasp-tip-offset", "#grasp-warning-distance", "#grasp-danger-distance"].forEach((selector) => {
    $(selector)?.addEventListener("change", updateGraspSettings);
  });
  $("#grasp-reset-target")?.addEventListener("click", () => {
    const camera = (state.cameras || DEFAULT_CAMERAS).find(
      (item) => item.name === state.graspAssist.mainCamera,
    );
    const target = defaultGraspTarget(camera);
    state.graspAssist.targetX = target.x;
    state.graspAssist.targetY = target.y;
    applyGraspAssistStateToDom();
    pollGraspDepth();
  });
  document.addEventListener("fullscreenchange", () => {
    syncTeleopCameraExpandButton();
    scheduleGraspOverlayGeometrySync();
  });
  window.addEventListener("resize", scheduleGraspOverlayGeometrySync);
  $("#teleop-suction-toggle")?.addEventListener("click", toggleSuction);
  document.addEventListener("keydown", (event) => {
    if (event.repeat || shouldIgnoreShortcut(event)) return;
    if (state.activeView !== "teleop-view") return;
    if (event.key === "F11") {
      event.preventDefault();
      event.stopPropagation();
      toggleTeleopCameraFullscreen().catch((err) => toast(err.message));
      return;
    }
    const key = event.key.toLowerCase();
    if (key === "n") {
      event.preventDefault();
      refreshCameras({ forceReconnect: true })
        .then(() => toast("三路摄像头已重新连接"))
        .catch((err) => toast(err.message));
      return;
    }
    if (key === "g") {
      event.preventDefault();
      toggleGraspAssist();
      return;
    }
    if (key === "f") {
      event.preventDefault();
      toggleSuction();
    }
  });

  $("#ideal-map-canvas").addEventListener("click", handleIdealMapClick);
  $("#ideal-map-mode-2d")?.addEventListener("click", () => setIdealMapMode("2d"));
  $("#ideal-map-mode-3d")?.addEventListener("click", () => setIdealMapMode("3d"));
  $("#scene3d-fixture")?.addEventListener("change", () => {
    state.idealNav.selectedScene3dFixtureId = $("#scene3d-fixture").value;
    renderCompetitionScene3D();
    syncIdealMap3D();
  });
  $("#scene3d-pick-list")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-scene3d-pick-id]");
    if (button) selectScene3DPickPointById(button.dataset.scene3dPickId);
  });
  $("#save-scene3d-fixture")?.addEventListener("click", () => {
    saveCompetitionScene3DFixture().catch((err) => toast(err.message));
  });
  $("#capture-scene3d-pregrasp")?.addEventListener("click", () => {
    captureScene3DPregraspPose().catch((err) => toast(err.message));
  });
  $("#preview-scene3d-pregrasp")?.addEventListener("click", () => {
    previewRecordedPregrasp().catch((err) => toast(err.message));
  });
  $("#execute-scene3d-pregrasp")?.addEventListener("click", () => {
    executeRecordedPregrasp().catch((err) => toast(err.message));
  });
  ["#ideal-start-x", "#ideal-start-y"].forEach((selector) => {
    $(selector)?.addEventListener("change", () => {
      setPlanningStart({
        x: Number($("#ideal-start-x").value),
        y: Number($("#ideal-start-y").value),
      });
    });
  });
  ["#ideal-goal-x", "#ideal-goal-y"].forEach((selector) => {
    $(selector).addEventListener("change", () => {
      setIdealTarget({
        x: Number($("#ideal-goal-x").value),
        y: Number($("#ideal-goal-y").value),
      });
    });
  });
  $("#new-ideal-route").addEventListener("click", newIdealRoute);
  $("#new-ideal-point").addEventListener("click", () => addIdealRouteWaypoint());
  $("#remove-ideal-waypoint").addEventListener("click", removeLastIdealRouteWaypoint);
  $("#clear-ideal-route").addEventListener("click", clearIdealRoute);
  $("#ideal-avoid-obstacles").addEventListener("change", () => {
    setIdealAvoidObstaclesEnabled(Boolean($("#ideal-avoid-obstacles").checked));
    updateIdealPathSummary();
    renderIdealAvoidanceSummary();
  });
  $("#map-plan-avoid-obstacles").addEventListener("change", () => {
    setIdealAvoidObstaclesEnabled(Boolean($("#map-plan-avoid-obstacles").checked));
    updateIdealPathSummary();
    renderIdealAvoidanceSummary();
  });
  [
    "#ideal-scan-topic",
    "#ideal-stop-distance",
    "#ideal-front-deg",
    "#ideal-stale-s",
    "#ideal-replan-radius",
    "#ideal-max-replans",
  ].forEach((selector) => {
    $(selector).addEventListener("change", () => {
      syncIdealAvoidanceInputs();
      renderIdealAvoidanceSummary();
    });
  });
  $("#ideal-auto-replan").addEventListener("change", () => {
    syncIdealAvoidanceInputs();
    renderIdealAvoidanceSummary();
  });
  $("#execute-ideal-path").addEventListener("click", () => {
    executeIdealPath().catch((err) => toast(err.message));
  });
  $("#replay-current-path").addEventListener("click", replayCurrentPath);
  $("#run-map-plan").addEventListener("click", () => {
    previewMapPlan({ syncRoute: true }).catch((err) => toast(err.message));
  });
  $("#execute-map-plan").addEventListener("click", () => {
    executeMapPlanRoute().catch((err) => toast(err.message));
  });
  $("#clear-map-plan").addEventListener("click", clearMapPlanPreview);
  $("#competition-waypoint-list")?.addEventListener("click", (event) => {
    const row = event.target.closest("[data-waypoint-id]");
    if (row) selectCompetitionWaypoint(row.dataset.waypointId);
  });
  $("#save-competition-waypoint")?.addEventListener("click", () => {
    saveCompetitionWaypoint(false).catch((err) => toast(err.message));
  });
  $("#update-competition-waypoint")?.addEventListener("click", () => {
    saveCompetitionWaypoint(true).catch((err) => toast(err.message));
  });
  $("#delete-competition-waypoint")?.addEventListener("click", () => {
    deleteCompetitionWaypoint().catch((err) => toast(err.message));
  });
  $("#plan-to-competition-waypoint")?.addEventListener("click", () => {
    planToCompetitionWaypoint().catch((err) => toast(err.message));
  });
  $("#execute-competition-waypoint")?.addEventListener("click", () => {
    executeCompetitionWaypoint().catch((err) => toast(err.message));
  });
  $("#competition-waypoint-auto-lift")?.addEventListener("change", () => {
    updateCompetitionWaypointControls();
  });
  $("#competition-waypoint-yaw")?.addEventListener("change", () => {
    const yawDeg = Number($("#competition-waypoint-yaw").value);
    if (Number.isFinite(yawDeg)) {
      state.idealNav.finalYaw = normalizeAngle(yawDeg * Math.PI / 180);
      updateIdealPathSummary();
      drawIdealMap();
    }
  });
  $("#reset-ideal-start").addEventListener("click", () => {
    resetIdealStart().catch((err) => toast(err.message));
  });
  $("#auto-ideal-localize")?.addEventListener("click", () => {
    autoLocateIdealStart().catch((err) => toast(err.message));
  });
  $("#reset-ideal-pose").addEventListener("click", () => {
    resetIdealPose().catch((err) => toast(err.message));
  });
  $("#load-ideal-run").addEventListener("click", () => {
    loadIdealRun($("#ideal-run-select").value).catch((err) => toast(err.message));
  });
  $("#replay-ideal-run").addEventListener("click", replayLoadedRun);

  $("#estop-button").addEventListener("click", () => {
    fireGlobalEstop().catch(() => {});
  });
  $("#estop-reset-button")?.addEventListener("click", () => {
    resetGlobalEstop().catch(() => {});
  });
  $("#teleop-safety-open-generalist")?.addEventListener("click", () => {
    switchView("generalist-view");
  });

  $$("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const mode = $("#airbot-mode")?.value || "execution";
      const action = button.dataset.action;
      const routes = {
        "airbot-check": ["/api/services/airbot/check", { mode }],
        "airbot-start": ["/api/services/airbot/start", { mode }],
        "airbot-stop": ["/api/services/airbot/stop", {}],
        "zero-check": ["/api/services/zerograsp/check", {}],
        "zero-start": ["/api/services/zerograsp/start", {}],
        "zero-stop": ["/api/services/zerograsp/stop", {}],
        "teleop-check": ["/api/services/teleop/check", {}],
        "teleop-start": ["/api/services/teleop/start", { lead_url: $("#teleop-lead-url").value }],
        "teleop-stop": ["/api/services/teleop/stop", {}],
        "arms-reset": ["/api/arms/reset", { stop_teleop: true, release_seconds: 2.0, settle_seconds: 3.0, speed: "DEFAULT" }],
      };
      const [path, body] = routes[action];
      postJob(path, body).catch((err) => toast(err.message));
    });
  });

  $$("[data-move]").forEach((button) => {
    button.addEventListener("click", () => {
      postJob("/api/base/move", {
        command: button.dataset.move,
        speed: Number($("#base-speed").value),
        yaw_speed: Number($("#base-yaw").value),
        duration: Number($("#base-duration").value),
      }).catch((err) => toast(err.message));
    });
  });

  $$("[data-lift]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.lift;
      if (action === "status") {
        postJob("/api/lift/status", { timeout_s: 1 }).catch((err) => toast(err.message));
        return;
      }
      postJob("/api/lift/action", {
        action,
        position: Number($("#lift-position").value),
      }).catch((err) => toast(err.message));
    });
  });

  ["#grasp-prompt", "#grasp-arm", "#grasp-strategy", "#grasp-speed", "#grasp-endpoint"].forEach((selector) => {
    $(selector).addEventListener("change", () => {
      if (!samePreparedGrasp(state.graspFlow.payload, graspFormPayload())) {
        invalidateRealityApproval("抓取参数已变化，请重新检查并预演。 ");
      }
    });
  });

  $("#start-grasp").addEventListener("click", async () => {
    const payload = graspFormPayload();
    resetGraspFlow({
      phase: "validating",
      payload,
    });
    renderGraspFlow();
    try {
      const result = await startJob("/api/grasp/validate", payload);
      state.graspFlow.validateJobId = result.job.id;
      toast(`Started: ${result.job.label}`);
      await refreshJobs();
    } catch (err) {
      state.graspFlow.phase = "validation-failed";
      renderGraspFlow(err.message);
      toast(err.message);
    }
  });

  $("#run-grasp-reality").addEventListener("click", async () => {
    const flow = state.graspFlow;
    if (!flow.readyForReality || !flow.payload) return;
    const blockedReason = realityBlockedReason(flow);
    if (blockedReason) {
      renderGraspFlow(blockedReason);
      toast(blockedReason);
      return;
    }
    flow.phase = "executing-real";
    flow.readyForReality = false;
    renderGraspFlow();
    try {
      const result = await startJob("/api/grasp/run", {
        ...flow.payload,
        dry_run: false,
        record_video: $("#grasp-record").checked,
        confirm: "RUN",
      });
      flow.realJobId = result.job.id;
      toast(`Started: ${result.job.label}`);
      await refreshJobs();
    } catch (err) {
      flow.phase = "real-failed";
      renderGraspFlow(err.message);
      toast(err.message);
    }
  });

  $("#job-list")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-job-id]");
    if (!button) return;
    state.selectedJobId = button.dataset.jobId;
    refreshSelectedJob().catch((err) => toast(err.message));
  });
}


/* Removed obsolete overridden function. */


window.addEventListener("rucwone-map3d-ready", () => {
  syncIdealMap3D();
  window.RucWoneMap3D?.setVisible(state.idealNav.mapMode === "3d");
});

function buildIdealRouteBody(waypoints, finalYaw, overrides = {}) {
  const avoidance = currentIdealAvoidanceConfig();
  const avoidObstacles = overrides.avoidObstacles ?? Boolean($("#ideal-avoid-obstacles")?.checked);
  const body = {
    use_memory: overrides.useMemory ?? true,
    waypoints: waypoints.map((point) => ({
      x: Number(point.x),
      y: Number(point.y),
      ...(Number.isFinite(point.yaw) ? { yaw: Number(point.yaw) } : {}),
      ...(point.motion === "reverse" ? { motion: "reverse" } : { motion: "forward" }),
    })),
    final_yaw: finalYaw,
    max_speed: overrides.maxSpeed ?? Number($("#ideal-max-speed").value),
    max_yaw_speed: overrides.maxYawSpeed ?? Number($("#ideal-max-yaw").value),
    timeout: overrides.timeout ?? 30,
    pos_tol: overrides.posTol ?? 0.02,
    yaw_tol: overrides.yawTol ?? 0.04,
    avoid_obstacles: avoidObstacles,
    scan_topic: overrides.scanTopic ?? avoidance.scanTopic,
    obstacle_stop_distance: overrides.stopDistance ?? avoidance.stopDistance,
    obstacle_front_deg: overrides.frontDeg ?? avoidance.frontDeg,
    obstacle_stale_s: overrides.staleS ?? avoidance.staleS,
  };
  if (!body.use_memory && overrides.initialPose) {
    body.initial_x = Number(overrides.initialPose.x);
    body.initial_y = Number(overrides.initialPose.y);
    body.initial_yaw = Number(overrides.initialPose.yaw) * 180 / Math.PI;
  }
  if (Number.isFinite(overrides.completionLiftHeightMm)) {
    body.completion_lift_height_mm = Math.round(overrides.completionLiftHeightMm);
  }
  if (overrides.pregraspPose) {
    const pregrasp = overrides.pregraspPose;
    body.pregrasp_pose = {
      fixture_id: String(pregrasp.fixtureId || ""),
      point_id: String(pregrasp.pointId || ""),
      restore_gripper: Boolean(pregrasp.restoreGripper),
    };
  }
  return body;
}

function updateIdealExecutionControls() {
  [
    "#execute-ideal-path",
    "#new-ideal-route",
    "#new-ideal-point",
    "#remove-ideal-waypoint",
    "#clear-ideal-route",
    "#reset-ideal-start",
    "#auto-ideal-localize",
    "#reset-ideal-pose",
    "#ideal-start-x",
    "#ideal-start-y",
    "#replay-current-path",
    "#ideal-goal-x",
    "#ideal-goal-y",
    "#ideal-max-speed",
    "#ideal-max-yaw",
    "#ideal-avoid-obstacles",
    "#ideal-scan-topic",
    "#ideal-stop-distance",
    "#ideal-front-deg",
    "#ideal-stale-s",
    "#ideal-auto-replan",
    "#ideal-replan-radius",
    "#ideal-max-replans",
    "#ideal-absolute-trim-enabled",
    "#run-ideal-absolute-trim",
  ].forEach((selector) => {
    const control = $(selector);
    if (control) control.disabled = state.idealNav.isRunning;
  });
  const executeButton = $("#execute-ideal-path");
  if (executeButton) {
    executeButton.disabled = state.idealNav.isRunning || state.estop.latched !== false;
    executeButton.textContent = state.idealNav.isRunning && state.idealNav.executionMode === "absolute-trim"
      ? "绝对精修中"
      : (state.idealNav.isRunning ? "执行中" : "执行路径");
  }
  const absoluteTrimButton = $("#run-ideal-absolute-trim");
  if (absoluteTrimButton) {
    absoluteTrimButton.disabled = state.idealNav.isRunning
      || state.estop.latched !== false
      || !currentIdealFinalGoal();
    absoluteTrimButton.textContent = state.idealNav.absoluteTrimActive ? "绝对精修中" : "终点绝对精修";
  }
  updateCompetitionWaypointControls();
}

async function startIdealRouteExecution(waypoints, options = {}) {
  if (state.estop.latched !== false) {
    throw new Error("Web Console STOP 状态尚未解除，导航启动已阻止");
  }
  const finalYaw = Number.isFinite(options.finalYaw) ? Number(options.finalYaw) : state.idealNav.finalYaw;
  const finalGoalPoint = options.finalGoal || waypoints[waypoints.length - 1];
  const finalGoal = finalGoalPoint
    ? {
      x: Number(finalGoalPoint.x),
      y: Number(finalGoalPoint.y),
      yaw: Number.isFinite(finalGoalPoint.yaw) ? Number(finalGoalPoint.yaw) : finalYaw,
    }
    : null;
  const avoidObstacles = options.avoidObstacles ?? Boolean($("#ideal-avoid-obstacles")?.checked);
  state.idealNav.avoidObstacles = avoidObstacles;
  state.idealNav.isRunning = true;
  state.idealNav.executionMode = options.executionMode || "route";
  state.idealNav.absoluteTrimActive = state.idealNav.executionMode === "absolute-trim";
  state.idealNav.routeJobId = null;
  state.idealNav.executionGoal = finalGoal;
  state.idealNav.executionWaypoints = waypoints.map((point) => ({ ...point }));
  updateIdealExecutionControls();
  renderIdealAvoidanceSummary();
  renderIdealAbsoluteSummary();
  try {
    const payload = await startJob(
      "/api/ideal-nav/route",
      buildIdealRouteBody(waypoints, finalYaw, {
        avoidObstacles,
        useMemory: options.useMemory,
        initialPose: options.initialPose,
        ...(options.routeOverrides || {}),
      }),
    );
    state.idealNav.routeJobId = payload.job?.id || null;
    state.idealNav.pendingRouteRunIds = payload.run_ids || [];
    state.idealNav.activeRouteIndex = 0;
    state.idealNav.replayPoint = options.replayPoint || { ...state.idealNav.pose };
    drawIdealMap();
    startLiveIdealRoute(payload.run_ids || [], payload.job.id, { waypoints, finalGoal, finalYaw });
    if (!options.silentToast) {
      toast(`已开始：${payload.job.label}`);
    }
    await refreshJobs();
    return payload;
  } catch (err) {
    state.idealNav.isRunning = false;
    state.idealNav.executionMode = "route";
    state.idealNav.absoluteTrimActive = false;
    state.idealNav.routeJobId = null;
    state.idealNav.pendingRouteRunIds = [];
    state.idealNav.activeRouteIndex = 0;
    updateIdealExecutionControls();
    renderIdealAvoidanceSummary();
    renderIdealAbsoluteSummary();
    drawIdealMap();
    throw err;
  }
}

async function maybeStartAbsoluteTrimAfterRoute(finalGoal) {
  const trim = currentIdealAbsoluteTrimConfig();
  if (!trim.enabled || !finalGoal) return false;
  const absolute = await refreshIdealAbsoluteState({ silent: true });
  if (!absolute?.pose) {
    toast("主路线完成，已跳过终点绝对精修：没有读到绝对位姿");
    return false;
  }
  if (!absolute?.gate?.ok) {
    toast(`主路线完成，已跳过终点绝对精修：${formatAbsoluteReason(absolute.gate?.reasons?.[0])}`);
    return false;
  }
  const delta = absoluteTrimDelta(absolute.pose, finalGoal);
  if (delta.distance <= trim.posTol && delta.yawError <= trim.yawTol) {
    toast("主路线完成，绝对位姿已在精修阈值内");
    return false;
  }
  await startAbsoluteTrimToGoal(finalGoal, {
    absoluteState: absolute,
    silentToast: true,
  });
  toast(`主路线完成，开始终点绝对精修（${delta.distance.toFixed(2)}m / ${(delta.yawError * 180 / Math.PI).toFixed(1)}deg）`);
  return true;
}

async function finishLiveIdealRoute(success) {
  const completedMode = state.idealNav.executionMode || "route";
  const finalGoal = state.idealNav.executionGoal ? { ...state.idealNav.executionGoal } : currentIdealFinalGoal();
  const competitionAction = state.idealNav.competitionWaypointExecution;
  window.clearInterval(state.idealNav.liveTimer);
  state.idealNav.liveTimer = null;
  state.idealNav.isRunning = false;
  state.idealNav.pendingRouteRunIds = [];
  state.idealNav.activeRouteIndex = 0;
  state.idealNav.routeJobId = null;
  state.idealNav.executionGoal = null;
  state.idealNav.executionWaypoints = [];
  state.idealNav.recoveryBusy = false;
  if (success && completedMode === "route") {
    try {
      const trimStarted = await maybeStartAbsoluteTrimAfterRoute(finalGoal);
      if (trimStarted) {
        await refreshIdealRuns();
        await refreshIdealState({ force: true });
        await refreshJobs();
        return;
      }
    } catch (err) {
      state.idealNav.absoluteTrimActive = false;
      toast(`终点绝对精修启动失败：${err.message}`);
    }
  }
  state.idealNav.executionMode = "route";
  state.idealNav.absoluteTrimActive = false;
  if (success) {
    state.idealNav.route = [];
    state.idealNav.routeActive = false;
    state.idealNav.routeMode = "normal";
  }
  updateIdealPathSummary();
  updateIdealMemorySummary();
  updateIdealExecutionControls();
  renderIdealAvoidanceSummary();
  renderIdealAbsoluteSummary();
  renderIdealRouteList();
  drawIdealMap();
  await refreshIdealRuns();
  await refreshIdealState({ force: true });
  await refreshJobs();
  if (completedMode === "competition-waypoint") {
    state.idealNav.competitionWaypointExecution = null;
    if (success) {
      const completedLiftHeight = competitionWaypointLiftHeight({
        lift_height_mm: competitionAction?.liftHeightMm,
      });
      const heightText = completedLiftHeight === null
        ? ""
        : `，升降已到 ${completedLiftHeight} mm`;
      toast(`已到达“${competitionAction?.name || "比赛点位"}”并完成最终朝向${heightText}`);
    } else {
      toast(`“${competitionAction?.name || "比赛点位"}”作业未完成，请查看终端中的导航或升降错误`);
    }
  }
}

function switchView(viewId) {
  state.activeView = viewId;
  renderEstopState();
  state.cameraRenderToken += 1;
  $$("[data-view-panel], .workspace-view").forEach((panel) => {
    panel.hidden = panel.id !== viewId;
  });
  $$("[data-view]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === viewId);
  });
  clearInactiveCameraGrids();
  if (state.cameras) {
    renderActiveCameraGrid(state.cameras);
  } else {
    renderCameraPlaceholders(DEFAULT_CAMERAS);
    refreshCameras().catch((err) => toast(err.message));
  }
  syncGraspDepthPolling();
  if (viewId === "map-view") {
    drawIdealMap();
    refreshIdealState().catch((err) => toast(err.message));
    refreshIdealRuns().catch((err) => toast(err.message));
    refreshIdealAbsoluteState({ silent: true }).catch(() => {});
    refreshCompetitionWaypoints().catch((err) => toast(err.message));
    refreshCompetitionScene3D().catch((err) => toast(err.message));
    refreshPregraspPoses().catch((err) => toast(err.message));
  }
  if (viewId === "agentic-view") {
    refreshArtifacts().catch((err) => toast(err.message));
    refreshJobs().catch((err) => toast(err.message));
    refreshTask1Current({ silent: true }).catch(() => {});
  }
  if (viewId === "jobs-view") refreshJobs().catch((err) => toast(err.message));
  if (viewId === "system-view") refreshStatus().catch((err) => toast(err.message));
}

function bindAbsoluteTrimEvents() {
  $("#ideal-absolute-trim-enabled")?.addEventListener("change", () => {
    renderIdealAbsoluteSummary();
    updateIdealExecutionControls();
  });
  $("#refresh-ideal-absolute")?.addEventListener("click", () => {
    refreshIdealAbsoluteState({ successToast: true }).catch((err) => toast(err.message));
  });
  $("#run-ideal-absolute-trim")?.addEventListener("click", () => {
    const goal = currentIdealFinalGoal();
    if (!goal) {
      toast("没有可用于绝对精修的终点");
      return;
    }
    startAbsoluteTrimToGoal(goal).catch((err) => toast(err.message));
  });
}

async function boot() {
  bindEvents();
  bindAbsoluteTrimEvents();
  syncIdealAvoidanceInputs();
  syncPlanningStartInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  syncIdealTargetInputs();
  updateIdealPathSummary();
  updateIdealMemorySummary();
  updateIdealExecutionControls();
  renderIdealRouteList();
  renderMapPlanResult();
  syncCompetitionWaypointForm(null);
  renderCompetitionWaypoints();
  renderIdealAvoidanceSummary();
  renderIdealAbsoluteSummary();
  renderTask1();
  renderSuctionState();
  drawIdealMap();
  if (state.activeView === "map-view") {
    previewMapPlan({ silent: true }).catch((err) => toast(err.message));
  }
  renderGraspFlow();
  switchView(state.activeView);
  try {
    await refreshStatus();
    await Promise.allSettled([
      refreshTask1Config({ silent: true }),
      refreshTask1Current({ silent: true }),
      refreshEstopStatus({ silent: true }),
      refreshSuctionStatus(),
    ]);
    if (state.activeView === "jobs-view") await refreshJobs();
    if (state.activeView === "agentic-view") {
      await Promise.allSettled([refreshArtifacts(), refreshJobs()]);
    }
    if (state.activeView === "map-view") {
      await refreshIdealState();
      await refreshIdealRuns();
      await refreshIdealAbsoluteState({ silent: true });
      await refreshCompetitionWaypoints();
      await refreshCompetitionScene3D();
      await refreshPregraspPoses();
    }
  } catch (err) {
    toast(err.message);
  }
  startTask1Polling();
  window.setInterval(() => refreshStatus().catch(() => {}), 4000);
  window.setInterval(() => refreshEstopStatus({ silent: true }).catch(() => {}), 2000);
  window.setInterval(() => {
    if (["jobs-view", "agentic-view"].includes(state.activeView)) refreshJobs().catch(() => {});
  }, 2500);
  window.setInterval(() => {
    if (state.activeView === "agentic-view") refreshArtifacts().catch(() => {});
  }, 6000);
  window.setInterval(() => {
    if (state.activeView === "map-view") refreshIdealState().catch(() => {});
  }, 1500);
  window.setInterval(() => {
    if (state.activeView === "map-view") refreshIdealRuns().catch(() => {});
  }, 5000);
}

boot();
