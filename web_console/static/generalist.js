(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  if (!byId("generalist-view")) return;

  const armServicePanel = byId("teleop-arm-service-panel");
  const armServiceDock = byId("teleop-arm-service-dock");
  if (armServicePanel && armServiceDock) {
    const previousContainer = armServicePanel.parentElement;
    armServiceDock.replaceWith(armServicePanel);
    if (previousContainer?.classList.contains("generalist-service-area")) {
      previousContainer.remove();
    }
  }

  const generalist = {
    // Fail closed until the backend explicitly reports latched:false.
    halted: true,
    estopState: "loading",
    estopConfirmed: false,
    driving: false,
    driveCommand: "",
    driveTimer: null,
    baseHoldId: "",
    baseCommandSequence: 0,
    baseRequestCounts: new Map(),
    heldDriveKeys: new Set(),
    chargingInterlock: null,
    chargingInterlockBusy: false,
    baseBusy: false,
    liftBusy: false,
    fastTelemetryController: null,
    baseControlReserveUntil: 0,
    liftHolding: "",
    liftHeldKey: "",
    liftHoldId: "",
    health: {
      backend: false,
      hardware: false,
      arms: false,
      followerService: false,
      lead: false,
      cameras: false,
      base: false,
      lift: false,
      follow: false,
    },
    odom: null,
    liftPosition: null,
    liftTarget: null,
    armServices: null,
    armServiceJobs: new Map(),
    armSafetyRefreshBusy: false,
    temperatureAlert: null,
    temperatureAlertActive: false,
    jointLimitAlert: null,
    jointLimitAlertActive: false,
    jointLimitAlertSeverity: "",
    baseSettings: {
      loaded: false,
      editing: false,
      dirty: false,
      saving: false,
      saved: {
        speed: 0.05,
        yaw_speed: 0.15,
        duration: 0.35,
      },
    },
  };

  const basePresets = {
    slow: { speed: 0.03, yaw_speed: 0.10, duration: 0.30 },
    medium: { speed: 0.05, yaw_speed: 0.15, duration: 0.35 },
    fast: { speed: 0.08, yaw_speed: 0.25, duration: 0.40 },
  };
  // Refresh held motion well inside the backend deadman lease. Keeping this
  // independent of the saved pulse duration avoids intermittent zero-velocity
  // gaps when camera rendering or telemetry briefly delays a browser timer.
  const BASE_HOLD_HEARTBEAT_MS = 150;
  const BASE_HEARTBEAT_REQUEST_TIMEOUT_MS = 350;
  const driveCommandLabels = {
    forward: "前进",
    backward: "后退",
    "turn-left": "左转",
    "turn-right": "右转",
    "forward-left": "前进＋左转",
    "forward-right": "前进＋右转",
    "backward-left": "后退＋左转",
    "backward-right": "后退＋右转",
  };
  const LIFT_MIN_POSITION_MM = 0;
  const LIFT_MAX_POSITION_MM = 300;
  const LIFT_HOLD_LEASE_S = 0.9;
  const LIFT_HOLD_HEARTBEAT_MS = 150;
  const ARM_SERVICE_JOB_POLL_MS = 400;
  const ARM_SERVICE_JOB_TIMEOUT_MS = 120000;
  const ARM_SERVICE_TERMINAL_STATES = new Set([
    "success",
    "failed",
    "cancelled",
    "canceled",
  ]);

  const html = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));

  const dot = (id, status) => {
    const node = byId(id);
    if (!node) return;
    node.className = `status-dot ${status === true ? "ok" : status === false ? "bad" : "muted"}`;
  };

  const serviceState = (id, state, text) => {
    const node = byId(id);
    if (!node) return;
    node.className = `service-inline-state is-${state}`;
    node.textContent = text;
  };

  const miniRows = (rows) => rows.map(([key, value]) => (
    `<div><span>${html(key)}</span><span>${html(value)}</span></div>`
  )).join("");

  const numeric = (value, fallback = null) => {
    const result = Number(value);
    return Number.isFinite(result) ? result : fallback;
  };

  function renderReadiness() {
    document.querySelectorAll(
      "[data-gen-move], [data-gen-lift], [data-gen-lift-preset], "
      + "[data-gen-lift-hold], [data-gen-service], [data-gen-arm-service]",
    ).forEach((button) => {
      const action = button.dataset.genMove
        || button.dataset.genLift
        || button.dataset.genService
        || button.dataset.genArmService
        || (button.dataset.genLiftPreset ? "lift-preset" : "")
        || (button.dataset.genLiftHold ? "lift-hold" : "");
      const safeWhileBlocked = action === "stop" || action.endsWith("-stop") || action.endsWith("-check");
      const chargingBlocksBase = Boolean(
        button.dataset.genMove
        && button.dataset.genMove !== "stop"
        && generalist.chargingInterlock !== false,
      );
      if (!safeWhileBlocked) button.disabled = generalist.halted || chargingBlocksBase;
    });
    const entries = Object.entries(generalist.health);
    const readyCount = entries.filter(([, ready]) => ready).length;
    const requiredCount = entries.length;
    const badge = byId("gen-readiness");
    if (!badge) return;
    if (generalist.halted) {
      badge.className = "readiness-badge blocked";
      if (["loading", "unknown"].includes(generalist.estopState)) {
        badge.textContent = "安全状态未知 · 禁止运动";
      } else if (generalist.estopState === "stopping") {
        badge.textContent = "STOP 请求中";
      } else if (generalist.estopConfirmed) {
        badge.textContent = "STOP 已确认 · 操作已停止";
      } else {
        badge.textContent = "STOP 未确认 · 使用物理急停";
      }
      return;
    }
    if (!generalist.health.backend) {
      badge.className = "readiness-badge blocked";
      badge.textContent = "后端离线";
      return;
    }
    if (generalist.chargingInterlock === true) {
      badge.className = "readiness-badge warning";
      badge.textContent = "充电联锁已开启 · 底盘禁用";
      return;
    }
    if (readyCount === requiredCount) {
      badge.className = "readiness-badge ready";
      badge.textContent = "遥操作就绪";
      return;
    }
    badge.className = generalist.health.hardware ? "readiness-badge warning" : "readiness-badge blocked";
    badge.textContent = `${readyCount}/${requiredCount} 项就绪`;
  }

  function renderFollowStatus(running, unavailable = false) {
    const isRunning = Boolean(running) && !unavailable;
    generalist.health.follow = isRunning;
    dot("gen-follow-dot", unavailable ? null : isRunning);
    byId("gen-follow-summary").textContent = unavailable
      ? "Master Follow 状态连接中断"
      : isRunning
        ? "Master Follow 正在运行"
        : "Master Follow 未运行";
    byId("gen-follow-detail").innerHTML = miniRows([
      ["tmux", "airbot_teleop"],
      ["Follow 进程", unavailable ? "状态未知" : isRunning ? "运行中" : "未运行"],
    ]);
    serviceState(
      "gen-follow-inline-status",
      unavailable ? "blocked" : isRunning ? "ready" : "idle",
      unavailable ? "状态未知" : isRunning ? "运行中" : "未运行",
    );
  }

  function renderSystemStatus(status) {
    const config = status.config || {};
    generalist.health.backend = true;
    generalist.health.hardware = Boolean(config.enable_hardware_actions);
    renderFollowStatus(Boolean(status.teleop?.tmux));

    dot("gen-safety-dot", generalist.health.hardware);
    byId("gen-safety-summary").textContent = generalist.health.hardware
      ? "硬件命令已开启"
      : "安全模式：硬件命令关闭";
    byId("gen-safety-detail").innerHTML = miniRows([
      ["Profile", config.profile || "-"],
      ["Role", config.role || "-"],
      ["ROS Master", config.ros_master_uri || "-"],
    ]);

    const maxLinear = numeric(config.max_linear_mps, 0.2);
    const maxAngular = numeric(config.max_angular_radps, 0.5);
    byId("gen-base-speed").max = String(maxLinear);
    byId("gen-base-yaw").max = String(maxAngular);
    renderReadiness();
  }

  function armEndpointLabel(arm) {
    const can = arm.can_interface && arm.can_interface !== "remote"
      ? `${arm.can_interface}${arm.can_up ? "↑" : "↓"} · `
      : "";
    return `${can}${arm.host}:${arm.port} ${arm.port_open ? "✓" : "×"}`;
  }

  function renderArmTemperatureAlert(alert = {}) {
    const root = byId("arm-temperature-alert");
    const detail = byId("arm-temperature-alert-detail");
    if (!root || !detail) return;

    generalist.temperatureAlert = alert;
    const active = alert.active === true;
    const events = Array.isArray(alert.events) ? alert.events : [];
    const eventText = events.map((event) => {
      const robot = event.robot_id && event.robot_id !== "unknown"
        ? event.robot_id
        : "机械臂";
      const joint = event.joint && event.joint !== "unknown"
        ? event.joint
        : "关节";
      const hasTemperature = event.temperature_c !== null
        && event.temperature_c !== undefined
        && event.temperature_c !== ""
        && Number.isFinite(Number(event.temperature_c));
      const temperature = hasTemperature
        ? ` ${Number(event.temperature_c).toFixed(0)}°C`
        : " 线圈过热";
      return `${robot} · ${joint}${temperature}`;
    }).join("；");

    const alertText = active
      ? `${eventText ? `${eventText}。` : ""}${alert.message || "立即停止操作并等待机械臂冷却。"}`
      : "";
    root.hidden = !active;
    detail.textContent = alertText;
    const fullscreenRoot = byId("arm-temperature-fullscreen-alert");
    const fullscreenDetail = byId("arm-temperature-fullscreen-alert-detail");
    const cameraGrid = byId("teleop-camera-grid");
    if (fullscreenRoot) fullscreenRoot.hidden = !active;
    if (fullscreenDetail) fullscreenDetail.textContent = alertText;
    cameraGrid?.classList.toggle("has-arm-temperature-alert", active);
    document.body.classList.toggle("has-arm-temperature-alert", active);
    if (active && !generalist.temperatureAlertActive) {
      toast("机械臂高温：立即停止操作");
    }
    generalist.temperatureAlertActive = active;
  }

  window.rucSyncArmTemperatureAlert = () => {
    renderArmTemperatureAlert(generalist.temperatureAlert || {});
  };

  function renderArmJointLimitAlert(alert = {}) {
    const root = byId("arm-joint-limit-alert");
    const title = byId("arm-joint-limit-alert-title");
    const detail = byId("arm-joint-limit-alert-detail");
    if (!root || !title || !detail) return;

    generalist.jointLimitAlert = alert;
    const active = alert.active === true;
    const severity = alert.severity === "danger" ? "danger" : "warning";
    const events = Array.isArray(alert.events) ? alert.events : [];
    const sideNames = { left: "左从臂", right: "右从臂" };
    const directionNames = { lower: "下限", upper: "上限" };
    const eventText = events.slice(0, 4).map((event) => {
      const arm = sideNames[event.side]
        || (event.robot_id && event.robot_id !== "unknown" ? event.robot_id : "机械臂");
      const joint = event.joint || "关节";
      if (event.state === "limit_triggered" || event.state === "over_limit") {
        return `${arm} · ${joint} 已触发限位`;
      }
      const remaining = Number(event.remaining_deg);
      const distance = Number.isFinite(remaining) ? `仅剩 ${remaining.toFixed(1)}°` : "接近限位";
      const direction = directionNames[event.direction] || "限位";
      return `${arm} · ${joint} 距${direction}${distance}`;
    }).join("；");
    const overflow = events.length > 4 ? `；另有 ${events.length - 4} 项` : "";
    const isDanger = active && severity === "danger";
    const alertTitle = isDanger
      ? "机械臂限位：立即停止当前动作"
      : "机械臂接近限位：请减速回退";
    const alertText = active
      ? `${eventText}${overflow}${eventText ? "。" : ""}${alert.message || ""}`
      : "";

    root.hidden = !active;
    root.classList.toggle("is-danger", isDanger);
    title.textContent = alertTitle;
    detail.textContent = alertText;

    const fullscreenRoot = byId("arm-joint-limit-fullscreen-alert");
    const fullscreenTitle = byId("arm-joint-limit-fullscreen-alert-title");
    const fullscreenDetail = byId("arm-joint-limit-fullscreen-alert-detail");
    const cameraGrid = byId("teleop-camera-grid");
    if (fullscreenRoot) {
      fullscreenRoot.hidden = !active;
      fullscreenRoot.classList.toggle("is-danger", isDanger);
    }
    if (fullscreenTitle) fullscreenTitle.textContent = alertTitle;
    if (fullscreenDetail) fullscreenDetail.textContent = alertText;
    cameraGrid?.classList.toggle("has-arm-joint-limit-alert", active);
    cameraGrid?.classList.toggle("has-arm-joint-limit-danger", isDanger);
    document.body.classList.toggle("has-arm-joint-limit-alert", active);

    const escalated = isDanger && generalist.jointLimitAlertSeverity !== "danger";
    if (active && (!generalist.jointLimitAlertActive || escalated)) {
      toast(alertTitle);
    }
    generalist.jointLimitAlertActive = active;
    generalist.jointLimitAlertSeverity = active ? severity : "";
  }

  window.rucSyncArmSafetyAlerts = () => {
    renderArmTemperatureAlert(generalist.temperatureAlert || {});
    renderArmJointLimitAlert(generalist.jointLimitAlert || {});
  };

  function renderArmServices(payload) {
    generalist.armServices = payload;
    renderArmTemperatureAlert(payload.temperature_alert);
    renderArmJointLimitAlert(payload.joint_limit_alert);
    renderFollowStatus(Boolean(payload.follow?.running));
    const follower = payload.follower || {};
    const followerArms = follower.arms || [];
    generalist.health.followerService = Boolean(follower.ready);
    dot(
      "gen-follower-service-dot",
      follower.ready ? true : followerArms.some((arm) => arm.can_present) ? null : false,
    );
    byId("gen-follower-service-summary").textContent = follower.ready
      ? "两个从臂服务已就绪"
      : `${followerArms.filter((arm) => arm.port_open).length}/2 端口监听`;
    byId("gen-follower-service-detail").innerHTML = miniRows([
      ["左从臂", followerArms[0] ? armEndpointLabel(followerArms[0]) : "缺少状态"],
      ["右从臂", followerArms[1] ? armEndpointLabel(followerArms[1]) : "缺少状态"],
      ["tmux", follower.session_running ? follower.session : "未运行"],
    ]);
    const followerPortCount = followerArms.filter((arm) => arm.port_open).length;
    serviceState(
      "gen-follower-inline-status",
      follower.ready ? "ready" : followerPortCount ? "pending" : "idle",
      follower.ready ? "已就绪" : followerPortCount ? `${followerPortCount}/2 就绪` : "未就绪",
    );

    const lead = payload.lead || {};
    const leadArms = lead.arms || [];
    const modeNames = {
      "local-can": "本机 CAN",
      "remote-airbot": "远程 AIRBOT",
      unconfigured: "未配置",
    };
    generalist.health.lead = Boolean(lead.control_ready);
    const leadVisual = lead.control_ready
      ? true
      : lead.endpoint_ready || lead.local_can_pair
        ? null
        : false;
    dot("gen-lead-service-dot", leadVisual);
    if (lead.control_ready) {
      byId("gen-lead-service-summary").textContent = "两个主臂控制端点已就绪";
    } else if (lead.mode === "unconfigured") {
      byId("gen-lead-service-summary").textContent = "无本地主臂 CAN，等待远程配置";
    } else {
      byId("gen-lead-service-summary").textContent = "主臂端点未就绪";
    }
    const leadRows = [["模式", modeNames[lead.mode] || lead.mode || "-"]];
    if (leadArms.length) {
      leadRows.push(
        ["左主臂", armEndpointLabel(leadArms[0])],
        ["右主臂", armEndpointLabel(leadArms[1])],
      );
    } else {
      leadRows.push(["本机 CAN", "未检测到主臂专用 CAN 对"]);
    }
    if (lead.ssh_target) leadRows.push(["远程启动", lead.ssh_target]);
    if (lead.mode === "remote-airbot") {
      leadRows.push([
        "远端 CAN",
        `左=${lead.remote_left_iface || "can0"} · 右=${lead.remote_right_iface || "can1"}`,
      ]);
    }
    leadRows.push(["tmux", lead.session_running ? lead.session : "未运行"]);
    byId("gen-lead-service-detail").innerHTML = miniRows(leadRows);
    serviceState(
      "gen-lead-inline-status",
      lead.control_ready ? "ready" : lead.endpoint_ready || lead.local_can_pair ? "pending" : "idle",
      lead.control_ready ? "已就绪" : lead.mode === "unconfigured" ? "待配置" : "未就绪",
    );
    const armServiceActionLabels = {
      start: "启动中",
      stop: "停止中",
      check: "检查中",
    };
    generalist.armServiceJobs.forEach((job, service) => {
      serviceState(
        `gen-${service}-inline-status`,
        "pending",
        armServiceActionLabels[job.action] || "执行中",
      );
    });

    const remoteHostInput = byId("gen-remote-lead-host");
    const sshInput = byId("gen-lead-ssh-target");
    const leftCanInput = byId("gen-remote-left-can");
    const rightCanInput = byId("gen-remote-right-can");
    if (!remoteHostInput.value && lead.remote_host) remoteHostInput.value = lead.remote_host;
    if (!sshInput.value && lead.ssh_target) sshInput.value = lead.ssh_target;
    if (document.activeElement !== leftCanInput) {
      leftCanInput.value = lead.remote_left_iface || "can0";
    }
    if (document.activeElement !== rightCanInput) {
      rightCanInput.value = lead.remote_right_iface || "can1";
    }
    if (!byId("gen-lead-url").value && lead.remote_host) {
      byId("gen-lead-url").value = lead.remote_host;
    }
    byId("gen-lead-protocol-note").textContent = lead.mode === "remote-airbot"
      ? lead.ssh_target
        ? "主臂按钮会先通过 SSH 启动远端 can0/can1 Docker，再检查 50050/50052。"
        : "未填写 SSH 目标：主臂按钮只检查远端 50050/50052，远端服务需预先启动。"
      : "按 REMOTE_TELEOP.md：四个端口都就绪后，最后单独启动 Follow。";
    renderReadiness();
  }

  function renderArm(name, arm) {
    const prefix = `gen-${name}-arm`;
    const connected = Boolean(arm?.connected);
    const ready = Boolean(arm?.teleop_ready);
    dot(`${prefix}-dot`, connected && ready ? true : connected ? null : false);
    const summary = byId(`${prefix}-summary`);
    const detail = byId(`${prefix}-detail`);
    if (!arm) {
      summary.textContent = "没有状态";
      detail.innerHTML = "";
      return;
    }
    if (!connected) {
      summary.textContent = arm.port_open ? "端口通，但 SDK 未连接" : "gRPC 端口未监听";
    } else if (ready) {
      summary.textContent = "执行臂遥操作就绪";
    } else if (arm.command_ready) {
      summary.textContent = "可下发命令，但非跟随模式";
    } else {
      summary.textContent = `驱动状态：${arm.driver_state || "UNKNOWN"}`;
    }
    const eef = Array.isArray(arm.eef_width) && arm.eef_width.length
      ? `${(Number(arm.eef_width[0]) * 1000).toFixed(0)} mm`
      : "-";
    detail.innerHTML = miniRows([
      ["端口", `${arm.host}:${arm.port}`],
      ["Driver", arm.driver_state || "-"],
      ["Mode", arm.control_mode || "-"],
      ["夹爪", eef],
      ...(arm.error ? [["错误", arm.error]] : []),
    ]);
  }

  function renderArmStatus(payload) {
    const arms = payload.arms || [];
    renderArm("left", arms.find((item) => item.name === "left"));
    renderArm("right", arms.find((item) => item.name === "right"));
    generalist.health.arms = Boolean(payload.teleop_ready);
    renderReadiness();
  }

  function renderCameras(cameras) {
    const enabled = cameras.filter((camera) => camera.enabled);
    const configured = enabled.filter((camera) => camera.source);
    const expectedCount = enabled.length;
    const live = configured.filter((camera) => camera.runtime?.live);
    const verified = configured.filter((camera) => {
      if (camera.runtime?.live) return true;
      const age = camera.runtime?.last_live_age_s;
      return typeof age === "number";
    });
    const failed = configured.filter((camera) => {
      const runtime = camera.runtime || {};
      return Number(runtime.subscribers || 0) > 0
        && !runtime.live
        && !["", "starting", "idle"].includes(String(runtime.message || "").toLowerCase());
    });
    const allConfigured = expectedCount > 0 && configured.length === expectedCount;
    generalist.health.cameras = (
      allConfigured
      && verified.length === expectedCount
      && failed.length === 0
    );
    dot(
      "gen-camera-dot",
      failed.length
        ? false
        : generalist.health.cameras
          ? true
          : allConfigured
            ? null
            : false,
    );
    if (failed.length) {
      byId("gen-camera-summary").textContent = `${failed.length} 路画面异常`;
    } else if (live.length === expectedCount && expectedCount > 0) {
      byId("gen-camera-summary").textContent = `${expectedCount}/${expectedCount} 路正在传输`;
    } else if (live.length > 0 && verified.length === expectedCount) {
      byId("gen-camera-summary").textContent = `${live.length}/${expectedCount} 路正在传输，其余已验证`;
    } else if (verified.length === expectedCount && expectedCount > 0) {
      byId("gen-camera-summary").textContent = `${expectedCount}/${expectedCount} 路已验证，当前未拉流`;
    } else if (allConfigured) {
      byId("gen-camera-summary").textContent = `${expectedCount}/${expectedCount} 路已配置，等待画面验证`;
    } else {
      byId("gen-camera-summary").textContent = `${configured.length}/${expectedCount} 路已配置`;
    }
    const priority = ["left", "front", "right"];
    const orderedNames = [
      ...priority.filter((name) => cameras.some((camera) => camera.name === name)),
      ...cameras.map((camera) => camera.name).filter((name) => !priority.includes(name)),
    ];
    byId("gen-camera-detail").innerHTML = miniRows(
      orderedNames.map((name) => {
        const camera = cameras.find((item) => item.name === name);
        if (!camera) return [name, "缺少配置"];
        const runtime = camera.runtime || {};
        const age = runtime.last_live_age_s;
        const hasVerifiedFrame = typeof age === "number";
        let status = "待 Teleop 拉流";
        if (!camera.enabled || !camera.source) {
          status = "未配置";
        } else if (runtime.live) {
          status = "正在传输";
        } else if (
          Number(runtime.subscribers || 0) > 0
          && !["", "starting", "idle"].includes(String(runtime.message || "").toLowerCase())
        ) {
          status = runtime.message || "拉流失败";
        } else if (hasVerifiedFrame && age <= 300) {
          status = `最近已验证 · ${Math.round(age)}s`;
        } else if (hasVerifiedFrame) {
          status = `上次正常 · ${Math.max(1, Math.round(age / 60))}min`;
        } else if (runtime.message === "starting") {
          status = "正在连接";
        }
        return [
          name,
          status,
        ];
      }),
    );
    renderReadiness();
  }

  function readPose(data) {
    const pose = data?.pose || data?.position || {};
    const velocity = data?.velocity || data?.twist || {};
    return {
      x: numeric(pose.x, 0),
      y: numeric(pose.y, 0),
      yaw: numeric(pose.yaw ?? pose.theta, 0),
      vx: numeric(velocity.x ?? velocity.vx, 0),
      vy: numeric(velocity.y ?? velocity.vy, 0),
      wz: numeric(velocity.yaw ?? velocity.wz, 0),
      stamp: numeric(data?.stamp, 0),
    };
  }

  function renderBaseState(payload) {
    renderChargingInterlock(payload.charging_interlock);
    const pose = readPose(payload.data);
    generalist.odom = pose;
    generalist.health.base = true;
    dot("gen-base-dot", true);
    byId("gen-base-summary").textContent = "里程计数据正常";
    byId("gen-base-detail").innerHTML = miniRows([
      ["位置", `${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}`],
      ["航向", `${(pose.yaw * 180 / Math.PI).toFixed(1)}°`],
      ["速度", `${Math.hypot(pose.vx, pose.vy).toFixed(2)} m/s`],
    ]);
    byId("gen-odom-x").textContent = pose.x.toFixed(2);
    byId("gen-odom-y").textContent = pose.y.toFixed(2);
    byId("gen-odom-yaw").textContent = (pose.yaw * 180 / Math.PI).toFixed(1);
    byId("gen-odom-speed").textContent = Math.hypot(pose.vx, pose.vy).toFixed(2);
    byId("gen-odom-wz").textContent = pose.wz.toFixed(2);
    renderReadiness();
  }

  function renderBaseError(error) {
    generalist.health.base = false;
    dot("gen-base-dot", false);
    byId("gen-base-summary").textContent = "底盘状态不可用";
    byId("gen-base-detail").innerHTML = miniRows([["错误", error.message || error]]);
    ["gen-odom-x", "gen-odom-y", "gen-odom-yaw", "gen-odom-speed", "gen-odom-wz"]
      .forEach((id) => { byId(id).textContent = "--"; });
    renderReadiness();
  }

  function renderChargingInterlock(interlock) {
    const charging = interlock?.charging;
    if (typeof charging !== "boolean") return;
    const becameBlocked = generalist.chargingInterlock !== true && charging;
    generalist.chargingInterlock = charging;
    const toggle = byId("base-charging-toggle");
    toggle.checked = charging;
    toggle.disabled = generalist.chargingInterlockBusy;
    const panel = byId("base-charging-interlock");
    panel.classList.toggle("is-charging", charging);
    byId("base-charging-help").textContent = charging
      ? "充电联锁已开启：底盘移动命令会被后端拒绝。断开充电连接后再关闭此开关。"
      : "连接充电桩时请开启；开启后后端会拒绝所有底盘移动命令。";
    if (becameBlocked) stopDrive({ sendStop: false });
    renderReadiness();
  }

  async function setChargingInterlock(charging) {
    if (generalist.chargingInterlockBusy) return;
    generalist.chargingInterlockBusy = true;
    byId("base-charging-toggle").disabled = true;
    try {
      const payload = await api("/api/base/charging-interlock", {
        method: "POST",
        body: JSON.stringify({ charging }),
      });
      renderChargingInterlock(payload.charging_interlock);
      toast(charging ? "充电联锁已开启，底盘移动已禁用" : "充电联锁已关闭");
    } catch (error) {
      byId("base-charging-toggle").checked = generalist.chargingInterlock === true;
      throw error;
    } finally {
      generalist.chargingInterlockBusy = false;
      byId("base-charging-toggle").disabled = false;
    }
  }

  async function refreshChargingInterlock() {
    const payload = await api("/api/base/charging-interlock");
    renderChargingInterlock(payload.charging_interlock);
    return payload.charging_interlock;
  }

  function liftPositionFrom(data) {
    if (Array.isArray(data) && data.length) {
      return numeric(data[data.length - 1]?.position_mm);
    }
    return numeric(data?.position_mm ?? data?.position);
  }

  function renderLiftState(payload) {
    const position = liftPositionFrom(payload.data);
    if (position === null) throw new Error("升降杆返回数据中没有 position_mm");
    generalist.liftPosition = position;
    if (generalist.liftTarget === null) setLiftTarget(position);
    generalist.health.lift = true;
    dot("gen-lift-dot", true);
    byId("gen-lift-summary").textContent = `当前位置 ${position.toFixed(0)} mm`;
    byId("gen-lift-detail").innerHTML = miniRows([
      ["实际高度", `${position.toFixed(0)} mm`],
      ["目标高度", `${numeric(byId("gen-lift-target").value, 0).toFixed(0)} mm`],
    ]);
    ["gen-lift-position-overview", "gen-lift-position-live"].forEach((id) => {
      const node = byId(id);
      if (node) node.textContent = position.toFixed(0);
    });
    renderReadiness();
  }

  function renderLiftError(error) {
    generalist.health.lift = false;
    dot("gen-lift-dot", false);
    byId("gen-lift-summary").textContent = "升降杆状态不可用";
    byId("gen-lift-detail").innerHTML = miniRows([["错误", error.message || error]]);
    ["gen-lift-position-overview", "gen-lift-position-live"].forEach((id) => {
      const node = byId(id);
      if (node) node.textContent = "--";
    });
    renderReadiness();
  }

  function findCollision(value, seen = new Set()) {
    if (!value || typeof value !== "object" || seen.has(value)) return null;
    seen.add(value);
    const collisionSignals = new Set([
      "collisionwarning",
      "collision_warning",
      "collision_shutdown",
      "collision_shutdown_",
      "collision_active",
    ]);
    for (const [key, item] of Object.entries(value)) {
      const lower = key.toLowerCase();
      if (
        collisionSignals.has(lower)
        && ["boolean", "number", "string"].includes(typeof item)
      ) {
        const active = item === true
          || (typeof item === "number" && item !== 0)
          || (typeof item === "string" && !["", "0", "false", "none", "ok"].includes(item.toLowerCase()));
        if (active) return { active: true, key, value: item };
      }
      const nested = findCollision(item, seen);
      if (nested?.active) return nested;
    }
    return { active: false };
  }

  function renderDiagnostics(payload) {
    const collision = findCollision(payload.data) || { active: false };
    const node = byId("gen-collision");
    const root = node.closest(".telemetry-alert");
    node.textContent = collision.active ? "⚠ ACTIVE" : "OK";
    root.classList.toggle("is-danger", collision.active);
    if (collision.active) {
      byId("gen-base-summary").textContent = "碰撞告警，请停止移动";
      dot("gen-base-dot", false);
      generalist.health.base = false;
      stopDrive();
    }
    renderReadiness();
  }

  async function refreshArmServices() {
    const payload = await api("/api/arm-services/status");
    renderArmServices(payload);
    return payload;
  }

  async function refreshArmSafety() {
    if (generalist.armSafetyRefreshBusy) return null;
    generalist.armSafetyRefreshBusy = true;
    try {
      return await refreshArmServices();
    } finally {
      generalist.armSafetyRefreshBusy = false;
    }
  }

  async function refreshSystem() {
    const [statusResult, armsResult, camerasResult, armServicesResult] = await Promise.allSettled([
      api("/api/status"),
      api("/api/arms/status"),
      api("/api/cameras"),
      api("/api/arm-services/status"),
    ]);
    if (statusResult.status === "fulfilled") {
      renderSystemStatus(statusResult.value.status);
    } else {
      generalist.health.backend = false;
      renderFollowStatus(false, true);
      renderReadiness();
    }
    if (armsResult.status === "fulfilled") {
      renderArmStatus(armsResult.value);
    } else {
      generalist.health.arms = false;
      renderArm("left", null);
      renderArm("right", null);
    }
    if (camerasResult.status === "fulfilled") {
      renderCameras(camerasResult.value.cameras || []);
    } else {
      generalist.health.cameras = false;
      dot("gen-camera-dot", false);
      byId("gen-camera-summary").textContent = "摄像头状态读取失败";
    }
    if (armServicesResult.status === "fulfilled") {
      renderArmServices(armServicesResult.value);
    } else {
      generalist.health.followerService = false;
      generalist.health.lead = false;
      renderFollowStatus(false, true);
      dot("gen-follower-service-dot", false);
      dot("gen-lead-service-dot", false);
      byId("gen-follower-service-summary").textContent = "从臂服务状态读取失败";
      byId("gen-lead-service-summary").textContent = "主臂服务状态读取失败";
    }
    renderReadiness();
  }

  function cancelFastTelemetry() {
    generalist.fastTelemetryController?.abort();
    generalist.fastTelemetryController = null;
    generalist.baseBusy = false;
    generalist.liftBusy = false;
  }

  async function refreshFastTelemetry() {
    // Persistent MJPEG streams occupy most browser connections. While the base
    // is moving, reserve every remaining lane for MOVE/STOP and lift hold.
    if (
      generalist.driving
      || Date.now() < generalist.baseControlReserveUntil
      || generalist.baseBusy
      || generalist.liftBusy
    ) return;
    const controller = new AbortController();
    generalist.fastTelemetryController = controller;
    generalist.baseBusy = true;
    generalist.liftBusy = true;
    const liftTelemetryPaused = Boolean(generalist.liftHolding);
    let baseResult;
    let liftResult;
    try {
      [baseResult, liftResult] = await Promise.allSettled([
        api("/api/base/state", { signal: controller.signal }),
        liftTelemetryPaused
          ? Promise.resolve(null)
          : api("/api/lift/state?timeout_s=0.25", { signal: controller.signal }),
      ]);
    } finally {
      if (generalist.fastTelemetryController === controller) {
        generalist.fastTelemetryController = null;
        generalist.baseBusy = false;
        generalist.liftBusy = false;
      }
    }
    if (controller.signal.aborted) return;
    if (baseResult.status === "fulfilled") renderBaseState(baseResult.value);
    else renderBaseError(baseResult.reason);
    if (liftTelemetryPaused) {
      return;
    }
    if (liftResult.status === "fulfilled") {
      try {
        renderLiftState(liftResult.value);
      } catch (error) {
        renderLiftError(error);
      }
    } else {
      renderLiftError(liftResult.reason);
    }
  }

  async function refreshDiagnostics() {
    try {
      renderDiagnostics(await api("/api/base/diagnostics"));
    } catch (error) {
      byId("gen-collision").textContent = "--";
    }
  }

  async function refreshAll(options = {}) {
    await Promise.allSettled([
      refreshSystem(),
      refreshFastTelemetry(),
      refreshDiagnostics(),
    ]);
    if (!options.silent) toast("Generalist 状态已刷新");
  }

  function baseSettingInputs() {
    return [
      byId("gen-base-speed"),
      byId("gen-base-yaw"),
      byId("gen-base-duration"),
    ];
  }

  function readBaseSettingsForm() {
    return {
      speed: numeric(byId("gen-base-speed").value, 0.05),
      yaw_speed: numeric(byId("gen-base-yaw").value, 0.15),
      duration: numeric(byId("gen-base-duration").value, 0.35),
    };
  }

  function applyBaseSettings(settings) {
    byId("gen-base-speed").value = String(settings.speed);
    byId("gen-base-yaw").value = String(settings.yaw_speed);
    byId("gen-base-duration").value = String(settings.duration);
    renderBasePresetSelection();
  }

  function sameBaseSettings(left, right) {
    return ["speed", "yaw_speed", "duration"].every(
      (key) => Math.abs(numeric(left[key], 0) - numeric(right[key], 0)) < 1e-9,
    );
  }

  function renderBasePresetSelection() {
    const current = readBaseSettingsForm();
    document.querySelectorAll("[data-base-preset]").forEach((button) => {
      button.classList.toggle(
        "is-active",
        sameBaseSettings(current, basePresets[button.dataset.basePreset] || {}),
      );
    });
  }

  function setBaseSettingsEditing(editing) {
    generalist.baseSettings.editing = editing;
    const saving = generalist.baseSettings.saving;
    byId("gen-base-panel").classList.toggle("is-settings-editing", editing && !saving);
    byId("gen-base-panel").classList.toggle("is-settings-saving", saving);
    baseSettingInputs().forEach((input) => {
      input.readOnly = !editing;
      input.disabled = !editing || saving;
    });
    document.querySelectorAll("[data-base-preset]").forEach((button) => {
      button.disabled = !editing || saving;
    });
    byId("base-settings-edit").disabled = editing || saving;
    byId("base-settings-save").disabled = !editing || saving;
    const stateNode = byId("base-settings-state");
    if (saving) {
      stateNode.textContent = "正在保存";
      stateNode.className = "base-settings-state is-saving";
    } else if (!editing) {
      stateNode.textContent = "参数已锁定";
      stateNode.className = "base-settings-state";
    } else if (generalist.baseSettings.dirty) {
      stateNode.textContent = "有未保存修改";
      stateNode.className = "base-settings-state is-dirty";
    } else {
      stateNode.textContent = "编辑中";
      stateNode.className = "base-settings-state is-editing";
    }
    renderBasePresetSelection();
  }

  function updateBaseSettingsDirty() {
    generalist.baseSettings.dirty = !sameBaseSettings(
      readBaseSettingsForm(),
      generalist.baseSettings.saved,
    );
    setBaseSettingsEditing(generalist.baseSettings.editing);
  }

  async function loadBaseSettings() {
    const payload = await api("/api/base/settings");
    generalist.baseSettings.saved = {
      speed: numeric(payload.settings?.speed, 0.05),
      yaw_speed: numeric(payload.settings?.yaw_speed, 0.15),
      duration: numeric(payload.settings?.duration, 0.35),
    };
    generalist.baseSettings.loaded = true;
    generalist.baseSettings.dirty = false;
    applyBaseSettings(generalist.baseSettings.saved);
    setBaseSettingsEditing(false);
  }

  function beginBaseSettingsEdit() {
    if (generalist.baseSettings.saving) return;
    if (generalist.driving) stopDrive({ always: true });
    applyBaseSettings(generalist.baseSettings.saved);
    generalist.baseSettings.dirty = false;
    setBaseSettingsEditing(true);
    byId("gen-base-speed").focus();
  }

  function revertBaseSettings(reason = "") {
    if (!generalist.baseSettings.editing) return false;
    applyBaseSettings(generalist.baseSettings.saved);
    generalist.baseSettings.dirty = false;
    setBaseSettingsEditing(false);
    if (reason) toast(reason);
    return true;
  }

  async function saveBaseSettings() {
    if (generalist.baseSettings.saving) return;
    const invalid = baseSettingInputs().find((input) => !input.checkValidity());
    if (invalid) {
      toast("参数不在允许范围内，请修改后再保存");
      invalid.reportValidity();
      invalid.focus();
      return;
    }
    generalist.baseSettings.saving = true;
    setBaseSettingsEditing(true);
    try {
      const payload = await api("/api/base/settings", {
        method: "POST",
        body: JSON.stringify(readBaseSettingsForm()),
      });
      generalist.baseSettings.saved = {
        speed: numeric(payload.settings?.speed, 0.05),
        yaw_speed: numeric(payload.settings?.yaw_speed, 0.15),
        duration: numeric(payload.settings?.duration, 0.35),
      };
      generalist.baseSettings.loaded = true;
      generalist.baseSettings.dirty = false;
      applyBaseSettings(generalist.baseSettings.saved);
      generalist.baseSettings.saving = false;
      setBaseSettingsEditing(false);
      byId("base-settings-edit").focus();
      toast("底盘参数已保存，输入框已锁定");
    } catch (error) {
      generalist.baseSettings.saving = false;
      setBaseSettingsEditing(true);
      throw error;
    }
  }

  function basePayload(command, options = {}) {
    const payload = {
      command,
      speed: numeric(byId("gen-base-speed").value, 0.05),
      yaw_speed: numeric(byId("gen-base-yaw").value, 0.15),
      duration: numeric(byId("gen-base-duration").value, 0.35),
    };
    if (options.holdId) {
      payload.hold_id = options.holdId;
      payload.sequence = options.sequence;
    }
    return payload;
  }

  async function sendBaseMove(command, options = {}) {
    const request = {
      method: "POST",
      body: JSON.stringify(basePayload(command, options)),
      keepalive: command === "stop",
      cache: "no-store",
      priority: "high",
    };
    const controller = options.timeoutMs ? new AbortController() : null;
    const timeout = controller
      ? window.setTimeout(() => controller.abort(), options.timeoutMs)
      : null;
    if (controller) request.signal = controller.signal;
    try {
      const payload = await api("/api/base/move", request);
      if (payload.job) state.selectedJobId = payload.job.id;
      return payload;
    } finally {
      if (timeout !== null) window.clearTimeout(timeout);
    }
  }

  function newBaseHoldId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }

  function sendBaseHeartbeat(command, holdId, options = {}) {
    if (!holdId || generalist.baseHoldId !== holdId) return;
    const inFlight = generalist.baseRequestCounts.get(holdId) || 0;
    if (inFlight > 0 && !options.force) return;
    generalist.baseCommandSequence += 1;
    const sequence = generalist.baseCommandSequence;
    generalist.baseRequestCounts.set(holdId, inFlight + 1);
    sendBaseMove(command, {
      holdId,
      sequence,
      timeoutMs: BASE_HEARTBEAT_REQUEST_TIMEOUT_MS,
    }).then((payload) => {
      const controller = payload.controller || {};
      if (
        generalist.baseHoldId === holdId
        && controller.stale === true
        && controller.active === false
      ) {
        stopDrive({ sendStop: false });
        toast("底盘控制心跳已超时，请松开方向键后重新按下");
      }
    }).catch((error) => {
      if (error?.name === "AbortError") return;
      markDriveError(error, holdId, sequence);
    }).finally(() => {
      const remaining = (generalist.baseRequestCounts.get(holdId) || 1) - 1;
      if (remaining > 0) {
        generalist.baseRequestCounts.set(holdId, remaining);
      } else {
        generalist.baseRequestCounts.delete(holdId);
      }
    });
  }

  function markDriveError(error, holdId, sequence) {
    // A delayed failure from an older hold must not cancel a newer command.
    if (holdId && generalist.baseHoldId !== holdId) return;
    if (sequence < generalist.baseCommandSequence) return;
    stopDrive({ sendStop: false });
    toast(`底盘命令失败：${error.message}`);
  }

  function startDrive(command, button = null) {
    if (generalist.halted) {
      toast("急停状态下不能移动");
      return false;
    }
    if (generalist.chargingInterlock !== false) {
      toast(generalist.chargingInterlock === true
        ? "底盘正在充电，移动已被联锁"
        : "充电联锁状态读取中，暂不能移动");
      return false;
    }
    revertBaseSettings("未保存的底盘参数已回滚");
    if (command === "stop") {
      stopDrive();
      return true;
    }
    cancelFastTelemetry();
    if (!generalist.baseHoldId) {
      generalist.baseHoldId = newBaseHoldId();
      generalist.baseCommandSequence = 0;
    }
    const holdId = generalist.baseHoldId;
    generalist.driving = true;
    generalist.driveCommand = command;
    document.querySelectorAll("[data-gen-move]").forEach((item) => item.classList.remove("is-held"));
    const activeButton = button || document.querySelector(`[data-gen-move="${command}"]`);
    if (activeButton) activeButton.classList.add("is-held");
    byId("gen-drive-chip").className = "control-chip active";
    byId("gen-drive-chip").textContent = `正在下发 ${driveCommandLabels[command] || command}`;
    // A direction change is latency-sensitive and may overtake one older
    // heartbeat. Sequence ordering on the backend makes that safe.
    sendBaseHeartbeat(command, holdId, { force: true });
    window.clearInterval(generalist.driveTimer);
    generalist.driveTimer = window.setInterval(() => {
      if (
        generalist.driving
        && generalist.baseHoldId === holdId
        && generalist.driveCommand === command
      ) {
        sendBaseHeartbeat(command, holdId);
      }
    }, BASE_HOLD_HEARTBEAT_MS);
    return true;
  }

  function stopDrive(options = {}) {
    const wasDriving = generalist.driving;
    const holdId = generalist.baseHoldId;
    const sequence = holdId ? generalist.baseCommandSequence + 1 : null;
    if (!options.preserveHeldKeys) generalist.heldDriveKeys.clear();
    generalist.driving = false;
    generalist.driveCommand = "";
    generalist.baseHoldId = "";
    generalist.baseCommandSequence = 0;
    // Do not let telemetry reclaim the freed browser lanes while the zero and
    // a possible quick re-press are still traversing the control path.
    generalist.baseControlReserveUntil = Date.now() + 600;
    window.clearInterval(generalist.driveTimer);
    generalist.driveTimer = null;
    document.querySelectorAll("[data-gen-move]").forEach((item) => item.classList.remove("is-held"));
    byId("gen-drive-chip").className = "control-chip";
    byId("gen-drive-chip").textContent = "待命";
    if ((wasDriving || options.always) && options.sendStop !== false) {
      sendBaseMove(
        "stop",
        { holdId, sequence },
      ).catch((error) => {
        if (generalist.health.hardware) toast(`底盘停止失败：${error.message}`);
      });
    }
  }

  function commandFromHeldDriveKeys() {
    const held = generalist.heldDriveKeys;
    const forward = held.has("KeyW") || held.has("ArrowUp");
    const backward = held.has("KeyS") || held.has("ArrowDown");
    const left = held.has("KeyA") || held.has("ArrowLeft");
    const right = held.has("KeyD") || held.has("ArrowRight");
    const linear = Number(forward) - Number(backward);
    const yaw = Number(left) - Number(right);

    if (linear > 0 && yaw > 0) return "forward-left";
    if (linear > 0 && yaw < 0) return "forward-right";
    if (linear < 0 && yaw > 0) return "backward-left";
    if (linear < 0 && yaw < 0) return "backward-right";
    if (linear > 0) return "forward";
    if (linear < 0) return "backward";
    if (yaw > 0) return "turn-left";
    if (yaw < 0) return "turn-right";
    return "stop";
  }

  function syncHeldDriveKeys() {
    const command = commandFromHeldDriveKeys();
    if (command === "stop") {
      // Preserve opposing keys so releasing either one immediately resumes
      // the remaining direction. A lost focus/page still clears every key.
      stopDrive({ preserveHeldKeys: true });
      return;
    }
    if (!startDrive(command)) generalist.heldDriveKeys.clear();
  }

  function setLiftTarget(value) {
    const target = Math.min(
      LIFT_MAX_POSITION_MM,
      Math.max(LIFT_MIN_POSITION_MM, numeric(value, LIFT_MIN_POSITION_MM)),
    );
    generalist.liftTarget = target;
    byId("gen-lift-target").value = String(Math.round(target));
    return target;
  }

  function renderLiftHolding() {
    document.querySelectorAll("[data-gen-lift-hold]").forEach((button) => {
      button.classList.toggle(
        "is-held",
        button.dataset.genLiftHold === generalist.liftHolding,
      );
    });
    const status = byId("gen-lift-hold-status");
    if (!status) return;
    if (generalist.liftHolding === "up") {
      status.className = "lift-hold-status is-moving";
      status.textContent = "正在上升";
    } else if (generalist.liftHolding === "down") {
      status.className = "lift-hold-status is-moving";
      status.textContent = "正在下降";
    } else {
      status.className = "lift-hold-status";
      status.textContent = "待命";
    }
  }

  async function liftAction(action, explicitTarget = null, options = {}) {
    if (action !== "stop" && generalist.halted) {
      throw new Error("安全状态未明确复位，升降杆动作已阻止");
    }
    const body = { action };
    if (["up", "down"].includes(action)) {
      body.pulse_s = Number(options.pulse_s ?? 0.15);
    }
    if (action === "hold") {
      body.direction = options.direction;
      body.lease_s = Number(options.lease_s ?? LIFT_HOLD_LEASE_S);
      body.hold_id = options.holdId;
    }
    if (action === "stop" && options.holdId) {
      body.hold_id = options.holdId;
    }
    if (action === "goto") {
      body.position = setLiftTarget(
        explicitTarget ?? byId("gen-lift-target").value,
      );
    }
    const payload = await api("/api/lift/action", {
      method: "POST",
      body: JSON.stringify(body),
    });
    if (payload.job) state.selectedJobId = payload.job.id;
    if (!options.silent) {
      window.dispatchEvent(new CustomEvent("ruc:terminal-refresh"));
    }
    if (!options.silent && !payload.cancelled) toast(`升降杆 ${action} 已下发`);
    if (!generalist.liftHolding) {
      window.setTimeout(() => refreshFastTelemetry(), 400);
    }
    return payload;
  }

  function newLiftHoldId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }

  function startLiftHold(action, button = null, key = "") {
    if (!["up", "down"].includes(action)) return;
    if (generalist.halted) {
      toast("安全状态未明确复位，不能移动升降杆");
      return;
    }
    if (generalist.liftHolding) return;
    const holdId = newLiftHoldId();
    generalist.liftHolding = action;
    generalist.liftHeldKey = key;
    generalist.liftHoldId = holdId;
    renderLiftHolding();
    const activeButton = button
      || document.querySelector(`[data-gen-lift-hold="${action}"]`);
    if (activeButton) activeButton.classList.add("is-held");
    (async () => {
      try {
        // One child process owns the serial handle for the full hold. It writes
        // STOP from its own finally when keyup, pointer release, or the backend
        // heartbeat lease cancels that process.
        while (
          generalist.liftHolding === action
          && generalist.liftHoldId === holdId
          && !generalist.halted
        ) {
          const payload = await liftAction("hold", null, {
            silent: true,
            direction: action,
            lease_s: LIFT_HOLD_LEASE_S,
            holdId,
          });
          if (liftPositionFrom(payload.hold?.telemetry) !== null) {
            renderLiftState({ data: payload.hold.telemetry });
          }
          if (payload.cancelled || payload.stale) {
            if (generalist.liftHoldId === holdId) {
              generalist.liftHolding = "";
              generalist.liftHeldKey = "";
              generalist.liftHoldId = "";
              renderLiftHolding();
            }
            break;
          }
          await new Promise((resolve) => {
            window.setTimeout(resolve, LIFT_HOLD_HEARTBEAT_MS);
          });
        }
      } catch (error) {
        if (
          generalist.liftHolding === action
          && generalist.liftHoldId === holdId
        ) {
          stopLiftHold({ always: true, silent: true }).catch(() => {});
        }
        toast(`升降杆${action === "up" ? "上升" : "下降"}失败：${error.message}`);
      }
    })();
  }

  async function stopLiftHold(options = {}) {
    const wasHolding = Boolean(generalist.liftHolding);
    const holdId = generalist.liftHoldId;
    generalist.liftHolding = "";
    generalist.liftHeldKey = "";
    generalist.liftHoldId = "";
    renderLiftHolding();
    if ((!wasHolding && !options.always) || options.sendStop === false) return null;
    const status = byId("gen-lift-hold-status");
    if (status) {
      status.className = "lift-hold-status is-stopping";
      status.textContent = "停止中";
    }
    try {
      return await liftAction("stop", null, { silent: true, holdId });
    } finally {
      if (!generalist.liftHolding) renderLiftHolding();
    }
  }

  function triggerPoseSlot(slot, button = null) {
    const target = button || document.querySelector(`[data-teleop-pose="${slot}"]`);
    document.querySelectorAll("[data-teleop-pose]").forEach((item) => {
      item.classList.remove("is-triggered");
    });
    if (target) {
      target.classList.add("is-triggered");
      window.setTimeout(() => target.classList.remove("is-triggered"), 500);
    }
    toast(`预设位姿 ${slot} 已预留，尚未绑定真实动作`);
  }

  function remoteLeadPayload() {
    return {
      remote_host: byId("gen-remote-lead-host").value.trim(),
      ssh_target: byId("gen-lead-ssh-target").value.trim(),
      remote_left_iface: byId("gen-remote-left-can").value.trim(),
      remote_right_iface: byId("gen-remote-right-can").value.trim(),
      timeout_s: 20,
    };
  }

  async function saveRemoteLeadConfig(options = {}) {
    const payload = await api("/api/arm-services/config", {
      method: "POST",
      body: JSON.stringify(remoteLeadPayload()),
    });
    renderArmServices(payload.arm_services);
    const host = payload.config?.remote_host || "";
    if (host && !byId("gen-lead-url").value.trim()) {
      byId("gen-lead-url").value = host;
    }
    if (!options.silent) toast("远程主臂配置已保存");
    return payload;
  }

  async function watchArmServiceJob(job, service, action) {
    const deadline = Date.now() + ARM_SERVICE_JOB_TIMEOUT_MS;
    let currentJob = job;
    try {
      while (
        !ARM_SERVICE_TERMINAL_STATES.has(currentJob.status)
        && Date.now() < deadline
      ) {
        await new Promise((resolve) => {
          window.setTimeout(resolve, ARM_SERVICE_JOB_POLL_MS);
        });
        const payload = await api(`/api/jobs/${encodeURIComponent(job.id)}`);
        currentJob = payload.job || currentJob;
      }

      if (generalist.armServiceJobs.get(service)?.id === job.id) {
        generalist.armServiceJobs.delete(service);
      }
      await Promise.allSettled([refreshArmServices(), refreshJobs()]);
      window.dispatchEvent(new CustomEvent("ruc:terminal-refresh"));

      const serviceLabel = service === "lead" ? "主臂" : "从臂";
      const actionLabel = {
        start: "启动",
        stop: "停止",
        check: "检查",
      }[action] || action;
      if (currentJob.status === "success") {
        toast(`${serviceLabel}${actionLabel}完成`);
      } else if (ARM_SERVICE_TERMINAL_STATES.has(currentJob.status)) {
        toast(`${serviceLabel}${actionLabel}失败，请查看运行日志`);
      } else {
        toast(`${serviceLabel}${actionLabel}仍在执行，状态将继续自动刷新`);
      }
      return currentJob;
    } catch (error) {
      if (generalist.armServiceJobs.get(service)?.id === job.id) {
        generalist.armServiceJobs.delete(service);
      }
      await refreshArmServices().catch(() => {});
      throw error;
    }
  }

  async function runArmService(command) {
    const [service, action] = command.split("-");
    if (!service || !action) return;
    if (generalist.armServiceJobs.has(service)) {
      throw new Error(`${service === "lead" ? "主臂" : "从臂"}服务操作正在执行`);
    }
    if (action === "start" && generalist.halted) {
      throw new Error("安全状态未明确复位，机械臂服务启动已阻止");
    }
    const body = service === "lead" ? remoteLeadPayload() : { timeout_s: 20 };
    const payload = await api(`/api/arm-services/${service}/${action}`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    if (payload.arm_services) renderArmServices(payload.arm_services);
    if (payload.job) {
      generalist.armServiceJobs.set(service, {
        id: payload.job.id,
        action,
      });
      renderArmServices(generalist.armServices || payload.arm_services || {});
      state.selectedJobId = payload.job.id;
      window.dispatchEvent(new CustomEvent("ruc:job-started", { detail: payload.job }));
      refreshJobs().catch(() => {});
      await watchArmServiceJob(payload.job, service, action);
      return;
    }
    await refreshArmServices();
  }

  async function runService(action) {
    const routes = {
      "teleop-check": ["/api/services/teleop/check", {}],
      "teleop-start": ["/api/services/teleop/start", { lead_url: byId("gen-lead-url").value.trim() }],
      "teleop-stop": ["/api/services/teleop/stop", {}],
    };
    const route = routes[action];
    if (!route) return;
    if (action.endsWith("-start") && generalist.halted) {
      throw new Error("安全状态未明确复位，主从遥操作启动已阻止");
    }
    const payload = await api(route[0], {
      method: "POST",
      body: JSON.stringify(route[1]),
    });
    if (payload.job) {
      state.selectedJobId = payload.job.id;
      window.dispatchEvent(new CustomEvent("ruc:job-started", { detail: payload.job }));
    }
    refreshJobs().catch(() => {});
    window.setTimeout(() => refreshSystem(), 1200);
  }

  async function fireEstop() {
    stopDrive({ always: true });
    stopLiftHold({ sendStop: false });
    if (typeof window.rucFireEstop === "function") {
      return window.rucFireEstop();
    }
    throw new Error("全局 STOP 控制器未加载；请立即使用物理急停");
  }

  function bindControls() {
    byId("gen-refresh-all").addEventListener("click", () => refreshAll());

    byId("base-settings-edit").addEventListener("click", beginBaseSettingsEdit);
    byId("base-settings-save").addEventListener("click", () => {
      saveBaseSettings().catch((error) => toast(`底盘参数保存失败：${error.message}`));
    });
    byId("base-charging-toggle").addEventListener("change", (event) => {
      setChargingInterlock(event.target.checked).catch((error) => {
        toast(`充电联锁设置失败：${error.message}`);
      });
    });
    baseSettingInputs().forEach((input) => {
      input.addEventListener("input", updateBaseSettingsDirty);
    });
    document.querySelectorAll("[data-base-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        if (!generalist.baseSettings.editing) return;
        const preset = basePresets[button.dataset.basePreset];
        if (!preset) return;
        applyBaseSettings(preset);
        updateBaseSettingsDirty();
      });
    });

    document.querySelectorAll("[data-gen-move]").forEach((button) => {
      const command = button.dataset.genMove;
      if (command === "stop") {
        button.addEventListener("click", () => stopDrive({ always: true }));
        return;
      }
      button.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        generalist.heldDriveKeys.clear();
        if (button.setPointerCapture) button.setPointerCapture(event.pointerId);
        startDrive(command, button);
      });
      [
        "pointerup",
        "pointerleave",
        "pointercancel",
        "lostpointercapture",
      ].forEach((eventName) => {
        button.addEventListener(eventName, () => {
          if (button.classList.contains("is-held")) stopDrive();
        });
      });
      button.addEventListener("contextmenu", (event) => event.preventDefault());
    });

    byId("gen-lift-target").addEventListener("change", (event) => setLiftTarget(event.target.value));
    document.querySelectorAll("[data-gen-lift-preset]").forEach((button) => {
      button.addEventListener("click", () => {
        const target = setLiftTarget(button.dataset.genLiftPreset);
        liftAction("goto", target).catch((error) => toast(`升降杆失败：${error.message}`));
      });
    });
    document.querySelectorAll("[data-gen-lift]").forEach((button) => {
      button.addEventListener("click", () => {
        const action = button.dataset.genLift;
        const request = action === "stop"
          ? stopLiftHold({ always: true })
          : liftAction(action);
        request.catch((error) => toast(`升降杆失败：${error.message}`));
      });
    });
    document.querySelectorAll("[data-gen-lift-hold]").forEach((button) => {
      const action = button.dataset.genLiftHold;
      button.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        if (button.setPointerCapture) button.setPointerCapture(event.pointerId);
        startLiftHold(action, button);
      });
      ["pointerup", "pointerleave", "pointercancel", "lostpointercapture"].forEach((eventName) => {
        button.addEventListener(eventName, () => {
          if (button.classList.contains("is-held")) {
            stopLiftHold().catch((error) => toast(`升降杆停止失败：${error.message}`));
          }
        });
      });
      button.addEventListener("contextmenu", (event) => event.preventDefault());
    });

    document.querySelectorAll("[data-gen-service]").forEach((button) => {
      button.addEventListener("click", () => {
        runService(button.dataset.genService).catch(() => {
          window.dispatchEvent(new CustomEvent("ruc:terminal-refresh"));
        });
      });
    });

    document.querySelectorAll("[data-gen-arm-service]").forEach((button) => {
      button.addEventListener("click", () => {
        runArmService(button.dataset.genArmService)
          .catch((error) => {
            toast(`机械臂服务操作失败：${error.message}`);
            window.dispatchEvent(new CustomEvent("ruc:terminal-refresh"));
          });
      });
    });
    [
      "gen-remote-lead-host",
      "gen-lead-ssh-target",
      "gen-remote-left-can",
      "gen-remote-right-can",
    ].forEach((id) => {
      byId(id).addEventListener("change", () => {
        saveRemoteLeadConfig().catch((error) => toast(`远程主臂配置失败：${error.message}`));
      });
    });
    byId("gen-remote-lead-host").addEventListener("change", (event) => {
      if (event.target.value.trim()) byId("gen-lead-url").value = event.target.value.trim();
    });

    document.querySelectorAll("[data-teleop-pose]").forEach((button) => {
      button.addEventListener("click", () => {
        triggerPoseSlot(button.dataset.teleopPose, button);
      });
    });

    document.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => {
        if (button.dataset.view !== "teleop-view") {
          stopDrive();
          stopLiftHold().catch(() => {});
          revertBaseSettings();
        }
        if (button.dataset.view === "generalist-view") {
          window.setTimeout(() => refreshSystem(), 180);
        }
      });
    });

    const driveKeyCodes = new Set([
      "KeyW",
      "KeyS",
      "KeyA",
      "KeyD",
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
    ]);
    window.addEventListener("keydown", (event) => {
      if (state.activeView !== "teleop-view") return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
      const key = event.key.toLowerCase();
      if (key === " ") {
        event.preventDefault();
        fireEstop();
        return;
      }
      const liftDirection = event.code === "KeyI"
        ? "up"
        : event.code === "KeyJ"
          ? "down"
          : "";
      if (liftDirection) {
        if (event.repeat) return;
        event.preventDefault();
        startLiftHold(liftDirection, null, event.code);
        return;
      }
      if (["1", "2", "3", "4"].includes(key) && !event.repeat) {
        event.preventDefault();
        triggerPoseSlot(key);
        return;
      }
      if (driveKeyCodes.has(event.code)) {
        event.preventDefault();
        if (event.repeat || generalist.heldDriveKeys.has(event.code)) return;
        generalist.heldDriveKeys.add(event.code);
        syncHeldDriveKeys();
      }
    });
    window.addEventListener("keyup", (event) => {
      if (driveKeyCodes.has(event.code)) {
        event.preventDefault();
        generalist.heldDriveKeys.delete(event.code);
        syncHeldDriveKeys();
      }
      if (event.code === generalist.liftHeldKey) {
        stopLiftHold().catch((error) => toast(`升降杆停止失败：${error.message}`));
      }
    });
    window.addEventListener("blur", () => {
      stopDrive();
      stopLiftHold().catch(() => {});
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        stopDrive();
        stopLiftHold().catch(() => {});
      }
    });
    window.addEventListener("pagehide", () => {
      stopDrive();
      stopLiftHold().catch(() => {});
    });
  }

  window.addEventListener("ruc:estop-state", (event) => {
    const detail = event.detail || {};
    generalist.halted = detail.latched !== false;
    generalist.estopState = String(detail.transactionState || "unknown");
    generalist.estopConfirmed = detail.stopConfirmed === true;
    if (generalist.halted) {
      stopDrive({ sendStop: false });
      stopLiftHold({ sendStop: false });
    }
    renderReadiness();
  });

  if (state.estop) {
    generalist.halted = state.estop.latched !== false;
    generalist.estopState = String(state.estop.transactionState || "loading");
    generalist.estopConfirmed = state.estop.stopConfirmed === true;
  }

  bindControls();
  refreshChargingInterlock().catch(() => {
    // Keep base motion disabled until another read proves charging is clear.
  });
  loadBaseSettings().catch((error) => {
    setBaseSettingsEditing(false);
    toast(`底盘参数读取失败，使用默认值：${error.message}`);
  });
  refreshAll({ silent: true });
  window.setInterval(() => {
    if (["generalist-view", "teleop-view"].includes(state.activeView)) refreshFastTelemetry();
  }, 1500);
  window.setInterval(() => {
    if (["generalist-view", "teleop-view"].includes(state.activeView)) {
      refreshArmSafety().catch(() => {});
    }
  }, 1000);
  window.setInterval(() => {
    if (state.activeView === "generalist-view") {
      refreshSystem();
    }
  }, 4000);
  window.setInterval(() => {
    if (state.activeView === "generalist-view") refreshDiagnostics();
  }, 5000);
})();
