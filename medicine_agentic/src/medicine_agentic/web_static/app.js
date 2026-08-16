"use strict";

const API = Object.freeze({
  health: "/api/health",
  status: "/api/status",
  profile: "/api/camera/profile",
  fixedSuctionAxisStatus: "/api/calibrations/fixed-suction-axis/status",
  fixedSuctionAxisLockMarker: "/api/calibrations/fixed-suction-axis/lock-marker",
  fixedSuctionAxisSampleCup: "/api/calibrations/fixed-suction-axis/sample-cup",
  fixedSuctionAxisCommit: "/api/calibrations/fixed-suction-axis/commit",
  frame: "/api/camera/frame.jpg",
  leftWristCamera: "/api/wrist-cameras/left/frame.jpg",
  rightWristCamera: "/api/wrist-cameras/right/frame.jpg",
  detect: "/api/detect",
  recordings: "/api/recordings",
  recordingStatus: "/api/recordings/status",
  recordingStart: "/api/recordings/start",
  recordingStop: "/api/recordings/stop",
  recordingDelete: "/api/recordings/delete",
  replayPreflight: "/api/replay/preflight",
  replayStart: "/api/replay/start",
  replayStop: "/api/replay/stop",
  baseTrajectoryStatus: "/api/base-trajectory/status",
  baseTrajectories: "/api/base-trajectories",
  baseTrajectoryRecordStart: "/api/base-trajectory/record/start",
  baseTrajectoryRecordStop: "/api/base-trajectory/record/stop",
  baseTrajectoryReplayPreflight: "/api/base-trajectory/replay/preflight",
  baseTrajectoryReplayStart: "/api/base-trajectory/replay/start",
  baseTrajectoryReplayStop: "/api/base-trajectory/replay/stop",
  actModel: "/api/act/model",
  actRolloutStatus: "/api/act/rollout/status",
  actRolloutStart: "/api/act/rollout/start",
  actRolloutStop: "/api/act/rollout/stop",
  teleopStatus: "/api/teleop/status",
  teleopStart: "/api/teleop/start",
  teleopStop: "/api/teleop/stop",
  teleopHardRestart: "/api/teleop/hard-restart",
  cartesianJogStatus: "/api/cartesian-jog/status",
  cartesianJogCapture: "/api/cartesian-jog/capture-orientation",
  cartesianJogEnable: "/api/cartesian-jog/enable",
  cartesianJogQuickEnable: "/api/cartesian-jog/quick-enable",
  cartesianJogDisable: "/api/cartesian-jog/disable",
  cartesianJogMove: "/api/cartesian-jog/move",
  cartesianJogRestoreSafe: "/api/cartesian-jog/restore-safe-vertical",
  suctionStatus: "/api/suction/status",
  suction: "/api/suction",
  task1PickStatus: "/api/task1/pick/status",
  task1Pick: "/api/task1/pick",
  task1PickSkill: "/api/skills/task1/pick-carton",
  task1MoveWatcherPose: "/api/skills/task1/move-watcher-pose",
  task1DetectCarton: "/api/skills/task1/detect-carton",
  task1DetectBoxSlots: "/api/skills/task1/detect-box-slots",
  task1ConfirmBoxSlots: "/api/skills/task1/confirm-box-slots",
  task1PickStagedTop: "/api/skills/task1/pick-staged-carton-top",
  task1PlaceInBox: "/api/skills/task1/place-in-box",
  task1PlaceFixedTrajectory:
    "/api/skills/task1/place-carton-fixed-trajectory",
  task2PickStatus: "/api/task2/pick/status",
  task2PickSkill: "/api/skills/task2/pick-carton",
  task2ResetBothHome: "/api/skills/task2/reset-both-home",
  task2MoveWatcherPose: "/api/skills/task2/move-watcher-pose",
  task2DetectCarton: "/api/skills/task2/detect-carton",
  task2MoveReadyPoses: "/api/skills/task2/move-ready-poses",
  task2MoveSubtask2InitPoses:
    "/api/skills/task2/move-subtask2-init-poses",
  task2MoveSubtask3InitPoses:
    "/api/skills/task2/move-subtask3-init-poses",
  task2DetectShippingBox: "/api/skills/task2/detect-shipping-box",
  task2PlaceShippingBox: "/api/skills/task2/place-shipping-box",
  task2PlaceShippingBoxPreflight:
    "/api/skills/task2/place-shipping-box/preflight",
  task3MoveWatcherPose: "/api/skills/task3/move-watcher-pose",
  task3DetectCarton: "/api/skills/task3/detect-carton",
  task3DetectShippingBox: "/api/skills/task3/detect-shipping-box",
  task3PlaceShippingBox: "/api/skills/task3/place-shipping-box",
  task3PlaceShippingBoxPreflight:
    "/api/skills/task3/place-shipping-box/preflight",
  task3PickStatus: "/api/task3/pick/status",
  task3PickSkill: "/api/skills/task3/pick-flat-carton",
  task3ExpandCarton: "/api/skills/task3/expand-carton",
  task1WatchDetectPick: "/api/skills/task1/watch-detect-pick",
  task2WatchDetectPick: "/api/skills/task2/watch-detect-pick",
  task3WatchDetectPick: "/api/skills/task3/watch-detect-pick",
  task1ObserveCarton: "/api/skills/task1/observe-carton",
  task2ObserveCarton: "/api/skills/task2/observe-carton",
  task3ObserveCarton: "/api/skills/task3/observe-carton",
  task1PickCachedCarton: "/api/skills/task1/pick-cached-carton",
  task2PickCachedCarton: "/api/skills/task2/pick-cached-carton",
  task3PickCachedCarton: "/api/skills/task3/pick-cached-carton",
  leftArmResetHomeSkill: "/api/skills/left-arm/reset-home",
  rightArmResetHomeSkill: "/api/skills/right-arm/reset-home",
  runtimePoseMove: "/api/runtime-poses/move",
});

const TASK_PROFILES = Object.freeze({
  task1: {
    number: "01",
    kicker: "TASK1 / 药盒装箱",
    title: "药盒装箱 · 单盒 7 步流程",
    summary:
      "首次生成顺时针90°的20槽计划，之后按持久化编号循环装箱；第5步由左吸盘吸取竖置药盒。",
    readiness: "Task1 吸取技能已接入",
    safety: "源药盒抓取与带条码顶部窄面抓取使用各自独立标定",
    detectionLabel: "识别堆叠药盒",
    detectionProfile: "task1_3x3 · Task3同款主面识别 · 1–9个可见药盒",
    primaryLabel: "Task1 · 持久化20槽循环",
    flowTitle: "Task1 技能链",
    flowNote: "单盒7步，循环18次；第5步使用下方固定工位 RGB-D + 顶部条码约束",
    available: true,
    unavailableMessage: "",
    steps: [
      [
        "左 left_watcher / 右初始位姿",
        "已接入 · 双臂并行",
        true,
        "左臂移动到 left_watcher、右臂回系统初始位姿，两臂并行启动并等待双回执，清空前置相机视野；本步骤不运行识别或吸取。",
      ],
      [
        "药盒识别 + 确认箱子槽位",
        "已接入 · 并行识别并缓存",
        true,
        "第一次并行识别药盒与纸箱，生成顺时针90°的RGB-D 20槽并持久化；后续循环只更新药盒识别并复用槽位计划。",
      ],
      [
        "药盒吸取",
        "已接入",
        true,
        "使用缓存抓取点完成双吸盘吸取和试抬；本步骤结束后左吸盘保持开启。",
      ],
      [
        "固定轨迹放置药盒",
        "已接入 · 双臂并行到 zhuangxiang 后2×重放",
        true,
        "左右臂先并行到各自 zhuangxiang 保存位姿并校验双回执，再重放 zhuangxiang 奇数帧轨迹；原始第251帧自动关闭吸盘。右夹爪反馈继续记录，但不再因误差中止此专用轨迹。",
      ],
      [
        "识别并吸取药盒顶部",
        "已接入 · 固定吸取位 + 右臂清障",
        true,
        "左臂使用操作员示教的固定接触位吸附；等待1秒后完全张开右夹爪，等待0.5秒，再让右臂沿右基座Y负方向移动150毫米，最后左臂竖直抬升150毫米。",
      ],
      [
        "放入箱子",
        "已接入 · 双臂并行",
        true,
        "使用持久化的当前编号槽位；左法兰绕吸盘中心俯视顺时针旋转90°后分段放入，同时右臂抬升并回系统初始位姿。",
      ],
      [
        "前推、松开吸盘、上抬",
        "待实现",
        false,
        "药盒到槽后前推到位，关闭左吸盘，再沿竖直方向安全上抬。",
      ],
    ],
  },
  task2: {
    number: "02",
    kicker: "TASK2 / 装药板与闭盒",
    title: "装药板与闭盒 · 吸取药盒",
    summary:
      "初始最多4个单层成盒药盒，取走后支持4→3→2→1递减识别；本轮处理2个。",
    readiness: "Task2 单层成盒药盒吸取技能已接入",
    safety: "task2.pick_carton = true · 固定使用已采样单层接触高度",
    detectionLabel: "识别任务二药盒",
    detectionProfile: "task2_single_carton · 初始最多4盒 · 支持递减识别",
    primaryLabel: "Task2 药盒识别与吸取",
    flowTitle: "Task2 固定工作流 · v2.1",
    flowNote: "单盒14步；ACT1/2 后分别切换到 Task2/Task3 数据初始位姿",
    available: true,
    unavailableMessage: "",
    steps: [
      ["双臂并行复位", "接口可用 · 执行后校验双臂反馈", true],
      ["左 left_watcher / 右初始位姿", "接口可用 · 右臂成功后再执行左臂", true],
      ["药盒识别", "接口可用 · 成功后缓存 RGB-D 目标", true],
      ["药盒吸取", "接口可用 · 仅接受第3步有效缓存", true],
      ["左 paper_init / 右 init_pose", "接口可用 · 双臂并行并校验双回执", true],
      [
        "ZR-0（ACT1）放说明书",
        "模型预留 · 暂未接入",
        false,
        "已预留 task2.act1.insert_leaflet 模型槽位和 /api/skills/task2/act1-insert-leaflet 接口位置；当前不会调用模型或移动机械臂。",
      ],
      [
        "左 subtask2_left_init / 右 subtask2_right_init",
        "接口可用 · 双臂并行并校验双回执",
        true,
      ],
      [
        "ZR-0（ACT2）放药板",
        "模型预留 · 暂未接入",
        false,
        "已预留 task2.act2.insert_blister 模型槽位和 /api/skills/task2/act2-insert-blister 接口位置；当前不会调用模型或移动机械臂。",
      ],
      [
        "左 subtask3_left_init / 右 subtask3_right_init",
        "接口可用 · 双臂并行并校验双回执",
        true,
      ],
      [
        "ZR-0（ACT3）关盒",
        "模型预留 · 暂未接入",
        false,
        "已预留 task2.act3.close_carton 模型槽位和 /api/skills/task2/act3-close-carton 接口位置；未来执行时左吸盘保持，当前不会调用模型。",
      ],
      ["双臂回初始位姿", "接口可用 · 执行后校验双臂反馈", true],
      ["识别纸箱位置", "接口可用 · 成功后缓存 RGB-D 开口中心", true],
      [
        "左臂移动到药箱上方，关闭吸盘",
        "接口可用 · 自动预检第12步缓存与运动安全条件",
        true,
        "仅使用第12步30秒内缓存目标；动态预检通过后先升高、再平移、下降到箱沿上方60 mm关闭吸盘，最后垂直抬升。",
      ],
      ["双臂回初始位姿", "接口可用 · 执行后校验双臂反馈", true],
    ],
  },
  task3: {
    number: "03",
    kicker: "TASK3 / 扁盒展开与闭盒",
    title: "扁盒展开与闭盒 · 吸取主面",
    summary:
      "两列扁平药盒，每列叠放2张；每次只识别并吸取两叠当前最上层，取走后重新识别下层。",
    readiness: "Task3 桌面高度自动吸取已接入",
    safety: "task3.pick_flat_carton = true · 桌面Z加吸盘TCP偏移",
    detectionLabel: "识别扁平纸盒主面",
    detectionProfile: "task3_flat_carton · 只识别 130×85 mm 小熊正面 · 取正面中心",
    primaryLabel: "药盒识别 + 药盒吸取",
    flowTitle: "Task3 固定工作流 · v2.0",
    flowNote: "单盒9步；感知与动作分离；不使用 placement.enabled / workflow_locked 占位门禁",
    available: true,
    unavailableMessage: "",
    steps: [
      ["左 left_box_watcher / 右初始位姿", "接口可用 · 右臂成功后再执行左臂", true],
      ["药盒识别", "接口可用 · 仅识别并缓存，不移动机械臂", true],
      ["药盒吸取", "接口可用 · 仅接受第2步有效缓存", true],
      ["展开成盒", "接口可用 · 安全抬升 → left.expand_box → 轨迹重放", true],
      [
        "ZR-0（ACT3）关盒",
        "模型已配置 · 8899执行器当前未启用",
        false,
        "ZR-0 模型地址和哈希已配置，但 ACT rollout/execution 当前为 disabled；这是实际能力未就绪，不使用额外布尔流程锁伪装可用。",
      ],
      ["双臂回初始位姿", "接口可用 · 并行复位并校验回执", true],
      ["识别纸箱位置", "接口可用 · Task3 独立缓存", true],
      ["左臂移动到药箱上方，关闭吸盘", "接口可用 · 使用 Task3 标定与动态安全预检", true],
      ["双臂回初始位姿", "接口可用 · 并行复位并校验回执", true],
    ],
  },
});

const WORKFLOW_STEP_HANDLERS = Object.freeze({
  task1: Object.freeze({
    0: "moveTask1Watcher",
    1: "detectTask1Carton",
    2: "pickCached",
    3: "placeTask1FixedTrajectory",
    4: "pickTask1StagedTop",
    5: "placeTask1InBox",
  }),
  task2: Object.freeze({
    0: "resetBoth",
    1: "moveTask2Watcher",
    2: "detectTask2Carton",
    3: "pickCached",
    4: "moveBothInit",
    6: "moveSubtask2Init",
    8: "moveSubtask3Init",
    10: "resetBoth",
    11: "detectShippingBox",
    12: "placeShippingBox",
    13: "resetBoth",
  }),
  task3: Object.freeze({
    0: "moveTask3Watcher",
    1: "detectTask3Carton",
    2: "pickCached",
    3: "expandCarton",
    5: "resetBoth",
    6: "detectShippingBox",
    7: "placeShippingBox",
    8: "resetBoth",
  }),
});

const WORKFLOW_STEP_DESCRIPTIONS = Object.freeze({
  moveTask1Watcher: "Task1：左臂先以 DEFAULT 到 left_watcher，确认成功后右臂回系统初始位姿；本步骤不运行识别或吸取。",
  detectTask1Carton: "识别药盒；纸箱槽位只在首次运行时生成顺时针90°的20槽计划，后续直接复用持久化坐标和当前装箱编号。",
  detectTask1BoxSlots: "只使用 Center RGB-D：多帧识别纸箱四条内边线，按真实三维底面生成刚性 2×10 槽位，判断下一空槽并缓存右臂放置中心；不会移动机械臂。",
  confirmTask1BoxSlots: "读取第2步已经同步生成的槽位缓存与放置中心；缓存缺失时才自动重新识别，不移动机械臂。",
  pickTask1StagedTop: "左臂使用操作员示教的固定吸取位，接近距离80毫米；吸附后等待1秒，右夹爪完全张开，等待0.5秒并复核稳定反馈，右臂沿右基座Y负方向移动150毫米，随后左臂以DEFAULT速度分两段各75毫米竖直抬升，总计150毫米。",
  placeTask1InBox: "左法兰绕当前吸盘中心俯视顺时针旋转90°，使用持久化的当前编号槽位执行分段放箱并保持吸盘开启；成功后才推进到下一槽位。",
  placeTask1FixedTrajectory: "保持左吸盘吸合，左右臂并行到各自 zhuangxiang 保存位姿并校验双回执；随后以2×速度重放295个奇数帧，原始第251帧自动关闭吸盘。此专用轨迹不使用关节或右夹爪跟随误差中止；右夹爪命令与反馈仍逐帧执行和记录，速度及步长保护保留。",
  resetBoth: "Task2 使用 DEFAULT 速度让左右臂并行回到系统初始位姿；本步骤不会使用 init_pose。",
  moveTask2Watcher: "右臂先回系统初始位姿，再让左臂以 DEFAULT 到 left_watcher；本步骤不会运行识别或吸取。",
  moveTask3Watcher: "右臂先回系统初始位姿，再让左臂以 DEFAULT 到 left_box_watcher；本步骤不会运行识别或吸取。",
  detectTask2Carton: "只使用固定前置 RGB-D 相机识别 Task2 药盒并缓存三维目标；不会移动机械臂或控制吸盘。",
  detectTask3Carton: "只使用固定前置 RGB-D 相机识别 Task3 扁平药盒并缓存三维目标；不会移动机械臂或控制吸盘。",
  detectCache: "固定前置 RGB-D 相机直接识别当前任务区域并缓存目标；本步骤不会移动机械臂。",
  observeWatcher: "移动到任务观察位姿并识别目标。",
  observeBoxWatcher: "Task3：左臂先回到 left_box_watcher；只在固定左侧区域识别 130×85 mm 小熊正面并缓存中心。扁平纸板高度不作硬门禁，展开侧翼、窄侧面和上方纸箱区域不作为目标。",
  pickCached: "Task1 从 left_watcher 直接进入目标上方，法兰保持俯视并按俯视逆时针 90° 标定姿态；Task2 复用 left_pick_ready；Task3 经系统初始位姿进入。随后垂直下降、吸取并试抬。",
  expandCarton: "保持任务三吸盘吸紧：左臂先竖直升到安全高度，再移动到 left.expand_box，最后按原速重放 expand_box 示教轨迹。",
  moveBothInit: "保持吸盘，左臂到 left.paper_init、右臂到 right.init_pose；两臂以 DEFAULT 并行执行，双回执成功后才完成。",
  moveSubtask2Init: "ACT1 后，左臂到 left.subtask2_left_init、右臂到 right.subtask2_right_init；两臂以 DEFAULT 并行执行并校验双回执。",
  moveSubtask3Init: "ACT2 后，左臂到 left.subtask3_left_init、右臂到 right.subtask3_right_init；两臂以 DEFAULT 并行执行并校验双回执。",
  resetLeft: "仅执行左臂复位；右臂不会被下发动作。",
  resetRight: "仅执行右臂复位；左臂不会被下发动作。",
  detectShippingBox: "仅使用固定前置 RGB-D 检测四个内箱沿，按当前任务独立缓存开口中心、箱沿高度、箱底高度和左臂基座坐标；本步骤不会移动机械臂或关闭吸盘。",
  placeShippingBox: "仅使用当前任务上一步的30秒内缓存目标及对应吸盘标定；移动到箱沿上方60 mm后关闭吸盘，再垂直抬升。遥操、ACT、录制或吸盘未吸附时禁止执行。",
});

const POLL_INTERVAL_MS = 5000;
// The video gateway publishes Front at 15 FPS.  Polling every 1.5 seconds made
// the live view appear frozen even though shared-memory capture was healthy.
const FRAME_INTERVAL_MS = 100;
const RECORDING_FRAME_INTERVAL_MS = 200;
const BACKGROUND_FRAME_INTERVAL_MS = 1000;
const REQUEST_TIMEOUT_MS = 4500;
const CARTESIAN_MOTION_REQUEST_TIMEOUT_MS = 30000;
// Keep the annotated recognition frame visible long enough for an operator to
// inspect every candidate. A manual refresh or a new recognition releases it.
const DETECTION_FRAME_HOLD_MS = 300000;
const RECORDING_POLL_INTERVAL_MS = 1000;
const TELEOP_POLL_INTERVAL_MS = 1500;
const CARTESIAN_JOG_POLL_INTERVAL_MS = 600;
const ACT_ROLLOUT_POLL_INTERVAL_MS = 250;
const SUCTION_SYNC_INTERVAL_MS = 250;
const ACT_EXECUTE_STEPS_STORAGE_KEY = "medicine-act-execute-steps";
const WRIST_CAMERA_REFRESH_MS = 250;
const WRIST_CAMERA_RETRY_MS = 1000;
// Requests remain strictly serial; this is only the idle gap after each
// verified step.  A short gap makes held jogging responsive without queuing
// robot commands.
const CARTESIAN_JOG_HOLD_REPEAT_DELAY_MS = 30;

const appState = {
  activeTask: "task1",
  serviceOnline: false,
  safetyContractValid: false,
  actRolloutSafetyContractValid: false,
  cartesianJogSafetyContractValid: false,
  cameraOnline: false,
  profile: null,
  fixedSuctionAxis: null,
  fixedAxisCalibration: null,
  fixedAxisCalibrationBusy: false,
  fixedAxisMarkerPicking: false,
  detection: null,
  frameTimer: null,
  statusTimer: null,
  recordingTimer: null,
  teleopTimer: null,
  frameRequestActive: false,
  frameRequestId: 0,
  frameRequestTimeout: null,
  wristCameraActive: false,
  wristCameraTimers: { left: null, right: null },
  detectionBusy: false,
  recordingBusy: false,
  recording: null,
  recordings: [],
  recordingDeleteId: null,
  recordingDeleteConfirmId: null,
  recordingDeleteConfirmTimer: null,
  replayBusy: false,
  replay: null,
  baseTrajectoryTimer: null,
  baseTrajectoryBusy: false,
  baseTrajectory: null,
  baseTrajectories: [],
  baseTrajectoryReplayBusy: false,
  teleopBusy: false,
  teleop: null,
  teleopModeDirty: false,
  teleopHardRestartArmed: false,
  teleopHardRestartConfirmTimer: null,
  cartesianJogTimer: null,
  cartesianJogBusy: false,
  cartesianJog: null,
  cartesianJogHold: null,
  rightArmHome: null,
  cartesianJogAlertSticky: false,
  suction: null,
  suctionBusy: false,
  suctionTimer: null,
  gripperLock: { enabled: false, arm: "left", busy: false, available: false },
  task1Pick: null,
  task2Pick: null,
  task3Pick: null,
  task2Workflow: null,
  actRolloutTimer: null,
  actRolloutBusy: false,
  actModelSwitchBusy: false,
  actInference: null,
  actRollout: null,
  pollInFlight: {
    status: false,
    recording: false,
    teleop: false,
    baseTrajectory: false,
    cartesianJog: false,
    actRollout: false,
    suction: false,
  },
  task1PickBusy: false,
  workflowBusy: false,
  workflowAutoRunning: false,
  workflowAutoStopRequested: false,
  workflow: {
    task1: { selected: 0, completed: new Set(), message: "" },
    task2: { selected: 0, completed: new Set(), message: "" },
    task3: { selected: 0, completed: new Set(), message: "" },
  },
  lastRenderedRecordingKey: null,
  displayingDetectionFrame: false,
  detectionFramePinnedUntil: 0,
};

const el = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindInteractions();
  selectTask("task1", { resetDetection: false });
  setupOverlayResize();
  refreshApplication();
  startPolling();
});

function cacheElements() {
  [
    "connection-state",
    "connection-label",
    "camera-resolution",
    "camera-fps",
    "camera-live-badge",
    "camera-stage",
    "camera-frame",
    "detection-overlay",
    "fixed-axis-marker-indicator",
    "camera-unavailable",
    "camera-offline-detail",
    "frame-time",
    "camera-name",
    "camera-serial",
    "camera-mode",
    "camera-alignment",
    "refresh-frame",
    "left-wrist-camera",
    "left-wrist-camera-state",
    "right-wrist-camera",
    "right-wrist-camera-state",
    "safety-detail",
    "active-task-number",
    "active-task-kicker",
    "active-task-title",
    "active-task-summary",
    "active-task-readiness",
    "run-detection-label",
    "detection-profile-label",
    "primary-skill-label",
    "active-flow-title",
    "active-flow-note",
    "active-flow-steps",
    "workflow-step-kicker",
    "workflow-step-title",
    "workflow-step-description",
    "workflow-previous",
    "workflow-run-step",
    "workflow-run-detail",
    "workflow-next",
    "workflow-auto-panel",
    "workflow-auto-start",
    "workflow-auto-end",
    "workflow-run-range",
    "workflow-run-range-label",
    "workflow-stop-range",
    "workflow-auto-hint",
    "workflow-step-state",
    "workflow-message",
    "workflow-reset-progress",
    "detection-state",
    "base-coordinate",
    "suction-pixel",
    "detection-score",
    "candidate-count",
    "detection-pixel-label",
    "detection-count-label",
    "detection-angle-label",
    "surface-tilt",
    "detection-message",
    "run-detection",
    "run-task1-pick",
    "task1-pick-detail",
    "fixed-axis-state",
    "fixed-axis-cup-a",
    "fixed-axis-cup-b",
    "fixed-axis-angle",
    "fixed-axis-clearance",
    "fixed-axis-message",
    "fixed-axis-calibration-state",
    "fixed-axis-calibration-detail",
    "fixed-axis-lock-marker",
    "fixed-axis-sample-a",
    "fixed-axis-sample-b",
    "fixed-axis-commit",
    "recording-state",
    "recording-label",
    "recording-purpose",
    "start-recording",
    "stop-recording",
    "recording-duration",
    "recording-frames",
    "recording-left-samples",
    "recording-right-samples",
    "recording-message",
    "recording-path",
    "copy-recording-path",
    "recording-list",
    "base-trajectory-state",
    "base-trajectory-label",
    "base-trajectory-start",
    "base-trajectory-stop",
    "base-trajectory-duration",
    "base-trajectory-points",
    "base-trajectory-start-pose",
    "base-trajectory-end-pose",
    "base-trajectory-message",
    "base-trajectory-path",
    "copy-base-trajectory-path",
    "base-trajectory-list",
    "act-model-select",
    "act-model-state",
    "act-rollout-state",
    "act-rollout-message",
    "act-rollout-inferences",
    "act-rollout-commands",
    "act-rollout-latency",
    "act-execute-steps",
    "act-execute-steps-value",
    "act-execute-steps-total",
    "act-execute-steps-detail",
    "start-act-rollout",
    "stop-act-rollout",
    "teleop-state",
    "teleop-message",
    "teleop-endpoints",
    "teleop-operator-mode",
    "teleop-hold-pose",
    "teleop-hold-pose-label",
    "teleop-mode-detail",
    "start-teleop",
    "stop-teleop",
    "hard-restart-teleop",
    "teleop-mode-chip",
    "jog-state",
    "jog-alert",
    "jog-busy-state",
    "jog-position-x",
    "jog-position-y",
    "jog-position-z",
    "jog-quaternion",
    "jog-status-message",
    "jog-capture-orientation",
    "jog-confirmations",
    "jog-area-clear",
    "jog-estop-ready",
    "jog-suction-released",
    "jog-enable",
    "left-arm-reset-home",
    "right-arm-reset-home",
    "jog-restore-safe",
    "jog-hold-disable",
    "suction-state",
    "suction-detail",
    "suction-on",
    "suction-off",
    "engineering-gripper-lock-chip",
    "engineering-gripper-lock-arm",
    "engineering-gripper-lock-toggle",
    "engineering-gripper-lock-status",
    "gate-summary",
    "gate-list",
    "profile-status",
    "profile-color",
    "profile-depth",
    "profile-aligned",
    "profile-intrinsics",
    "profile-note",
    "service-name",
    "service-version",
    "status-time",
    "toast-region",
  ].forEach((id) => {
    el[id] = document.getElementById(id);
  });
  el.jogMoveButtons = Array.from(
    document.querySelectorAll("[data-jog-axis][data-jog-direction]")
  );
  el.jogStepInputs = Array.from(
    document.querySelectorAll('input[name="jog-step"]')
  );
  el.taskTabs = Array.from(document.querySelectorAll("[data-task-id]"));
}

function bindInteractions() {
  el["act-model-select"].addEventListener("change", switchActModel);
  ["left", "right"].forEach((side) => {
    const frame = el[`${side}-wrist-camera`];
    frame.addEventListener("load", () => {
      frame.classList.add("is-live");
      const state = el[`${side}-wrist-camera-state`];
      state.textContent = "LIVE";
      state.classList.add("is-live");
      state.classList.remove("is-error");
      scheduleWristCameraFrame(side, WRIST_CAMERA_REFRESH_MS);
    });
    frame.addEventListener("error", () => {
      frame.classList.remove("is-live");
      const state = el[`${side}-wrist-camera-state`];
      state.textContent = "未连接";
      state.classList.remove("is-live");
      state.classList.add("is-error");
      scheduleWristCameraFrame(side, WRIST_CAMERA_RETRY_MS);
    });
  });
  el.taskTabs.forEach((button) => {
    button.addEventListener("click", () => {
      selectTask(String(button.dataset.taskId || "task1"));
    });
  });
  el["refresh-frame"].addEventListener("click", () => {
    releaseDetectionFrame();
    refreshFrame(true);
  });
  el["run-detection"].addEventListener("click", runDetection);
  el["run-task1-pick"].addEventListener("click", runActiveTaskPick);
  el["workflow-previous"].addEventListener("click", () => selectWorkflowStep(-1, true));
  el["workflow-next"].addEventListener("click", () => selectWorkflowStep(1, true));
  el["workflow-run-step"].addEventListener("click", executeWorkflowStep);
  el["workflow-auto-start"].addEventListener("change", () => {
    syncWorkflowAutoRange({ resetEnd: true });
    renderWorkflowPanel();
  });
  el["workflow-auto-end"].addEventListener("change", renderWorkflowPanel);
  el["workflow-run-range"].addEventListener("click", executeWorkflowRange);
  el["workflow-stop-range"].addEventListener("click", requestWorkflowRangeStop);
  el["workflow-reset-progress"].addEventListener("click", resetWorkflowProgress);
  el["active-flow-steps"].addEventListener("click", (event) => {
    const step = event.target.closest("[data-workflow-step-index]");
    if (!step) return;
    selectWorkflowStep(Number(step.dataset.workflowStepIndex));
  });
  el["active-flow-steps"].addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const step = event.target.closest("[data-workflow-step-index]");
    if (!step) return;
    event.preventDefault();
    selectWorkflowStep(Number(step.dataset.workflowStepIndex));
  });
  el["start-recording"].addEventListener("click", startRecording);
  el["stop-recording"].addEventListener("click", stopRecording);
  el["base-trajectory-start"].addEventListener(
    "click",
    startBaseTrajectoryRecording
  );
  el["base-trajectory-stop"].addEventListener(
    "click",
    stopBaseTrajectoryRecording
  );
  el["start-act-rollout"].addEventListener("click", startActRollout);
  el["stop-act-rollout"].addEventListener("click", stopActRollout);
  try {
    const storedExecuteSteps = Number(
      window.localStorage.getItem(ACT_EXECUTE_STEPS_STORAGE_KEY)
    );
    if (Number.isInteger(storedExecuteSteps) && storedExecuteSteps > 0) {
      el["act-execute-steps"].value = String(storedExecuteSteps);
      el["act-execute-steps"].dataset.userSet = "true";
    }
  } catch (_error) {
    // A private browser context may disable localStorage; the control still works.
  }
  el["act-execute-steps"].addEventListener("input", () => {
    el["act-execute-steps"].dataset.userSet = "true";
    syncActExecuteStepsControl();
    try {
      window.localStorage.setItem(
        ACT_EXECUTE_STEPS_STORAGE_KEY,
        el["act-execute-steps"].value
      );
    } catch (_error) {
      // Keep the current in-page selection when persistence is unavailable.
    }
  });
  syncActExecuteStepsControl();
  window.addEventListener("keydown", handleGlobalShortcut);
  el["copy-recording-path"].addEventListener("click", () => {
    copyPath(el["copy-recording-path"].dataset.copyPath || "");
  });
  el["copy-base-trajectory-path"].addEventListener("click", () => {
    copyPath(el["copy-base-trajectory-path"].dataset.copyPath || "");
  });
  el["recording-list"].addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy-recording-index]");
    if (copyButton && el["recording-list"].contains(copyButton)) {
      const index = Number(copyButton.dataset.copyRecordingIndex);
      copyPath(appState.recordings[index]?.path || "");
      return;
    }
    const deleteButton = event.target.closest("[data-delete-recording-index]");
    if (deleteButton && el["recording-list"].contains(deleteButton)) {
      const index = Number(deleteButton.dataset.deleteRecordingIndex);
      const item = appState.recordings[index];
      if (item) deleteRecording(item);
      return;
    }
    const replayButton = event.target.closest("[data-replay-recording-index]");
    if (!replayButton || !el["recording-list"].contains(replayButton)) return;
    const index = Number(replayButton.dataset.replayRecordingIndex);
    const item = appState.recordings[index];
    if (!item) return;
    if (
      appState.replay?.active === true &&
      appState.replay?.recording_id === item.id
    ) {
      stopTrajectoryReplay();
    } else {
      startTrajectoryReplay(item);
    }
  });
  el["base-trajectory-list"].addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy-base-trajectory-index]");
    if (copyButton && el["base-trajectory-list"].contains(copyButton)) {
      const index = Number(copyButton.dataset.copyBaseTrajectoryIndex);
      copyPath(appState.baseTrajectories[index]?.path || "");
      return;
    }
    const replayButton = event.target.closest("[data-replay-base-trajectory-index]");
    if (!replayButton || !el["base-trajectory-list"].contains(replayButton)) return;
    const index = Number(replayButton.dataset.replayBaseTrajectoryIndex);
    const item = appState.baseTrajectories[index];
    if (!item) return;
    if (
      appState.baseTrajectory?.replay?.recording_id === item.id &&
      String(appState.baseTrajectory?.replay?.state || "") === "replaying"
    ) {
      stopBaseTrajectoryReplay();
    } else {
      startBaseTrajectoryReplay(item);
    }
  });
  el["start-teleop"].addEventListener("click", startTeleop);
  el["stop-teleop"].addEventListener("click", stopTeleop);
  el["hard-restart-teleop"].addEventListener("click", hardRestartTeleop);
  el["teleop-operator-mode"].addEventListener("change", () => {
    appState.teleopModeDirty = true;
    updateTeleopButton();
  });
  el["teleop-hold-pose"].addEventListener("change", () => {
    appState.teleopModeDirty = true;
    updateTeleopButton();
  });
  el["jog-capture-orientation"].addEventListener(
    "click",
    captureCartesianJogOrientation
  );
  el["jog-enable"].addEventListener("click", enableCartesianJog);
  el["jog-restore-safe"]?.addEventListener(
    "click",
    restoreCartesianJogSafePose
  );
  el["left-arm-reset-home"].addEventListener("click", resetLeftArmHome);
  el["right-arm-reset-home"].addEventListener("click", resetRightArmHome);
  el["jog-hold-disable"].addEventListener("click", disableCartesianJog);
  el["suction-on"].addEventListener("click", () => setSuction(true));
  el["suction-off"].addEventListener("click", () => setSuction(false));
  el["engineering-gripper-lock-toggle"]?.addEventListener("click", toggleEngineeringGripperLock);
  document.querySelectorAll("[data-gripper-lock-arm]").forEach((button) => {
    button.addEventListener("click", () => {
      if (appState.gripperLock.busy) return;
      const arm = button.dataset.gripperLockArm === "right" ? "right" : "left";
      if (arm === appState.gripperLock.arm) return;
      el["engineering-gripper-lock-arm"].value = arm;
      setEngineeringGripperLock(appState.gripperLock.enabled);
    });
  });
  el["jog-area-clear"].addEventListener("change", updateCartesianJogControls);
  el["jog-estop-ready"].addEventListener("change", updateCartesianJogControls);
  el["jog-suction-released"].addEventListener(
    "change",
    updateCartesianJogControls
  );
  el.jogMoveButtons.forEach((button) => {
    bindCartesianJogHoldButton(button);
  });
  el.jogStepInputs.forEach((input) => {
    input.addEventListener("change", updateCartesianJogControls);
  });

  el["camera-frame"].addEventListener("load", () => {
    if (!loadedFrameRequestIsCurrent()) return;
    clearFrameRequestTimeout();
    appState.frameRequestActive = false;
    setCameraAvailability(true);
    el["frame-time"].textContent =
      appState.displayingDetectionFrame && appState.detection?.captured_at
        ? `检测帧 ${formatTime(appState.detection.captured_at)}`
        : recordingIsActive()
          ? `录制预览 ${formatTime(new Date())}`
          : formatTime(new Date());
    drawDetectionOverlay();
  });

  el["camera-frame"].addEventListener("error", () => {
    if (!loadedFrameRequestIsCurrent()) return;
    clearFrameRequestTimeout();
    appState.frameRequestActive = false;
    releaseDetectionFrame();
    setCameraAvailability(false, "无法读取 /api/camera/frame.jpg");
  });

  window.addEventListener("online", refreshApplication);
  window.addEventListener("offline", () => {
    stopCartesianJogHold("浏览器网络已离线");
    setServiceAvailability(false, "浏览器网络已离线");
    setCameraAvailability(false, "浏览器网络已离线");
  });
  window.addEventListener("blur", () => {
    stopCartesianJogHold("窗口失去焦点");
  });
  window.addEventListener("pointerup", (event) => {
    stopCartesianJogHoldForPointer(event, "方向按钮已松开");
  });
  window.addEventListener("pointercancel", (event) => {
    stopCartesianJogHoldForPointer(event, "指针操作已取消");
  });
  window.addEventListener("pagehide", () => {
    stopCartesianJogHold("页面已离开");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopCartesianJogHold("页面进入后台");
    } else if (appState.wristCameraActive) {
      scheduleWristCameraFrame("left", 0);
      scheduleWristCameraFrame("right", 0);
    }
    if (!document.hidden && appState.serviceOnline && !appState.suctionBusy) {
      refreshSuctionStatus().catch(() => {});
    }
    if (appState.frameTimer !== null) scheduleFramePoll(0);
  });
}

function scheduleWristCameraFrame(side, delayMs) {
  const existing = appState.wristCameraTimers[side];
  if (existing !== null) window.clearTimeout(existing);
  appState.wristCameraTimers[side] = null;
  if (!appState.wristCameraActive || document.hidden) return;
  appState.wristCameraTimers[side] = window.setTimeout(() => {
    appState.wristCameraTimers[side] = null;
    const frame = el[`${side}-wrist-camera`];
    const url = side === "left" ? API.leftWristCamera : API.rightWristCamera;
    if (frame && appState.wristCameraActive) {
      frame.src = `${url}?v=${Date.now()}`;
    }
  }, delayMs);
}

function setWristCameraStreaming(active) {
  appState.wristCameraActive = active;
  const streams = {
    left: API.leftWristCamera,
    right: API.rightWristCamera,
  };
  Object.keys(streams).forEach((side) => {
    const frame = el[`${side}-wrist-camera`];
    const state = el[`${side}-wrist-camera-state`];
    if (!frame || !state) return;
    const existing = appState.wristCameraTimers[side];
    if (existing !== null) window.clearTimeout(existing);
    appState.wristCameraTimers[side] = null;
    if (!active) {
      frame.removeAttribute("src");
      frame.classList.remove("is-live");
      state.textContent = "已暂停";
      state.classList.remove("is-live", "is-error");
      return;
    }
    state.textContent = "连接中";
    state.classList.remove("is-live", "is-error");
    scheduleWristCameraFrame(side, 0);
  });
}

// Keep the operator surface focused: task execution and engineering tools live
// on separate pages, while sharing the same DOM and backend state.
document.addEventListener("DOMContentLoaded", () => {
  const modeButtons = Array.from(
    document.querySelectorAll("[data-console-mode]")
  );
  const surfaces = Array.from(
    document.querySelectorAll("[data-console-surface]")
  );
  const engineeringButtons = Array.from(
    document.querySelectorAll("[data-engineering-tool-button]")
  );
  const engineeringPanels = Array.from(
    document.querySelectorAll("[data-engineering-tool]")
  );
  if (!modeButtons.length || !surfaces.length) return;

  let activeEngineeringTool = "parameters";
  try {
    activeEngineeringTool =
      window.localStorage.getItem("medicine-pack-engineering-tool") ||
      "parameters";
  } catch (_error) {
    activeEngineeringTool = "parameters";
  }
  if (!engineeringButtons.some(
    (button) => button.dataset.engineeringToolButton === activeEngineeringTool
  )) activeEngineeringTool = "parameters";

  const selectEngineeringTool = (requestedTool) => {
    const tool = engineeringButtons.some(
      (button) => button.dataset.engineeringToolButton === requestedTool
    ) ? requestedTool : "parameters";
    activeEngineeringTool = tool;
    document.body.dataset.engineeringTool = tool;
    engineeringButtons.forEach((button) => {
      const active = button.dataset.engineeringToolButton === tool;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    engineeringPanels.forEach((panel) => {
      panel.hidden =
        document.body.dataset.consoleMode !== "engineering" ||
        panel.dataset.engineeringTool !== tool;
    });
    try {
      window.localStorage.setItem("medicine-pack-engineering-tool", tool);
    } catch (_error) {
      // Storage is optional.
    }
  };

  const selectConsoleMode = (requestedMode) => {
    const mode = requestedMode === "engineering" ? "engineering" : "run";
    document.body.dataset.consoleMode = mode;
    modeButtons.forEach((button) => {
      const active = button.dataset.consoleMode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    surfaces.forEach((surface) => {
      surface.hidden = surface.dataset.consoleSurface !== mode;
    });
    setWristCameraStreaming(mode === "run");
    if (mode === "engineering") {
      selectEngineeringTool(activeEngineeringTool);
    } else {
      engineeringPanels.forEach((panel) => { panel.hidden = true; });
      stopCartesianJogHold("已离开手动控制页");
    }
    try {
      window.localStorage.setItem("medicine-pack-console-mode", mode);
    } catch (_error) {
      // Storage is optional; the page remains fully usable without it.
    }
    window.requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  };

  modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      selectConsoleMode(button.dataset.consoleMode);
    });
  });
  engineeringButtons.forEach((button) => {
    button.addEventListener("click", () => {
      stopCartesianJogHold("切换工程工具");
      selectEngineeringTool(button.dataset.engineeringToolButton);
      window.requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    });
  });

  let initialMode = "run";
  try {
    initialMode = window.localStorage.getItem("medicine-pack-console-mode") || "run";
  } catch (_error) {
    initialMode = "run";
  }
  selectConsoleMode(initialMode);
});

function selectTask(taskId, { resetDetection = true } = {}) {
  const normalizedTaskId = Object.hasOwn(TASK_PROFILES, taskId)
    ? taskId
    : "task1";
  if (appState.workflowBusy && normalizedTaskId !== appState.activeTask) {
    showToast("工作流执行期间不能切换任务。", "error");
    return;
  }
  const profile = TASK_PROFILES[normalizedTaskId];
  appState.activeTask = normalizedTaskId;

  el.taskTabs.forEach((button) => {
    const active = button.dataset.taskId === normalizedTaskId;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    button.disabled = appState.workflowBusy;
  });

  el["active-task-number"].textContent = profile.number;
  el["active-task-kicker"].textContent = profile.kicker;
  el["active-task-title"].textContent = profile.title;
  el["active-task-summary"].textContent = profile.summary;
  el["active-task-readiness"].textContent = profile.readiness;
  el["safety-detail"].textContent = profile.safety;
  el["run-detection-label"].textContent = profile.detectionLabel;
  el["detection-profile-label"].textContent = profile.detectionProfile;
  el["primary-skill-label"].textContent = profile.primaryLabel;
  el["active-flow-title"].textContent = profile.flowTitle;
  el["active-flow-note"].textContent = profile.flowNote;
  el["active-flow-steps"].innerHTML = profile.steps
    .map(
      ([label, status, available], index) => `
        <li
          class="flow-step ${available ? "flow-step--available" : "flow-step--locked"}"
          data-workflow-step-index="${index}"
          role="button"
          tabindex="0"
          aria-label="第 ${index + 1} 步：${escapeHTML(String(label))}"
        >
          <span>${index + 1}</span>
          <div>
            <strong>${escapeHTML(String(label))}</strong>
            <small>${escapeHTML(String(status))}</small>
          </div>
        </li>
      `
    )
    .join("");
  populateWorkflowAutoRange();
  renderWorkflowPanel();

  if (resetDetection) {
    appState.detection = null;
    releaseDetectionFrame();
    resetDetectionMetrics();
    el["candidate-count"].textContent = "—";
    setDetectionState("未运行", "idle");
    el["detection-message"].textContent = profile.available
      ? "点击识别或运行药盒识别步骤后，才会获取并更新坐标。"
      : profile.unavailableMessage;
    renderFixedAxis(null);
    if (appState.serviceOnline) refreshFrame(false);
  } else {
    renderFixedAxis(appState.detection);
  }

  updateDetectionButton();
  updateTask1PickControls();
}

function activeWorkflowState() {
  return appState.workflow[appState.activeTask];
}

function workflowHandler(taskId, index) {
  return WORKFLOW_STEP_HANDLERS[taskId]?.[index] || "";
}

function lastContiguousWorkflowStep(taskId, startIndex) {
  const stepCount = TASK_PROFILES[taskId].steps.length;
  let endIndex = startIndex;
  while (endIndex + 1 < stepCount && workflowHandler(taskId, endIndex + 1)) {
    endIndex += 1;
  }
  return endIndex;
}

function populateWorkflowAutoRange() {
  const taskId = appState.activeTask;
  const profile = TASK_PROFILES[taskId];
  el["workflow-auto-panel"].hidden = false;
  const options = profile.steps
    .map(
      ([label], index) =>
        `<option value="${index}">${index + 1} · ${escapeHTML(String(label))}</option>`
    )
    .join("");
  el["workflow-auto-start"].innerHTML = options;
  el["workflow-auto-end"].innerHTML = options;
  syncWorkflowAutoRange({ resetStart: true, resetEnd: true });
}

function syncWorkflowAutoRange({ resetStart = false, resetEnd = false } = {}) {
  const taskId = appState.activeTask;
  const state = activeWorkflowState();
  if (resetStart) el["workflow-auto-start"].value = String(state.selected);
  const startIndex = Number(el["workflow-auto-start"].value);
  if (resetEnd || Number(el["workflow-auto-end"].value) < startIndex) {
    el["workflow-auto-end"].value = String(
      lastContiguousWorkflowStep(taskId, startIndex)
    );
  }
}

function selectedWorkflowRange() {
  const taskId = appState.activeTask;
  const startIndex = Number(el["workflow-auto-start"].value);
  const endIndex = Number(el["workflow-auto-end"].value);
  if (
    !Object.hasOwn(TASK_PROFILES, taskId) ||
    !Number.isInteger(startIndex) ||
    !Number.isInteger(endIndex)
  ) {
    return { valid: false, reason: "当前任务的自动执行区间无效。" };
  }
  if (startIndex > endIndex) {
    return { valid: false, reason: "起始步骤 A 不能晚于结束步骤 B。" };
  }
  const unavailable = [];
  for (let index = startIndex; index <= endIndex; index += 1) {
    if (!workflowHandler(taskId, index)) unavailable.push(index + 1);
  }
  if (unavailable.length) {
    return {
      valid: false,
      reason: `区间包含尚未接入的第 ${unavailable.join("、")} 步，不能自动执行。`,
    };
  }
  return { valid: true, taskId, startIndex, endIndex, reason: "" };
}

function renderWorkflowAutoControls() {
  el["workflow-auto-panel"].hidden = false;
  const selection = selectedWorkflowRange();
  const startLabel = Number(el["workflow-auto-start"].value) + 1;
  const endLabel = Number(el["workflow-auto-end"].value) + 1;
  el["workflow-run-range-label"].textContent =
    `自动执行 ${startLabel} → ${endLabel}`;
  el["workflow-auto-start"].disabled = appState.workflowBusy;
  el["workflow-auto-end"].disabled = appState.workflowBusy;
  el["workflow-run-range"].disabled = appState.workflowBusy || !selection.valid;
  el["workflow-stop-range"].hidden = !appState.workflowAutoRunning;
  el["workflow-stop-range"].disabled = appState.workflowAutoStopRequested;
  el["workflow-stop-range"].textContent = appState.workflowAutoStopRequested
    ? "将在本步完成后停止"
    : "完成本步后停止";
  el["workflow-auto-hint"].textContent = appState.workflowAutoRunning
    ? appState.workflowAutoStopRequested
      ? "停止请求已记录；当前步骤完成后不会继续。"
      : `正在严格串行执行第 ${startLabel} 至 ${endLabel} 步。`
    : selection.valid
      ? `将依次执行第 ${startLabel} 至 ${endLabel} 步；任一步失败都会立即停止。`
      : selection.reason;
  el.taskTabs.forEach((button) => {
    button.disabled = appState.workflowBusy;
  });
}

function selectWorkflowStep(indexOrDelta, relative = false) {
  if (appState.workflowBusy) return;
  const state = activeWorkflowState();
  const stepCount = TASK_PROFILES[appState.activeTask].steps.length;
  const requested = relative ? state.selected + indexOrDelta : indexOrDelta;
  state.selected = Math.max(0, Math.min(stepCount - 1, requested));
  state.message = "";
  syncWorkflowAutoRange({ resetStart: true, resetEnd: true });
  renderWorkflowPanel();
}

function resetWorkflowProgress() {
  if (appState.workflowBusy) return;
  const state = activeWorkflowState();
  state.selected = 0;
  state.completed.clear();
  state.message = "调试进度已回到第 1 步；没有向机械臂下发动作。";
  syncWorkflowAutoRange({ resetStart: true, resetEnd: true });
  renderWorkflowPanel();
}

function renderWorkflowPanel(status = "idle") {
  const taskId = appState.activeTask;
  const profile = TASK_PROFILES[taskId];
  const state = activeWorkflowState();
  const index = Math.max(0, Math.min(profile.steps.length - 1, state.selected));
  state.selected = index;
  const [label, declaredStatus, _available, configuredDescription] =
    profile.steps[index];
  const handler = WORKFLOW_STEP_HANDLERS[taskId]?.[index] || "";
  const implemented = Boolean(handler);
  const productionDisabled = declaredStatus.includes("待现场启用");

  el["workflow-step-kicker"].textContent =
    `${profile.number} · STEP ${index + 1} / ${profile.steps.length}`;
  el["workflow-step-title"].textContent = label;
  el["workflow-step-description"].textContent = implemented
    ? WORKFLOW_STEP_DESCRIPTIONS[handler]
    : configuredDescription ||
      `${declaredStatus}。当前步骤尚未接入实机接口，可先选择其他已接入步骤调试。`;
  el["workflow-previous"].disabled = appState.workflowBusy || index === 0;
  el["workflow-next"].disabled =
    appState.workflowBusy || index === profile.steps.length - 1;
  el["workflow-run-step"].disabled = appState.workflowBusy || !implemented;
  el["workflow-reset-progress"].disabled = appState.workflowBusy;
  el["workflow-run-detail"].textContent = implemented
    ? "只执行这一步，完成后自动停住"
    : productionDisabled
      ? "生产运动未启用"
      : "该步骤尚未接入";

  const stateLabel = appState.workflowBusy
    ? "执行中"
    : state.completed.has(index)
      ? "已完成"
      : implemented
        ? "待执行"
        : productionDisabled
          ? "待启用"
          : "未接入";
  const pill = el["workflow-step-state"];
  pill.textContent = stateLabel;
  pill.classList.remove(
    "result-pill--idle",
    "result-pill--success",
    "result-pill--error",
    "result-pill--busy"
  );
  pill.classList.add(
    appState.workflowBusy
      ? "result-pill--busy"
      : status === "error"
        ? "result-pill--error"
        : state.completed.has(index)
          ? "result-pill--success"
          : "result-pill--idle"
  );
  el["workflow-message"].textContent = state.message || (
    implemented
      ? "选择步骤后点击“执行当前步骤”。"
      : productionDisabled
        ? `第${index + 1}步代码与只读预检已就绪；需现场授权后才能启用真实运动。`
        : "该步骤只保留在流程中，等待姿态采样、ACT 训练或接口接入。"
  );

  Array.from(el["active-flow-steps"].children).forEach((step, stepIndex) => {
    step.classList.toggle("is-current", stepIndex === index);
    step.classList.toggle("is-complete", state.completed.has(stepIndex));
    step.setAttribute("aria-current", stepIndex === index ? "step" : "false");
  });
  renderWorkflowAutoControls();
}

function renderTask2WorkflowStatus(workflow) {
  appState.task2Workflow = workflow && typeof workflow === "object"
    ? workflow
    : null;
  if (appState.activeTask !== "task2" || !appState.task2Workflow) return;
  const stepIndex = Number(appState.task2Workflow.step_index);
  if (!Number.isInteger(stepIndex)) return;
  const state = activeWorkflowState();
  if (state.selected !== stepIndex) return;
  const stageLabels = {
    right_home: "右臂正在回系统初始位姿",
    left_watcher: "右臂已校验；左臂正在前往 left_watcher",
    parallel_ready_poses: "双臂并行运动中：左 paper_init / 右 init_pose",
    parallel_subtask2_init_poses:
      "双臂并行运动中：左 subtask2_left_init / 右 subtask2_right_init",
    parallel_subtask3_init_poses:
      "双臂并行运动中：左 subtask3_left_init / 右 subtask3_right_init",
    subtask2_init_verified: "Task2 数据初始位姿已收到左右臂执行回执",
    subtask3_init_verified: "Task3 数据初始位姿已收到左右臂执行回执",
    feedback_verified: "右臂初始位姿与左臂 left_watcher 均已收到执行回执",
    failed: "本步骤未完成",
  };
  const label = stageLabels[appState.task2Workflow.stage] ||
    appState.task2Workflow.message || "正在同步 Task2 执行状态";
  state.message = appState.task2Workflow.error
    ? `${label}：${appState.task2Workflow.error}`
    : label;
  renderWorkflowPanel(
    appState.task2Workflow.state === "failed" ? "error" :
      appState.task2Workflow.state === "running" ? "busy" : "idle"
  );
}

function workflowPost(url, timeoutMs = 90000, body = {}) {
  return requestJSON(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    timeoutMs
  );
}

function task2PlacementBlockerLabel(blocker) {
  return ({
    run_task2_step12_before_step13: "请先执行第12步识别纸箱",
    run_task3_step7_before_step8: "请先执行第7步识别纸箱",
    task2_shipping_box_target_expired: "第12步纸箱目标已超过30秒，请重新识别",
    task2_shipping_box_target_invalid: "第12步纸箱目标无效，请重新识别",
    task3_shipping_box_target_expired: "第7步纸箱目标已超过30秒，请重新识别",
    task3_shipping_box_target_invalid: "第7步纸箱目标无效，请重新识别",
    task2_shipping_box_opening_margin_invalid: "纸箱开口尺寸或安全余量不足",
    task2_shipping_box_flange_target_outside_workspace: "放置目标超出左臂安全工作空间",
    task3_shipping_box_opening_margin_invalid: "纸箱开口尺寸或安全余量不足",
    task3_shipping_box_flange_target_outside_workspace: "放置目标超出左臂安全工作空间",
    task2_suction_unavailable: "吸盘不可用",
    task2_suction_must_be_engaged: "左吸盘未保持吸附",
    task3_suction_unavailable: "吸盘不可用",
    task3_suction_must_be_engaged: "左吸盘未保持吸附",
    stop_trajectory_recording: "请先停止轨迹录制",
    stop_trajectory_replay: "请先停止轨迹回放",
    stop_act_rollout: "请先停止ACT",
    stop_teleoperation: "请先停止遥操",
    left_arm_busy: "左臂当前忙碌",
  })[blocker] || blocker;
}

function applyWorkflowPayload(payload) {
  if (payload?.detection) {
    appState.detection = payload.detection;
    renderDetection(payload.detection);
    showDetectionFrame(payload.detection);
  }
  if (payload?.box_slots_detection) {
    showDetectionFrame(payload.box_slots_detection);
  }
  if (payload?.cartesian_jog) renderCartesianJog(payload.cartesian_jog);
  if (payload?.right_arm_home) appState.rightArmHome = payload.right_arm_home;
  if (payload?.suction) renderSuction(payload.suction);
  if (payload?.task1_pick) renderTask1Pick(payload.task1_pick);
  if (payload?.task2_pick) renderTask2Pick(payload.task2_pick);
  if (payload?.task3_pick) renderTask3Pick(payload.task3_pick);
}

function assertWorkflowStepSucceeded(payload, taskId, index) {
  if (!payload || payload.ok !== true) {
    throw new Error(`第 ${index + 1} 步未返回成功结果`);
  }
  if (payload.skill && payload.skill.status !== "succeeded") {
    throw new Error(
      `第 ${index + 1} 步后端状态不是 succeeded：${payload.skill.status || "missing"}`
    );
  }
  const motions = payload.result?.motions;
  if (motions && typeof motions === "object") {
    for (const [arm, result] of Object.entries(motions)) {
      if (!result || result.executed !== true) {
        throw new Error(`第 ${index + 1} 步 ${arm} 臂没有已执行回执`);
      }
    }
  }
  if (taskId === "task1" && index === 0) {
    if (payload.skill?.id !== "task1.move_watcher_pose") {
      throw new Error("Task1 第1步返回了错误的技能回执");
    }
    if (
      payload.result?.execution !== "parallel" ||
      payload.result?.motions?.left?.executed !== true ||
      payload.result?.motions?.right?.executed !== true
    ) {
      throw new Error("Task1 第1步必须并行完成左臂 watcher 和右臂复位并确认双回执");
    }
  }
  if (taskId === "task1" && index === 4) {
    if (payload.skill?.id !== "task1.pick_staged_carton_top") {
      throw new Error("Task1 第5步返回了错误的技能回执");
    }
    if (
      payload.result?.flange_center_alignment !==
        "operator_taught_fixed_contact_pose" ||
      payload.result?.motions?.right_clearance?.executed !== true ||
      payload.result?.motions?.test_lift?.executed !== true
    ) {
      throw new Error("Task1 第5步必须完成固定吸取、右臂清障和左臂抬升");
    }
  }
  if (taskId === "task1" && index === 5) {
    if (payload.skill?.id !== "task1.place_in_box") {
      throw new Error("Task1 第6步返回了错误的技能回执");
    }
    if (
      payload.result?.execution !==
        "parallel_left_place_right_retreat_and_system_home" ||
      payload.result?.motions?.left?.executed !== true ||
      payload.result?.motions?.right?.executed !== true ||
      payload.result?.suction_remains_engaged !== true
    ) {
      throw new Error("Task1 第6步必须并行完成左臂放箱和右臂抬升返回");
    }
  }
  if (taskId === "task1" && index === 3) {
    if (payload.skill?.id !== "task1.place_carton_fixed_trajectory") {
      throw new Error("Task1 第4步返回了错误的技能回执");
    }
    if (
      payload.result?.execution !== "parallel_poses_then_replay" ||
      payload.result?.motions?.left?.executed !== true ||
      payload.result?.motions?.right?.executed !== true ||
      payload.result?.replay_completed?.state !== "completed" ||
      payload.result?.replay_completed?.suction_release_state !== "released"
    ) {
      throw new Error("Task1 第4步必须完成双臂 zhuangxiang 位姿、轨迹重放和吸盘释放");
    }
  }
  if (taskId === "task2" && index === 1) {
    if (payload.skill?.id !== "task2.move_watcher_pose") {
      throw new Error("Task2 第2步返回了错误的技能回执");
    }
    if (
      payload.result?.motions?.right?.executed !== true ||
      payload.result?.motions?.left?.executed !== true
    ) {
      throw new Error("Task2 第2步必须同时确认右臂复位和左臂 left_watcher 到位");
    }
  }
  return payload;
}

async function performWorkflowStep(taskId, index) {
  const handler = workflowHandler(taskId, index);
  if (!handler) {
    throw new Error(`第 ${index + 1} 步尚未接入，不能执行`);
  }
  // A detection overlay is a historical frame.  Release it before every
  // workflow operation so arm motion is always shown on the live feed.
  releaseDetectionFrame();
  refreshFrame(true);
  let payload = null;
  if (handler === "resetBoth") {
    if (["task2", "task3"].includes(taskId)) {
      payload = await workflowPost(API.task2ResetBothHome, 120000);
    } else {
      const left = await workflowPost(API.leftArmResetHomeSkill);
      applyWorkflowPayload(left);
      const right = await workflowPost(API.rightArmResetHomeSkill);
      applyWorkflowPayload(right);
      payload = right;
    }
  } else if (handler === "moveBothInit") {
    if (taskId === "task2") {
      payload = await workflowPost(API.task2MoveReadyPoses, 120000);
    } else {
      const right = await workflowPost(
        API.runtimePoseMove,
        90000,
        { arm: "right", name: "init_pose" }
      );
      applyWorkflowPayload(right);
      const left = await workflowPost(
        API.runtimePoseMove,
        90000,
        { arm: "left", name: "init_pose" }
      );
      applyWorkflowPayload(left);
      payload = left;
    }
  } else if (handler === "moveSubtask2Init") {
    payload = await workflowPost(API.task2MoveSubtask2InitPoses, 120000);
  } else if (handler === "moveSubtask3Init") {
    payload = await workflowPost(API.task2MoveSubtask3InitPoses, 120000);
  } else if (handler === "moveTask1Watcher") {
    payload = await workflowPost(API.task1MoveWatcherPose, 120000);
  } else if (handler === "moveTask2Watcher") {
    payload = await workflowPost(API.task2MoveWatcherPose, 120000);
  } else if (handler === "moveTask3Watcher") {
    payload = await workflowPost(API.task3MoveWatcherPose, 120000);
  } else if (handler === "detectTask1Carton") {
    payload = await workflowPost(API.task1DetectCarton, 90000);
  } else if (handler === "detectTask1BoxSlots") {
    payload = await workflowPost(API.task1DetectBoxSlots, 90000);
  } else if (handler === "confirmTask1BoxSlots") {
    payload = await workflowPost(API.task1ConfirmBoxSlots, 90000);
  } else if (handler === "pickTask1StagedTop") {
    payload = await workflowPost(API.task1PickStagedTop, 120000);
  } else if (handler === "placeTask1InBox") {
    payload = await workflowPost(API.task1PlaceInBox, 120000);
  } else if (handler === "placeTask1FixedTrajectory") {
    payload = await workflowPost(API.task1PlaceFixedTrajectory, 120000);
  } else if (handler === "detectTask2Carton") {
    payload = await workflowPost(API.task2DetectCarton, 30000);
  } else if (handler === "detectTask3Carton") {
    payload = await workflowPost(API.task3DetectCarton, 30000);
  } else if (handler === "resetLeft") {
    payload = await workflowPost(API.leftArmResetHomeSkill);
  } else if (handler === "resetRight") {
    payload = await workflowPost(API.rightArmResetHomeSkill);
  } else if (["detectCache", "observeWatcher", "observeBoxWatcher"].includes(handler)) {
    payload = await workflowPost({
      task1: API.task1ObserveCarton,
      task2: API.task2ObserveCarton,
      task3: API.task3ObserveCarton,
    }[taskId]);
  } else if (handler === "pickCached") {
    payload = await workflowPost({
      task1: API.task1PickCachedCarton,
      task2: API.task2PickCachedCarton,
      task3: API.task3PickCachedCarton,
    }[taskId]);
  } else if (handler === "expandCarton") {
    payload = await workflowPost(API.task3ExpandCarton, 120000);
  } else if (handler === "detectShippingBox") {
    payload = await workflowPost(
      taskId === "task3" ? API.task3DetectShippingBox : API.task2DetectShippingBox,
      30000
    );
  } else if (handler === "placeShippingBox") {
    const preflightPayload = await requestJSON(
      taskId === "task3"
        ? API.task3PlaceShippingBoxPreflight
        : API.task2PlaceShippingBoxPreflight
    );
    const preflight = preflightPayload?.preflight;
    if (preflight?.ready !== true) {
      const blockers = Array.isArray(preflight?.blockers)
        ? preflight.blockers.map(task2PlacementBlockerLabel)
        : ["无法读取放置预检结果"];
      throw new Error(`第${index + 1}步预检未通过：${blockers.join("；")}`);
    }
    payload = await workflowPost(
      taskId === "task3" ? API.task3PlaceShippingBox : API.task2PlaceShippingBox,
      120000,
      { confirmation: `PLACE_${taskId.toUpperCase()}_IN_SHIPPING_BOX` }
    );
  }
  assertWorkflowStepSucceeded(payload, taskId, index);
  applyWorkflowPayload(payload);
  const keepsDetectionOverlay =
    handler.startsWith("detect") ||
    ["observeWatcher", "observeBoxWatcher", "confirmTask1BoxSlots"].includes(handler);
  if (!keepsDetectionOverlay) {
    releaseDetectionFrame();
    refreshFrame(true);
  }
  return payload;
}

async function executeWorkflowStep() {
  if (appState.workflowBusy) return;
  const taskId = appState.activeTask;
  const profile = TASK_PROFILES[taskId];
  const state = activeWorkflowState();
  const index = state.selected;
  if (!workflowHandler(taskId, index)) return;

  appState.workflowBusy = true;
  state.message = `正在执行第 ${index + 1} 步：${profile.steps[index][0]}…`;
  renderWorkflowPanel("busy");
  try {
    const payload = await performWorkflowStep(taskId, index);
    state.completed.add(index);
    state.message = taskId === "task1" && index === 5
      ? `第${payload.result?.placement_sequence_number}个药盒已进入槽位` +
        `${payload.result?.slot_id}；` +
        (payload.result?.next_slot_id == null
          ? "20个槽位已全部完成。"
          : `下一次使用槽位${payload.result.next_slot_id}。`)
      : `第 ${index + 1} 步已完成，机械臂已停住。`;
    showToast(state.message, "info");
    if (index < profile.steps.length - 1) state.selected = index + 1;
  } catch (error) {
    state.message = `第 ${index + 1} 步失败：${readableError(error)}`;
    showToast(state.message, "error");
    appState.workflowBusy = false;
    renderWorkflowPanel("error");
    return;
  }
  appState.workflowBusy = false;
  syncWorkflowAutoRange({ resetStart: true, resetEnd: true });
  renderWorkflowPanel();
}

function requestWorkflowRangeStop() {
  if (!appState.workflowAutoRunning) return;
  appState.workflowAutoStopRequested = true;
  activeWorkflowState().message = "已请求停止；当前步骤完成后不会执行下一步。";
  renderWorkflowPanel("busy");
}

async function executeWorkflowRange() {
  if (appState.workflowBusy) return;
  const selection = selectedWorkflowRange();
  if (!selection.valid) {
    showToast(selection.reason, "error");
    renderWorkflowPanel("error");
    return;
  }
  const { taskId, startIndex, endIndex } = selection;
  const profile = TASK_PROFILES[taskId];
  const state = activeWorkflowState();
  appState.workflowBusy = true;
  appState.workflowAutoRunning = true;
  appState.workflowAutoStopRequested = false;
  let finalStatus = "idle";
  let stoppedAfterStep = null;

  try {
    for (let index = startIndex; index <= endIndex; index += 1) {
      state.selected = index;
      state.message =
        `自动执行 ${startIndex + 1}→${endIndex + 1}：` +
        `正在执行第 ${index + 1} 步 ${profile.steps[index][0]}…`;
      renderWorkflowPanel("busy");
      await performWorkflowStep(taskId, index);
      state.completed.add(index);
      if (appState.workflowAutoStopRequested) {
        stoppedAfterStep = index;
        break;
      }
    }
    state.selected = stoppedAfterStep ?? endIndex;
    state.message = stoppedAfterStep === null
      ? `第 ${startIndex + 1} 至 ${endIndex + 1} 步已全部自动执行完成。`
      : `自动执行已在第 ${stoppedAfterStep + 1} 步完成后停止。`;
    showToast(state.message, "info");
  } catch (error) {
    const failedIndex = state.selected;
    state.message =
      `自动执行已在第 ${failedIndex + 1} 步停止：${readableError(error)}`;
    showToast(state.message, "error");
    finalStatus = "error";
  } finally {
    appState.workflowAutoRunning = false;
    appState.workflowAutoStopRequested = false;
    appState.workflowBusy = false;
    renderWorkflowPanel(finalStatus);
  }
}

function setupOverlayResize() {
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      drawDetectionOverlay();
      updateFixedAxisMarkerIndicator();
    });
    observer.observe(el["camera-stage"]);
  } else {
    window.addEventListener("resize", drawDetectionOverlay);
  }
}

function startPolling() {
  window.clearInterval(appState.statusTimer);
  window.clearTimeout(appState.frameTimer);
  window.clearInterval(appState.recordingTimer);
  window.clearInterval(appState.teleopTimer);
  window.clearInterval(appState.cartesianJogTimer);
  window.clearInterval(appState.actRolloutTimer);
  window.clearInterval(appState.suctionTimer);
  window.clearInterval(appState.baseTrajectoryTimer);
  appState.statusTimer = window.setInterval(() => {
    runExclusivePoll("status", refreshStatus);
  }, POLL_INTERVAL_MS);
  scheduleFramePoll(0);
  appState.recordingTimer = window.setInterval(() => {
    if (appState.serviceOnline) {
      runExclusivePoll("recording", refreshRecordingStatus);
    }
  }, RECORDING_POLL_INTERVAL_MS);
  appState.baseTrajectoryTimer = window.setInterval(() => {
    if (appState.serviceOnline) {
      runExclusivePoll("baseTrajectory", refreshBaseTrajectoryStatus);
    }
  }, RECORDING_POLL_INTERVAL_MS);
  appState.teleopTimer = window.setInterval(() => {
    if (appState.serviceOnline) {
      runExclusivePoll("teleop", () =>
        Promise.allSettled([
          refreshTeleopStatus(),
          refreshEngineeringGripperLock(),
        ])
      );
    }
  }, TELEOP_POLL_INTERVAL_MS);
  appState.cartesianJogTimer = window.setInterval(() => {
    if (appState.serviceOnline && !appState.cartesianJogBusy) {
      runExclusivePoll("cartesianJog", () =>
        Promise.allSettled([
          refreshCartesianJogStatus(),
          refreshRightArmHomeStatus(),
        ])
      );
    }
  }, CARTESIAN_JOG_POLL_INTERVAL_MS);
  appState.actRolloutTimer = window.setInterval(() => {
    if (appState.serviceOnline && !appState.actRolloutBusy) {
      runExclusivePoll("actRollout", refreshActRolloutStatus);
    }
  }, ACT_ROLLOUT_POLL_INTERVAL_MS);
  appState.suctionTimer = window.setInterval(() => {
    if (appState.serviceOnline && !appState.suctionBusy) {
      runExclusivePoll("suction", refreshSuctionStatus);
    }
  }, SUCTION_SYNC_INTERVAL_MS);
}

function runExclusivePoll(key, operation) {
  if (appState.pollInFlight[key]) return;
  appState.pollInFlight[key] = true;
  Promise.resolve()
    .then(operation)
    .catch(() => {})
    .finally(() => {
      appState.pollInFlight[key] = false;
    });
}

function scheduleFramePoll(delay = currentFrameInterval()) {
  window.clearTimeout(appState.frameTimer);
  appState.frameTimer = window.setTimeout(() => {
    if (appState.serviceOnline && !appState.detectionBusy) refreshFrame(false);
    scheduleFramePoll();
  }, delay);
}

function currentFrameInterval() {
  if (document.hidden) return BACKGROUND_FRAME_INTERVAL_MS;
  return recordingIsActive()
    ? RECORDING_FRAME_INTERVAL_MS
    : FRAME_INTERVAL_MS;
}

function loadedFrameRequestIsCurrent() {
  try {
    const loadedUrl = new URL(el["camera-frame"].currentSrc, window.location.href);
    return Number(loadedUrl.searchParams.get("frame_request")) ===
      appState.frameRequestId;
  } catch (_error) {
    return false;
  }
}

function clearFrameRequestTimeout() {
  window.clearTimeout(appState.frameRequestTimeout);
  appState.frameRequestTimeout = null;
}

function requestCameraFrame(url, timeoutDetail) {
  clearFrameRequestTimeout();
  const requestId = ++appState.frameRequestId;
  const separator = url.includes("?") ? "&" : "?";
  appState.frameRequestActive = true;
  el["camera-frame"].src =
    `${url}${separator}frame_request=${requestId}&ui=${Date.now()}`;
  appState.frameRequestTimeout = window.setTimeout(() => {
    if (appState.frameRequestId !== requestId) return;
    ++appState.frameRequestId;
    appState.frameRequestActive = false;
    el["camera-frame"].removeAttribute("src");
    setCameraAvailability(false, timeoutDetail);
  }, REQUEST_TIMEOUT_MS);
}

async function refreshApplication() {
  const results = await Promise.allSettled([
    loadHealth(),
    loadProfile(),
    loadFixedSuctionAxisStatus(),
    refreshStatus(),
    loadRecordings(),
    loadBaseTrajectories(),
    refreshBaseTrajectoryStatus(),
    refreshActRolloutStatus(),
    refreshTeleopStatus(),
    refreshCartesianJogStatus(),
    refreshRightArmHomeStatus(),
    refreshSuctionStatus(),
    refreshEngineeringGripperLock(),
    refreshTask1PickStatus(),
    refreshTask2PickStatus(),
    refreshTask3PickStatus(),
  ]);
  const anySucceeded = results.some((result) => result.status === "fulfilled");
  if (anySucceeded) refreshFrame(false);
}

function gripperLockArmLabel(arm) {
  return arm === "right" ? "右手" : "左手";
}

function renderEngineeringGripperLock(message = "") {
  const lock = appState.gripperLock;
  const arm = lock.arm === "right" ? "right" : "left";
  const armLabel = gripperLockArmLabel(arm);
  const chip = el["engineering-gripper-lock-chip"];
  const select = el["engineering-gripper-lock-arm"];
  const button = el["engineering-gripper-lock-toggle"];
  const status = el["engineering-gripper-lock-status"];
  if (!chip || !select || !button || !status) return;
  select.value = arm;
  select.disabled = lock.busy;
  document.querySelectorAll("[data-gripper-lock-arm]").forEach((button) => {
    const active = button.dataset.gripperLockArm === arm;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = lock.busy;
  });
  button.disabled = lock.busy || !lock.available;
  button.textContent = lock.busy ? "正在切换…" : lock.enabled ? "关闭夹爪锁" : "启用夹爪锁";
  chip.textContent = lock.available ? lock.enabled ? `${armLabel}已锁定` : "未启用" : "9999 不可用";
  chip.classList.toggle("is-enabled", lock.enabled && lock.available);
  status.textContent = message || (lock.available
    ? lock.enabled
      ? `${armLabel}夹爪信号已屏蔽并保持 0.0 mm 收紧；另一侧继续完整开度跟随。`
      : "夹爪锁已关闭；左右夹爪均按原始映射完整跟随。"
    : "无法读取 9999 的夹爪锁服务。自动化操作前请确认状态。"
  );
}

async function refreshEngineeringGripperLock() {
  if (appState.gripperLock.busy) return;
  try {
    const payload = await requestJSON("/api/gripper-signal-lock");
    appState.gripperLock.enabled = payload.enabled === true;
    appState.gripperLock.arm = payload.arm === "right" ? "right" : "left";
    appState.gripperLock.available = true;
    renderEngineeringGripperLock();
  } catch (error) {
    appState.gripperLock.available = false;
    renderEngineeringGripperLock(`夹爪锁状态读取失败：${error.message}`);
  }
}

async function setEngineeringGripperLock(enabled) {
  const lock = appState.gripperLock;
  if (lock.busy) return;
  const arm = el["engineering-gripper-lock-arm"]?.value === "right" ? "right" : "left";
  if (enabled && !window.confirm(
    `确认将${gripperLockArmLabel(arm)}设为吸盘夹爪模式？\n\n该侧从臂夹爪将立即收紧并停止接收主臂夹爪开度。`
  )) return;
  lock.busy = true;
  renderEngineeringGripperLock();
  try {
    const payload = await requestJSON("/api/gripper-signal-lock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled, arm }),
    });
    lock.enabled = payload.enabled === true;
    lock.arm = payload.arm === "right" ? "right" : "left";
    lock.available = true;
    showToast(lock.enabled
      ? `${gripperLockArmLabel(lock.arm)}夹爪锁已启用`
      : "夹爪锁已关闭，左右夹爪恢复完整跟随"
    );
  } catch (error) {
    showToast(`夹爪锁切换失败：${error.message}`, "error");
  } finally {
    lock.busy = false;
    await refreshEngineeringGripperLock();
  }
}

function toggleEngineeringGripperLock() {
  return setEngineeringGripperLock(!appState.gripperLock.enabled);
}

async function loadHealth() {
  try {
    const payload = await requestJSON(API.health);
    if (!payload || payload.ok === false) throw new Error("服务健康检查未通过");

    setServiceAvailability(true);
    el["service-name"].textContent = payload.service || "medicine-agentic";
    el["service-version"].textContent = `version ${payload.version || "—"}`;

    const safety = payload.safety || {};
    const isOperatorControlled = safety.mode === "operator_teleop";
    const dryRunDisabled = safety.dry_run === false;
    const boundedActRolloutEnabled =
      safety.act_rollout_api === true &&
      safety.act_rollout_stop_semantics === "synchronous_hold_current";
    const motionEnabled = safety.motion_api === true;
    const teleopLauncherEnabled = safety.teleop_enable_api === true;
    const autonomousMotionDisabled =
      safety.autonomous_motion_api === false;
    const boundedTaskSkillEnabled =
      safety.bounded_task_skill_api === true;
    const directJointCommandsDisabled =
      safety.direct_joint_command_api === false;
    const chassisDisabled = safety.chassis_api === false;
    const navigationDisabled = safety.navigation_api === false;
    appState.safetyContractValid =
      isOperatorControlled &&
      dryRunDisabled &&
      motionEnabled &&
      teleopLauncherEnabled &&
      autonomousMotionDisabled &&
      directJointCommandsDisabled &&
      chassisDisabled &&
      navigationDisabled;
    appState.cartesianJogSafetyContractValid =
      safety.cartesian_jog_api === true &&
      safety.cartesian_jog_dry_run === false &&
      safety.direct_cartesian_step_api === true &&
      autonomousMotionDisabled &&
      directJointCommandsDisabled &&
      chassisDisabled &&
      navigationDisabled;
    appState.actRolloutSafetyContractValid =
      boundedActRolloutEnabled &&
      autonomousMotionDisabled &&
      directJointCommandsDisabled &&
      chassisDisabled &&
      navigationDisabled;
    el["safety-detail"].textContent =
      `task1.pick_carton=${String(boundedTaskSkillEnabled)} · ` +
      `act.rollout=${String(boundedActRolloutEnabled)} · ` +
      `generic_autonomous_motion=${String(safety.autonomous_motion_api)} · ` +
      `chassis_api=${String(safety.chassis_api)}`;

    if (
      !appState.safetyContractValid &&
      !appState.cartesianJogSafetyContractValid
    ) {
      setServiceAvailability(false, "安全模式不符合操作员控制契约");
      showToast("安全契约异常：页面已禁止视觉和遥操请求。", "error");
    } else if (!appState.safetyContractValid) {
      showToast("遥操启动器当前关闭；左臂定姿 XYZ 微调仍可使用。", "info");
    }
    updateDetectionButton();
    updateTeleopButton();
    return payload;
  } catch (error) {
    appState.safetyContractValid = false;
    appState.cartesianJogSafetyContractValid = false;
    appState.actRolloutSafetyContractValid = false;
    setServiceAvailability(false, readableError(error));
    updateDetectionButton();
    updateTeleopButton();
    throw error;
  }
}
async function refreshStatus() {
  try {
    const payload = await requestJSON(API.status);
    if (!payload || payload.ok === false) throw new Error("状态接口返回异常");
    setServiceAvailability(true);

    if (payload.timestamp) {
      el["status-time"].textContent = `同步 ${formatTime(payload.timestamp)}`;
    }

    renderGates(Array.isArray(payload.gates) ? payload.gates : []);

    const camera = payload.camera || {};
    const cameraReady =
      ["online", "ready", "streaming", "connected"].includes(
        String(camera.state || "").toLowerCase()
      ) || camera.state === true;
    if (!cameraReady && camera.error) {
      setCameraAvailability(false, camera.error);
    }

    if (payload.last_detection) {
      const normalized =
        payload.last_detection.detection || payload.last_detection;
      if (normalized?.task_id === appState.activeTask) {
        appState.detection = normalized;
        renderDetection(normalized);
      }
    }
    if (payload.recording) {
      renderRecording(payload.recording);
    }
    if (payload.act_inference) {
      renderActModel(payload.act_inference);
    }
    if (payload.act_rollout) {
      renderActRollout(payload.act_rollout);
    }
    if (payload.replay) {
      renderReplay(payload.replay);
    }
    if (payload.base_trajectory) {
      renderBaseTrajectory(payload.base_trajectory);
    }
    if (payload.teleop) {
      renderTeleop(payload.teleop);
    }
    if (payload.suction) {
      renderSuction(payload.suction);
    }
    if (payload.task1_pick) {
      renderTask1Pick(payload.task1_pick);
    }
    if (payload.task2_pick) {
      renderTask2Pick(payload.task2_pick);
    }
    if (payload.task3_pick) {
      renderTask3Pick(payload.task3_pick);
    }
    if (payload.task2_workflow) {
      renderTask2WorkflowStatus(payload.task2_workflow);
    }
    return payload;
  } catch (error) {
    renderGatesOffline();
    throw error;
  }
}

async function loadProfile() {
  try {
    const payload = await requestJSON(API.profile);
    if (!payload || payload.ok === false || !payload.camera) {
      throw new Error("相机配置不可用");
    }

    const camera = payload.camera;
    appState.profile = camera;
    renderProfile(camera);
    return payload;
  } catch (error) {
    renderProfileOffline(readableError(error));
    throw error;
  }
}

async function loadFixedSuctionAxisStatus() {
  try {
    const payload = await requestJSON(API.fixedSuctionAxisStatus);
    if (!payload || payload.ok === false || !payload.fixed_suction_axis) {
      throw new Error("固定双吸盘轴状态不可用");
    }
    appState.fixedSuctionAxis = payload.fixed_suction_axis;
    appState.fixedAxisCalibration = payload.calibration_session || null;
    renderFixedAxisCalibration();
    renderFixedAxis(appState.detection);
    return payload;
  } catch (error) {
    appState.fixedSuctionAxis = null;
    appState.fixedAxisCalibration = null;
    renderFixedAxisCalibration();
    renderFixedAxis(appState.detection);
    throw error;
  }
}

function cameraSourcePointFromPointer(event) {
  const stage = el["camera-stage"];
  const image = el["camera-frame"];
  const sourceWidth = Number(appState.profile?.color?.width || image.naturalWidth);
  const sourceHeight = Number(appState.profile?.color?.height || image.naturalHeight);
  if (!stage || !sourceWidth || !sourceHeight) return null;
  const rect = stage.getBoundingClientRect();
  const stageRatio = rect.width / rect.height;
  const sourceRatio = sourceWidth / sourceHeight;
  let displayWidth;
  let displayHeight;
  let offsetX;
  let offsetY;
  if (sourceRatio > stageRatio) {
    displayWidth = rect.width;
    displayHeight = rect.width / sourceRatio;
    offsetX = 0;
    offsetY = (rect.height - displayHeight) / 2;
  } else {
    displayHeight = rect.height;
    displayWidth = rect.height * sourceRatio;
    offsetX = (rect.width - displayWidth) / 2;
    offsetY = 0;
  }
  const localX = event.clientX - rect.left - offsetX;
  const localY = event.clientY - rect.top - offsetY;
  if (
    localX < 0 || localY < 0 ||
    localX >= displayWidth || localY >= displayHeight
  ) return null;
  return [
    (localX / displayWidth) * sourceWidth,
    (localY / displayHeight) * sourceHeight,
  ];
}

function updateFixedAxisMarkerIndicator() {
  const indicator = el["fixed-axis-marker-indicator"];
  const stage = el["camera-stage"];
  const marker = appState.fixedAxisCalibration?.marker?.pixel;
  const sourceWidth = Number(
    appState.profile?.color?.width || el["camera-frame"].naturalWidth
  );
  const sourceHeight = Number(
    appState.profile?.color?.height || el["camera-frame"].naturalHeight
  );
  if (!indicator || !stage || !Array.isArray(marker) || !sourceWidth || !sourceHeight) {
    if (indicator) indicator.style.display = "none";
    return;
  }
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  const sourceRatio = sourceWidth / sourceHeight;
  const stageRatio = width / height;
  const displayWidth = sourceRatio > stageRatio ? width : height * sourceRatio;
  const displayHeight = sourceRatio > stageRatio ? width / sourceRatio : height;
  const offsetX = (width - displayWidth) / 2;
  const offsetY = (height - displayHeight) / 2;
  indicator.style.left = `${offsetX + Number(marker[0]) / sourceWidth * displayWidth}px`;
  indicator.style.top = `${offsetY + Number(marker[1]) / sourceHeight * displayHeight}px`;
  indicator.style.display = "block";
}

function renderFixedAxisCalibration() {
  if (!el["fixed-axis-calibration-state"]) return;
  const session = appState.fixedAxisCalibration || {};
  const marker = session.marker;
  const samples = session.samples || {};
  const preview = session.preview;
  const jogReady = appState.cartesianJog?.enabled === true &&
    appState.cartesianJog?.busy !== true;
  const busy = appState.fixedAxisCalibrationBusy;
  el["camera-stage"].classList.toggle(
    "is-marker-picking",
    appState.fixedAxisMarkerPicking
  );
  el["fixed-axis-lock-marker"].disabled = busy || !appState.cameraOnline;
  el["fixed-axis-sample-a"].disabled = busy || !marker || !jogReady;
  el["fixed-axis-sample-b"].disabled = busy || !marker || !jogReady;
  el["fixed-axis-commit"].disabled = busy || preview?.valid !== true || session.saved === true;
  if (session.saved) {
    el["fixed-axis-calibration-state"].textContent = "已保存并启用";
  } else if (preview) {
    el["fixed-axis-calibration-state"].textContent = preview.valid
      ? "校验通过 · 待保存"
      : "校验未通过 · 请重新采样";
  } else if (samples.A || samples.B) {
    el["fixed-axis-calibration-state"].textContent =
      `已记录 ${samples.A ? "A" : ""}${samples.A && samples.B ? "/" : ""}${samples.B ? "B" : ""}`;
  } else if (marker) {
    el["fixed-axis-calibration-state"].textContent =
      `方块已锁定 (${marker.pixel[0]}, ${marker.pixel[1]})`;
  } else {
    el["fixed-axis-calibration-state"].textContent = "尚未锁定方块";
  }
  if (appState.fixedAxisMarkerPicking) {
    el["fixed-axis-calibration-detail"].textContent =
      "请直接点击相机画面中的橙色方块顶面中心。";
  } else if (preview) {
    el["fixed-axis-calibration-detail"].textContent =
      `平面间距 ${Number(preview.measured_spacing_mm).toFixed(1)} mm · ` +
      `姿态差 ${Number(preview.orientation_delta_deg).toFixed(2)}° · ` +
      `Z 差 ${Number(preview.ignored_z_delta_mm).toFixed(1)} mm（已忽略）`;
  } else if (marker) {
    el["fixed-axis-calibration-detail"].textContent =
      "保持方块不动；只需依次将两个吸盘中心在 XY 对准它，Z 高度可不同。";
  } else {
    el["fixed-axis-calibration-detail"].textContent =
      "点击“锁定方块”后，再在相机画面中点击橙色方块顶面中心。";
  }
  updateFixedAxisMarkerIndicator();
}

function beginFixedAxisMarkerLock() {
  if (appState.fixedAxisCalibrationBusy || !appState.cameraOnline) return;
  releaseDetectionFrame();
  appState.fixedAxisMarkerPicking = true;
  renderFixedAxisCalibration();
}

async function lockFixedAxisMarkerFromPointer(event) {
  if (!appState.fixedAxisMarkerPicking || appState.fixedAxisCalibrationBusy) return;
  const point = cameraSourcePointFromPointer(event);
  if (!point) {
    showToast("请点击实际相机画面内部。", "error");
    return;
  }
  appState.fixedAxisMarkerPicking = false;
  appState.fixedAxisCalibrationBusy = true;
  renderFixedAxisCalibration();
  try {
    const payload = await requestJSON(
      API.fixedSuctionAxisLockMarker,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pixel_x: point[0], pixel_y: point[1] }),
      },
      20000
    );
    appState.fixedSuctionAxis = payload.fixed_suction_axis;
    appState.fixedAxisCalibration = payload.calibration_session;
    showToast("橙色方块坐标已锁定。", "info");
  } catch (error) {
    showToast(`方块锁定失败：${readableError(error)}`, "error");
  } finally {
    appState.fixedAxisCalibrationBusy = false;
    renderFixedAxisCalibration();
  }
}

async function sampleFixedAxisCup(cup) {
  if (appState.fixedAxisCalibrationBusy || !["A", "B"].includes(cup)) return;
  appState.fixedAxisCalibrationBusy = true;
  renderFixedAxisCalibration();
  try {
    const payload = await requestJSON(API.fixedSuctionAxisSampleCup, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cup }),
    });
    appState.fixedSuctionAxis = payload.fixed_suction_axis;
    appState.fixedAxisCalibration = payload.calibration_session;
    showToast(`吸盘 ${cup} 的法兰位置已记录。`, "info");
  } catch (error) {
    showToast(`吸盘 ${cup} 记录失败：${readableError(error)}`, "error");
  } finally {
    appState.fixedAxisCalibrationBusy = false;
    renderFixedAxisCalibration();
  }
}

async function commitFixedAxisCalibration() {
  if (appState.fixedAxisCalibrationBusy) return;
  appState.fixedAxisCalibrationBusy = true;
  renderFixedAxisCalibration();
  try {
    const payload = await requestJSON(API.fixedSuctionAxisCommit, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    appState.fixedSuctionAxis = payload.fixed_suction_axis;
    appState.fixedAxisCalibration = payload.calibration_session;
    renderFixedAxis(appState.detection);
    showToast("固定双吸盘轴标定已保存并启用。", "info");
  } catch (error) {
    showToast(`标定保存失败：${readableError(error)}`, "error");
  } finally {
    appState.fixedAxisCalibrationBusy = false;
    renderFixedAxisCalibration();
  }
}

function refreshFrame(userInitiated) {
  if (
    recordingIsActive() &&
    appState.recording?.preview_ready !== true
  ) {
    return;
  }
  if (appState.frameRequestActive) return;
  if (
    !userInitiated &&
    !recordingIsActive() &&
    appState.displayingDetectionFrame &&
    Date.now() < appState.detectionFramePinnedUntil
  ) {
    return;
  }
  if (appState.displayingDetectionFrame) releaseDetectionFrame();
  requestCameraFrame(
    API.frame,
    userInitiated ? "手动画面请求超时" : "实时画面请求超时"
  );
}

async function runDetection() {
  const profile = TASK_PROFILES[appState.activeTask];
  if (!profile?.available) {
    showToast(profile?.unavailableMessage || "当前任务识别 profile 尚未接入。", "error");
    return;
  }
  if (!appState.serviceOnline) {
    showToast("服务离线，无法运行识别。", "error");
    return;
  }
  if (recordingIsActive()) {
    showToast("录制期间暂停药盒识别，避免占用标定帧。", "error");
    return;
  }

  setDetectionBusy(true);
  try {
    const payload = await requestJSON(
      API.detect,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: appState.activeTask }),
      },
      20000
    );

    if (!payload || payload.ok === false || !payload.detection) {
      throw new Error(payload?.error || payload?.message || "识别接口返回异常");
    }

    appState.detection = payload.detection;
    renderDetection(payload.detection);
    showDetectionFrame(payload.detection);

    if (payload.detection.target_ready === true) {
      showToast("药盒识别及三维定位完成。");
    } else if (payload.detection.detected_2d === true) {
      showToast("二维识别完成；三维目标仍被门禁阻塞。");
    } else {
      showToast("识别完成，但没有通过门限的药盒。", "error");
    }
  } catch (error) {
    renderDetectionError(readableError(error));
    showToast(`识别失败：${readableError(error)}`, "error");
  } finally {
    setDetectionBusy(false);
  }
}

function actRolloutIsActive() {
  return ["starting", "running", "stopping"].includes(
    String(appState.actRollout?.state || "")
  );
}

function syncActExecuteStepsControl() {
  const input = el["act-execute-steps"];
  if (!input) return;
  const reportedHorizon = Number(appState.actRollout?.horizon);
  const horizon = Number.isInteger(reportedHorizon) && reportedHorizon > 0
    ? reportedHorizon
    : Math.max(1, Number(input.max) || 20);
  input.max = String(horizon);
  const selected = Math.min(
    horizon,
    Math.max(1, Math.round(Number(input.value) || horizon))
  );
  input.value = String(selected);
  el["act-execute-steps-value"].value = String(selected);
  el["act-execute-steps-value"].textContent = String(selected);
  el["act-execute-steps-total"].textContent = `/ ${horizon}`;
  el["act-execute-steps-detail"].textContent =
    `执行预测动作的前 ${selected} 帧，然后重新读取三路画面并推理`;
}

function updateActRolloutButtons() {
  const rollout = appState.actRollout || {};
  const teleop = appState.teleop || {};
  const jog = appState.cartesianJog || {};
  const blockedByOtherControl =
    recordingIsActive() ||
    baseTrajectoryIsActive() ||
    !appState.actRolloutSafetyContractValid ||
    appState.replay?.active === true ||
    teleop.running === true ||
    teleop.busy === true ||
    teleop.desired === true ||
    jog.enabled === true ||
    jog.busy === true;
  el["start-act-rollout"].disabled =
    !appState.serviceOnline ||
    appState.actRolloutBusy ||
    rollout.enabled !== true ||
    actRolloutIsActive() ||
    blockedByOtherControl;
  el["stop-act-rollout"].disabled =
    !appState.serviceOnline ||
    appState.actRolloutBusy ||
    !actRolloutIsActive() ||
    rollout.state === "stopping";
  el["act-execute-steps"].disabled =
    !appState.serviceOnline ||
    appState.actRolloutBusy ||
    rollout.enabled !== true ||
    actRolloutIsActive() ||
    baseTrajectoryIsActive();
}

function renderActModel(inference) {
  if (!inference || typeof inference !== "object") return;
  appState.actInference = inference;
  const select = el["act-model-select"];
  const state = el["act-model-state"];
  const container = select.closest(".act-model-selector");
  const profiles = Array.isArray(inference.profiles?.items)
    ? inference.profiles.items
    : [];
  if (profiles.length) {
    const signature = profiles.map((profile) => `${profile.id}:${profile.label}`).join("|");
    if (select.dataset.profileSignature !== signature) {
      select.replaceChildren(
        ...profiles.map((profile) => {
          const option = document.createElement("option");
          option.value = profile.id;
          option.textContent = profile.label;
          return option;
        })
      );
      select.dataset.profileSignature = signature;
    }
  }
  const active = String(
    inference.profile_id || inference.profiles?.active || "act1"
  );
  if (!appState.actModelSwitchBusy) select.value = active;
  const absolute = inference.action_representation === "absolute_joint_target";
  const ready = inference.ready === true && absolute;
  container.classList.toggle("is-ready", ready);
  container.classList.toggle("is-error", inference.ready === false || !absolute);
  select.disabled =
    appState.actModelSwitchBusy || appState.actRollout?.active === true;
  state.textContent = appState.actModelSwitchBusy
    ? "正在切换并校验…"
    : ready
      ? "绝对位姿 · 已就绪"
      : inference.ready === false
        ? "服务不可用"
        : "正在校验";
}

async function switchActModel() {
  const select = el["act-model-select"];
  const requested = select.value;
  const previous = String(
    appState.actInference?.profile_id ||
      appState.actInference?.profiles?.active ||
      "act1"
  );
  if (!requested || requested === previous || appState.actModelSwitchBusy) return;
  appState.actModelSwitchBusy = true;
  renderActModel(appState.actInference || { profile_id: previous, ready: null });
  try {
    const payload = await requestJSON(API.actModel, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: requested }),
    });
    if (!payload?.act_inference) throw new Error("模型切换接口返回异常");
    renderActModel(payload.act_inference);
    showToast(`${payload.act_inference.profile_label} 已切换为绝对位姿推理`, "info");
  } catch (error) {
    select.value = previous;
    showToast(`模型切换失败：${readableError(error)}`, "error");
  } finally {
    appState.actModelSwitchBusy = false;
    if (appState.actInference) renderActModel(appState.actInference);
  }
}

function renderActRollout(rollout) {
  const previousState = String(appState.actRollout?.state || "");
  appState.actRollout = rollout || {
    enabled: false,
    active: false,
    state: "idle",
  };
  const state = String(appState.actRollout.state || "idle");
  const labels = {
    idle: "已停止",
    starting: "预检中",
    running: "推理中",
    stopping: "停止中",
    stopped: "已停止",
    error: "已停止 / 异常",
  };
  el["act-rollout-state"].textContent = labels[state] || state;
  el["act-rollout-state"].className =
    `act-rollout-state act-rollout-state--${state}`;
  el["act-rollout-inferences"].textContent = String(
    Number(appState.actRollout.inference_count || 0)
  );
  el["act-rollout-commands"].textContent = String(
    Number(appState.actRollout.command_count || 0)
  );
  const latency = Number(appState.actRollout.last_inference_ms);
  el["act-rollout-latency"].textContent =
    Number.isFinite(latency) ? `${latency.toFixed(1)} ms` : "—";

  const commandHz = Number(appState.actRollout.command_hz);
  const executeSteps = Number(
    appState.actRollout.execute_steps_per_inference
  );
  if (
    Number.isInteger(executeSteps) &&
    executeSteps > 0 &&
    (actRolloutIsActive() ||
      el["act-execute-steps"].dataset.userSet !== "true")
  ) {
    el["act-execute-steps"].value = String(executeSteps);
  }
  syncActExecuteStepsControl();
  const rolloutCadence =
    Number.isFinite(commandHz) && Number.isFinite(executeSteps)
      ? `每轮按 ${commandHz.toFixed(0)} Hz 完整执行 ${executeSteps} 帧后重新观察。`
      : "每轮完整执行模型返回的动作段后重新观察。";
  const debugLogPath = String(appState.actRollout.debug_log_path || "");
  const debugLogSuffix = debugLogPath ? ` 调试日志：${debugLogPath}` : "";

  const messages = {
    idle: "模型已就绪后可开始；开始按钮会让双臂执行受限的闭环动作。",
    starting: "正在检查互锁、起始姿态、相机时效和第一段模型输出；尚未通过前不会发动作。",
    running: `闭环推理运行中；${rolloutCadence}停止按钮会同步保持当前位置。`,
    stopping: "停止请求已抢占后续动作，正在确认双臂当前位置保持。",
    stopped: appState.actRollout.hold_confirmed
      ? "模型推理已停止，双臂已确认保持当前位置。"
      : "模型推理已停止。",
    error: `控制器已停止：${appState.actRollout.error || "未知异常"}${debugLogSuffix}`,
  };
  el["act-rollout-message"].textContent = messages[state] || state;
  updateActRolloutButtons();
  updateRecordingButtons();
  updateTeleopButton();
  updateBaseTrajectoryButtons();

  if (state !== previousState) {
    if (state === "running") {
      showToast("ACT 连续推理已开始：从当前状态推理，动作按 30 Hz 原值执行。");
    } else if (state === "stopped" && appState.actRollout.hold_confirmed) {
      showToast("ACT 已停止，机械臂保持当前位置。");
    } else if (state === "error") {
      showToast(messages.error, "error");
    }
  }
}

async function refreshActRolloutStatus() {
  const payload = await requestJSON(API.actRolloutStatus);
  if (!payload?.act_rollout) {
    throw new Error("ACT rollout 状态接口返回异常");
  }
  renderActRollout(payload.act_rollout);
  return payload;
}

async function startActRollout() {
  if (
    !appState.serviceOnline ||
    appState.actRolloutBusy ||
    actRolloutIsActive() ||
    baseTrajectoryIsActive()
  ) return;
  appState.actRolloutBusy = true;
  updateActRolloutButtons();
  try {
    const executeSteps = Number(el["act-execute-steps"].value);
    if (!Number.isInteger(executeSteps) || executeSteps < 1) {
      throw new Error("每轮执行帧数无效");
    }
    const payload = await requestJSON(API.actRolloutStart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ execute_steps_per_inference: executeSteps }),
    }, 8000);
    if (!payload?.act_rollout) {
      throw new Error("ACT rollout 启动接口返回异常");
    }
    renderActRollout(payload.act_rollout);
    showToast("ACT 正在进行无动作预检；通过后才会进入闭环执行。");
  } catch (error) {
    showToast(`无法开始 ACT：${readableError(error)}`, "error");
    await refreshActRolloutStatus().catch(() => {});
  } finally {
    appState.actRolloutBusy = false;
    updateActRolloutButtons();
  }
}

async function stopActRollout() {
  if (
    !appState.serviceOnline ||
    appState.actRolloutBusy ||
    !actRolloutIsActive()
  ) return;
  appState.actRolloutBusy = true;
  updateActRolloutButtons();
  try {
    const payload = await requestJSON(API.actRolloutStop, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    }, 5000);
    if (!payload?.act_rollout) {
      throw new Error("ACT rollout 停止接口返回异常");
    }
    renderActRollout(payload.act_rollout);
    showToast(
      payload.act_rollout.hold_confirmed
        ? "停止已生效，机械臂保持当前位置。"
        : "停止已生效；控制器在预检阶段，尚未发送动作。"
    );
  } catch (error) {
    showToast(`ACT 停止失败：${readableError(error)}；请使用实体急停。`, "error");
    await refreshActRolloutStatus().catch(() => {});
  } finally {
    appState.actRolloutBusy = false;
    updateActRolloutButtons();
  }
}

async function startRecording() {
  if (
    !appState.serviceOnline ||
    appState.recordingBusy ||
    recordingIsActive() ||
    baseTrajectoryIsActive()
  ) {
    return;
  }
  const label = String(el["recording-label"].value || "").trim() || "recording";
  const purpose = String(el["recording-purpose"].value || "act_bimanual");
  setRecordingBusy(true);
  try {
    const payload = await requestJSON(API.recordingStart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, purpose }),
    });
    if (!payload?.recording) throw new Error("录制启动接口返回异常");
    renderRecording(payload.recording);
    showToast("录制器正在连接相机和所需机械臂反馈。");
  } catch (error) {
    showToast(`无法开始录制：${readableError(error)}`, "error");
  } finally {
    setRecordingBusy(false);
  }
}

async function stopRecording() {
  if (!recordingIsActive() || appState.recordingBusy) return;
  setRecordingBusy(true);
  try {
    const payload = await requestJSON(API.recordingStop, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!payload?.recording) throw new Error("录制停止接口返回异常");
    renderRecording(payload.recording);
    showToast("正在停止并封装录制文件。");
  } catch (error) {
    showToast(`无法停止录制：${readableError(error)}`, "error");
  } finally {
    setRecordingBusy(false);
  }
}

function handleGlobalShortcut(event) {
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
  if (
    event.target instanceof Element &&
    event.target.closest("input, textarea, select, [contenteditable='true']")
  ) {
    return;
  }
  if (event.code === "KeyF") {
    event.preventDefault();
    setSuction(appState.suction?.engaged !== true);
  } else if (event.code === "KeyO") {
    event.preventDefault();
    startRecording();
  } else if (event.code === "KeyP") {
    event.preventDefault();
    stopRecording();
  }
}

async function startTeleop() {
  if (
    !appState.serviceOnline ||
    !appState.safetyContractValid ||
    appState.teleopBusy ||
    appState.teleop?.running ||
    appState.teleop?.busy ||
    actRolloutIsActive() ||
    recordingIsActive() ||
    baseTrajectoryIsActive()
  ) {
    return;
  }
  setTeleopBusy(true);
  try {
    const mode = el["teleop-operator-mode"].value || "dual";
    const holdPose = el["teleop-hold-pose"].value || null;
    const payload = await requestJSON(API.teleopStart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirm: "START_FOLLOW",
        area_clear: true,
        estop_ready: true,
        initial_pose_aligned: true,
        mode,
        hold_pose: mode === "dual" ? null : holdPose,
      }),
    });
    if (!payload?.teleop) throw new Error("遥操启动接口返回异常");
    appState.teleopModeDirty = false;
    renderTeleop(payload.teleop);
    showToast(
      payload.already_running
        ? "Follow 已经在运行。"
        : mode === "dual"
          ? "正在后台检查端点并启动双臂 Follow。"
          : "静止臂已到预设位姿，正在启动单臂 Follow。"
    );
  } catch (error) {
    showToast(`无法启动遥操：${readableError(error)}`, "error");
    await refreshTeleopStatus().catch(() => {});
  } finally {
    setTeleopBusy(false);
  }
}

async function stopTeleop() {
  const teleop = appState.teleop || {};
  const canStop =
    teleop.enabled === true &&
    (
      teleop.running === true ||
      teleop.desired === true ||
      teleop.busy === true ||
      ["starting", "waiting-endpoints", "restarting", "hard-restarting", "unknown", "error"].includes(
        String(teleop.state || "")
      )
    );
  if (!appState.serviceOnline || appState.teleopBusy || !canStop) return;
  setTeleopBusy(true);
  try {
    const payload = await requestJSON(API.teleopStop, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "STOP_FOLLOW" }),
    });
    if (!payload?.teleop) throw new Error("遥操停止接口返回异常");
    renderTeleop(payload.teleop);
    showToast("正在正常停止并解除 Follow supervisor。");
  } catch (error) {
    showToast(`无法确认遥操已停止：${readableError(error)}`, "error");
    await refreshTeleopStatus().catch(() => {});
  } finally {
    setTeleopBusy(false);
  }
}

function resetHardRestartTeleopConfirmation() {
  if (appState.teleopHardRestartConfirmTimer !== null) {
    window.clearTimeout(appState.teleopHardRestartConfirmTimer);
    appState.teleopHardRestartConfirmTimer = null;
  }
  appState.teleopHardRestartArmed = false;
  const button = el["hard-restart-teleop"];
  button.classList.remove("is-armed");
  button.querySelector("strong").textContent = "彻底重启遥操";
  button.querySelector("small").textContent = "重建主臂、执行臂与 Follow";
}

async function hardRestartTeleop() {
  const button = el["hard-restart-teleop"];
  if (button.disabled) return;
  if (!appState.teleopHardRestartArmed) {
    appState.teleopHardRestartArmed = true;
    button.classList.add("is-armed");
    button.querySelector("strong").textContent = "再次点击确认重启";
    button.querySelector("small").textContent = "确认区域无人、急停可用、主臂保持不动";
    appState.teleopHardRestartConfirmTimer = window.setTimeout(() => {
      resetHardRestartTeleopConfirmation();
      updateTeleopButton();
    }, 8000);
    showToast("请确认机械臂区域无人、急停可用且主臂保持不动，再次点击执行彻底重启。");
    return;
  }

  resetHardRestartTeleopConfirmation();
  setTeleopBusy(true);
  try {
    const payload = await requestJSON(API.teleopHardRestart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirm: "HARD_RESTART_TELEOP",
        area_clear: true,
        estop_ready: true,
        master_arms_stable: true,
      }),
    });
    if (!payload?.teleop) throw new Error("遥操彻底重启接口返回异常");
    renderTeleop(payload.teleop);
    showToast("正在后台重建主臂、执行臂与 Follow，请勿移动主臂。");
  } catch (error) {
    showToast(`无法彻底重启遥操：${readableError(error)}`, "error");
    await refreshTeleopStatus().catch(() => {});
  } finally {
    setTeleopBusy(false);
  }
}

async function refreshTeleopStatus() {
  const payload = await requestJSON(API.teleopStatus);
  if (!payload?.teleop) throw new Error("遥操状态接口返回异常");
  const previousState = String(appState.teleop?.state || "");
  renderTeleop(payload.teleop);
  const nextState = String(payload.teleop.state || "");
  if (previousState === "starting" && nextState === "running") {
    showToast("Follow 遥操已启动。");
  } else if (
    previousState === "starting" &&
    nextState === "error"
  ) {
    showToast(
      `遥操启动失败：${payload.teleop.error || "请检查主从臂端点"}`,
      "error"
    );
  } else if (
    previousState === "stopping" &&
    nextState === "stopped"
  ) {
    showToast("Follow 已正常停止。");
  } else if (
    previousState === "hard-restarting" &&
    nextState === "running"
  ) {
    showToast("主臂、执行臂与 Follow 已彻底重启，遥操已恢复。");
  } else if (
    previousState === "hard-restarting" &&
    nextState === "error"
  ) {
    showToast(
      `彻底重启失败：${payload.teleop.error || "请检查主从臂连接"}`,
      "error"
    );
  }
  return payload;
}

function setTeleopBusy(busy) {
  appState.teleopBusy = busy;
  updateTeleopButton();
  updateRecordingButtons();
  updateBaseTrajectoryButtons();
}

function updateTeleopModeControls() {
  const teleop = appState.teleop || {};
  const modeSelect = el["teleop-operator-mode"];
  const poseSelect = el["teleop-hold-pose"];
  const mode = modeSelect.value || "dual";
  const inactiveArm = mode === "left_only"
    ? "right"
    : mode === "right_only"
      ? "left"
      : null;
  const previousPose = poseSelect.value;
  const available = inactiveArm
    ? teleop.available_hold_poses?.[inactiveArm] || []
    : [];
  poseSelect.replaceChildren();
  if (!inactiveArm) {
    poseSelect.append(new Option("双臂模式无需选择", ""));
  } else if (!available.length) {
    poseSelect.append(new Option("该机械臂没有已保存位姿", ""));
  } else {
    available.forEach((name) => poseSelect.append(new Option(name, name)));
    const persisted = teleop.hold_pose_by_arm?.[inactiveArm];
    const preferred = available.includes(previousPose)
      ? previousPose
      : available.includes(persisted)
        ? persisted
        : available.includes("init_pose")
          ? "init_pose"
          : available[0];
    poseSelect.value = preferred;
  }
  const armLabel = inactiveArm === "left"
    ? "左臂"
    : inactiveArm === "right"
      ? "右臂"
      : "静止臂";
  el["teleop-hold-pose-label"].textContent = `${armLabel}预设位姿`;
  el["teleop-mode-detail"].textContent = inactiveArm
    ? `${armLabel}会先自动到所选位姿并持续保持；ACT 仍记录左右两臂，静止臂 action 使用实际保持反馈。`
    : "双臂均跟随；ACT 仍记录左右两臂。";
  const locked =
    appState.teleopBusy ||
    teleop.running === true ||
    teleop.desired === true ||
    teleop.busy === true ||
    recordingIsActive() ||
    baseTrajectoryIsActive();
  modeSelect.disabled = locked;
  poseSelect.disabled = locked || !inactiveArm || !available.length;
}

function updateTeleopButton() {
  const teleop = appState.teleop || {};
  const jog = appState.cartesianJog || {};
  el["start-teleop"].disabled =
    !appState.serviceOnline ||
    !appState.safetyContractValid ||
    appState.teleopBusy ||
    teleop.enabled !== true ||
    teleop.busy === true ||
    teleop.running === true ||
    teleop.desired === true ||
    teleop.lead_ready !== true ||
    teleop.follower_ready !== true ||
    teleop.safety?.latched === true ||
    cartesianJogEnabled(jog) ||
    cartesianJogServerBusy(jog) ||
    actRolloutIsActive() ||
    baseTrajectoryIsActive() ||
    recordingIsActive() ||
    (
      el["teleop-operator-mode"].value !== "dual" &&
      !el["teleop-hold-pose"].value
    );
  const stopRelevant =
    teleop.running === true ||
    teleop.desired === true ||
    teleop.busy === true ||
    ["starting", "waiting-endpoints", "restarting", "hard-restarting", "unknown", "error"].includes(
      String(teleop.state || "")
    );
  el["stop-teleop"].disabled =
    !appState.serviceOnline ||
    appState.teleopBusy ||
    teleop.enabled !== true ||
    !stopRelevant;
  const hardRestartDisabled =
    !appState.serviceOnline ||
    !appState.safetyContractValid ||
    appState.teleopBusy ||
    teleop.enabled !== true ||
    teleop.busy === true ||
    teleop.safety?.latched === true ||
    cartesianJogEnabled(jog) ||
    cartesianJogServerBusy(jog) ||
    actRolloutIsActive() ||
    appState.replay?.active === true ||
    baseTrajectoryIsActive() ||
    recordingIsActive();
  el["hard-restart-teleop"].disabled = hardRestartDisabled;
  el["hard-restart-teleop"].title = hardRestartDisabled
    ? "请先停止录制、回放、ACT 或微调，并等待当前遥操操作结束"
    : "两次点击确认后，重建主臂、执行臂与 Follow";
  if (hardRestartDisabled && appState.teleopHardRestartArmed) {
    resetHardRestartTeleopConfirmation();
  }
  updateTeleopModeControls();
  updateActRolloutButtons();
}

function renderTeleop(teleop) {
  appState.teleop = teleop || {
    enabled: false,
    state: "disabled",
    running: false,
  };
  const state = String(appState.teleop.state || "idle");
  if (!appState.teleopModeDirty || appState.teleop.running === true) {
    el["teleop-operator-mode"].value =
      appState.teleop.operator_mode || "dual";
  }
  const labels = {
    disabled: "未配置",
    idle: "未启动",
    stopped: "未启动",
    starting: "启动中",
    stopping: "停止中",
    "waiting-endpoints": "等待端点",
    restarting: "重启中",
    "hard-restarting": "彻底重启中",
    unknown: "状态异常",
    running: "运行中",
    error: "操作失败",
  };
  el["teleop-state"].textContent = labels[state] || state;
  el["teleop-state"].className =
    `teleop-state teleop-state--${escapeHTML(state)}`;
  el["teleop-message"].textContent =
    appState.teleop.error ||
    appState.teleop.message ||
    "等待操作员启动";
  const lead = appState.teleop.lead_host || "—";
  const leadState = appState.teleop.lead_ready ? "就绪" : "未就绪";
  const followerState = appState.teleop.follower_ready ? "就绪" : "未就绪";
  el["teleop-endpoints"].textContent =
    `主臂 ${lead} · ${leadState} / 执行臂 ${followerState}`;
  el["teleop-mode-chip"].textContent = appState.teleop.running
    ? "遥操运行中"
    : appState.teleop.busy
      ? "遥操启动中"
      : appState.teleop.desired
        ? "遥操状态异常"
        : "遥操未启动";
  el["teleop-mode-chip"].classList.toggle(
    "mode-chip--safe",
    appState.teleop.running === true
  );
  el["teleop-mode-chip"].classList.toggle(
    "mode-chip--locked",
    appState.teleop.running !== true
  );
  updateTeleopModeControls();
  updateTeleopButton();
  updateRecordingButtons();
  updateBaseTrajectoryButtons();
  updateCartesianJogControls();
}

async function refreshCartesianJogStatus() {
  try {
    const payload = await requestJSON(API.cartesianJogStatus);
    const jog = extractCartesianJog(payload);
    if (!jog) throw new Error("定姿微调状态接口返回异常");
    renderCartesianJog(jog);
    return payload;
  } catch (error) {
    showJogAlert(
      `无法读取左臂微调状态：${readableError(error)}`,
      false
    );
    if (!appState.cartesianJog) {
      el["jog-state"].textContent = "状态不可用";
      el["jog-state"].className = "jog-state jog-state--error";
      el["jog-status-message"].textContent =
        "状态未知，所有左臂微调按钮保持禁用。";
    }
    updateCartesianJogControls();
    throw error;
  }
}

async function refreshRightArmHomeStatus() {
  try {
    const payload = await requestJSON(API.rightArmResetHomeSkill);
    if (!payload?.ok || payload?.skill?.id !== "right_arm.reset_home") {
      throw new Error("右臂复位 Skill 状态接口返回异常");
    }
    appState.rightArmHome = payload.skill;
    updateCartesianJogControls();
    return payload;
  } catch (error) {
    appState.rightArmHome = null;
    updateCartesianJogControls();
    throw error;
  }
}

function extractCartesianJog(payload) {
  if (!payload || typeof payload !== "object") return null;
  const candidates = [
    payload.cartesian_jog,
    payload.jog,
    payload.data?.cartesian_jog,
    payload.data?.jog,
    payload.status,
    payload,
  ];
  return (
    candidates.find(
      (candidate) =>
        candidate && typeof candidate === "object" && !Array.isArray(candidate)
    ) || null
  );
}

async function refreshSuctionStatus() {
  try {
    const payload = await requestJSON(API.suctionStatus);
    if (!payload?.suction) throw new Error("吸盘状态接口返回异常");
    renderSuction(payload.suction);
    return payload;
  } catch (error) {
    appState.suction = {
      available: false,
      error: readableError(error),
    };
    renderSuction(appState.suction);
    throw error;
  }
}

function renderSuction(suction) {
  appState.suction = suction || {};
  const available = appState.suction.available === true;
  const engaged = appState.suction.engaged;
  el["suction-state"].textContent = !available
    ? "串口不可用"
    : engaged === true
      ? "已吸紧"
      : engaged === false
        ? "已释放"
        : "串口就绪 · 状态待首次命令确认";
  el["suction-detail"].textContent = !available
    ? appState.suction.error || "请检查双吸盘串口。"
    : "按 F 可切换吸紧/释放；吸紧后选择5 mm步长，点击 Z+ 四次完成20 mm试抬。";
  const blocked =
    !appState.serviceOnline || appState.suctionBusy || !available;
  el["suction-on"].disabled = blocked || engaged === true;
  el["suction-off"].disabled = blocked || engaged === false;
  updateTask1PickControls();
}

async function refreshTask1PickStatus() {
  try {
    const payload = await requestJSON(API.task1PickStatus);
    if (!payload?.task1_pick) throw new Error("自动抓取状态接口返回异常");
    renderTask1Pick(payload.task1_pick);
    return payload;
  } catch (error) {
    renderTask1Pick({
      enabled: false,
      ready: false,
      error: readableError(error),
    });
    throw error;
  }
}

async function refreshTask2PickStatus() {
  try {
    const payload = await requestJSON(API.task2PickStatus);
    if (!payload?.task2_pick) throw new Error("Task2 自动抓取状态接口返回异常");
    renderTask2Pick(payload.task2_pick);
    return payload;
  } catch (error) {
    renderTask2Pick({
      enabled: false,
      ready: false,
      error: readableError(error),
    });
    throw error;
  }
}

async function refreshTask3PickStatus() {
  try {
    const payload = await requestJSON(API.task3PickStatus);
    if (!payload?.task3_pick) throw new Error("Task3 自动抓取状态接口返回异常");
    renderTask3Pick(payload.task3_pick);
    return payload;
  } catch (error) {
    renderTask3Pick({
      enabled: false,
      ready: false,
      error: readableError(error),
    });
    throw error;
  }
}

function renderTask1Pick(status) {
  appState.task1Pick = status || {};
  updateTask1PickControls();
}

function renderTask2Pick(status) {
  appState.task2Pick = status || {};
  updateTask1PickControls();
}

function renderTask3Pick(status) {
  appState.task3Pick = status || {};
  updateTask1PickControls();
}

function updateTask1PickControls() {
  if (!el["run-task1-pick"]) return;
  const status = {
    task1: appState.task1Pick || {},
    task2: appState.task2Pick || {},
    task3: appState.task3Pick || {},
  }[appState.activeTask] || {};
  const jog = appState.cartesianJog || {};
  const teleopBlocked = cartesianJogTeleopBlocked();
  const ready = status.ready === true;
  const jogReady = cartesianJogEnabled(jog);
  const suctionReady = appState.suction?.available === true;
  const taskAvailable = ["task1", "task2", "task3"].includes(appState.activeTask);
  const blocked =
    !taskAvailable ||
    !appState.serviceOnline ||
    !ready ||
    !suctionReady ||
    appState.suction?.engaged === true ||
    appState.task1PickBusy ||
    appState.cartesianJogBusy ||
    cartesianJogServerBusy(jog) ||
    appState.detectionBusy ||
    appState.suctionBusy ||
    recordingIsActive() ||
    teleopBlocked;
  el["run-task1-pick"].disabled = blocked;
  el["run-task1-pick"].classList.toggle(
    "is-loading",
    appState.task1PickBusy
  );

  let detail = appState.activeTask === "task2"
    ? "仅 Task2 先到 left_watcher 识别，再按单层高度吸取"
    : appState.activeTask === "task3"
      ? "先到 left_box_watcher 识别顶层扁盒，再直达目标上方 25 mm 后垂直接触吸取"
      : "先到 left_watcher 识别，再以俯视逆时针 90° 法兰姿态直接进入目标上方吸取";
  if (!taskAvailable) {
    detail = TASK_PROFILES[appState.activeTask]?.unavailableMessage ||
      "当前任务的吸取 profile 尚未接入";
  } else if (appState.task1PickBusy) {
    detail = "正在执行实机抓取；请观察机械臂并保持急停可用";
  } else if (status.error) {
    detail = `抓取未就绪：${status.error}`;
  } else if (!ready) {
    detail = "抓取标定或执行端未就绪";
  } else if (!jogReady) {
    detail = "Skill 将自动读取并锁定当前竖直吸盘姿态";
  } else if (!suctionReady) {
    detail = "双吸盘串口未就绪";
  } else if (appState.suction?.engaged === true) {
    detail = "先释放吸盘，再执行新的药盒抓取";
  } else if (recordingIsActive()) {
    detail = "录制期间不能自动抓取";
  } else if (baseTrajectoryIsActive()) {
    detail = "底盘轨迹录制或回放期间不能自动抓取";
  } else if (teleopBlocked) {
    detail = "先正常停止遥操";
  }
  el["task1-pick-detail"].textContent = detail;
}

async function runActiveTaskPick() {
  if (
    !["task1", "task2", "task3"].includes(appState.activeTask) ||
    el["run-task1-pick"].disabled ||
    appState.task1PickBusy
  ) return;
  appState.task1PickBusy = true;
  updateTask1PickControls();
  setDetectionState("抓取中", "busy");
  el["detection-message"].textContent = appState.activeTask === "task2"
    ? "正在前往 left_watcher 识别并缓存 Task2 目标，再完成吸取…"
    : appState.activeTask === "task3"
      ? "正在前往 left_box_watcher 识别并缓存 Task3 目标，再完成吸取…"
      : "正在前往 left_watcher 识别 Task1 目标，再以俯视逆时针 90° 法兰姿态直接吸取…";
  try {
    const taskId = appState.activeTask;
    const endpoint = {
      task1: API.task1WatchDetectPick,
      task2: API.task2WatchDetectPick,
      task3: API.task3WatchDetectPick,
    }[taskId];
    const payload = await requestJSON(
      endpoint,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
      90000
    );
    if (!payload?.ok || !payload?.detection) {
      throw new Error(payload?.error || "PICK_CARTON Skill 返回异常");
    }
    appState.detection = payload.detection;
    renderDetection(payload.detection);
    showDetectionFrame(payload.detection);
    if (payload.cartesian_jog) renderCartesianJog(payload.cartesian_jog);
    if (payload.suction) renderSuction(payload.suction);
    if (payload.task1_pick) renderTask1Pick(payload.task1_pick);
    if (payload.task2_pick) renderTask2Pick(payload.task2_pick);
    if (payload.task3_pick) renderTask3Pick(payload.task3_pick);
    const layer = Number(payload.detection?.layer_estimate?.layer);
    showToast(
      Number.isInteger(layer)
        ? `已按 ${layer} 层固定高度吸紧，并完成 100 mm 试抬。`
        : "药盒已吸紧并完成 100 mm 试抬。",
      "info"
    );
  } catch (error) {
    renderDetectionError(readableError(error));
    showToast(`自动抓取未完成：${readableError(error)}`, "error");
    await Promise.allSettled([
      refreshCartesianJogStatus(),
      refreshSuctionStatus(),
      refreshTask1PickStatus(),
      refreshTask2PickStatus(),
      refreshTask3PickStatus(),
    ]);
  } finally {
    appState.task1PickBusy = false;
    updateTask1PickControls();
  }
}

async function setSuction(engaged) {
  if (
    typeof engaged !== "boolean" ||
    appState.suctionBusy ||
    appState.suction?.available !== true
  ) {
    return;
  }
  appState.suctionBusy = true;
  renderSuction(appState.suction);
  try {
    const payload = await requestJSON(API.suction, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ engaged }),
    });
    if (!payload?.suction) throw new Error("吸盘控制接口返回异常");
    renderSuction(payload.suction);
    showToast(
      engaged
        ? "双吸盘已发送吸紧命令；现在可用 Z+ 试抬20 mm。"
        : "双吸盘已发送释放命令。"
    );
  } catch (error) {
    showToast(`吸盘控制失败：${readableError(error)}`, "error");
    await refreshSuctionStatus().catch(() => {});
  } finally {
    appState.suctionBusy = false;
    renderSuction(appState.suction);
  }
}

function renderCartesianJog(jog) {
  appState.cartesianJog = jog || {};
  const mode = cartesianJogMode(appState.cartesianJog);
  const enabled = cartesianJogEnabled(appState.cartesianJog);
  const busy = cartesianJogServerBusy(appState.cartesianJog);
  const captured = cartesianJogOrientationCaptured(appState.cartesianJog);
  const teleopBlocked = cartesianJogTeleopBlocked();
  const realMotion =
    appState.cartesianJog.available === true &&
    appState.cartesianJog.dry_run === false;
  const executionLabel = !enabled
    ? ""
    : appState.cartesianJog.dry_run === true
      ? " · 干运行"
      : realMotion
        ? " · 实机运动"
        : "";
  const stateClass = busy
    ? "busy"
    : enabled
      ? realMotion
        ? "live"
        : "enabled"
      : jogErrorText(appState.cartesianJog)
        ? "error"
        : captured
          ? "ready"
          : "idle";

  el["jog-state"].textContent = `${mode}${executionLabel}`;
  el["jog-state"].className = `jog-state jog-state--${stateClass}`;
  el["jog-busy-state"].textContent =
    appState.cartesianJogBusy || busy ? "忙 · 禁止重复操作" : "空闲";
  el["jog-busy-state"].classList.toggle(
    "is-busy",
    appState.cartesianJogBusy || busy
  );

  const position = cartesianJogPosition(appState.cartesianJog);
  el["jog-position-x"].textContent = formatJogPosition(position?.x);
  el["jog-position-y"].textContent = formatJogPosition(position?.y);
  el["jog-position-z"].textContent = formatJogPosition(position?.z);

  const quaternion = cartesianJogQuaternion(appState.cartesianJog);
  el["jog-quaternion"].textContent = quaternion
    ? ["x", "y", "z", "w"]
      .map((axis) => `${axis} ${formatNumber(quaternion[axis], 5)}`)
      .join(" · ")
    : "尚未捕获方向";

  const error = jogErrorText(appState.cartesianJog);
  if (error) {
    showJogAlert(`上一步未完成：${error}`, false);
  } else {
    clearJogAlert();
  }

  if (baseTrajectoryIsActive()) {
    el["jog-status-message"].textContent =
      "底盘轨迹录制或回放正在进行；请先停止底盘轨迹操作。";
  } else if (teleopBlocked) {
    el["jog-status-message"].textContent =
      "主从臂遥操正在运行或切换中；必须先正常停止遥操。";
  } else if (busy || appState.cartesianJogBusy) {
    el["jog-status-message"].textContent =
      appState.cartesianJog.message || "左臂正在执行一个安全微调请求…";
  } else if (enabled && appState.cartesianJog.dry_run === true) {
    el["jog-status-message"].textContent =
      "干运行已启用：按钮只计算并校验目标，不会驱动左臂。";
  } else if (enabled) {
    el["jog-status-message"].textContent =
      "任务一实机运动已启用；按住方向按钮连续小步移动，松开后停止续发。";
  } else if (captured) {
    el["jog-status-message"].textContent =
      "姿态已记录；点击“开启 XYZ 微调”即可继续。";
  } else {
    const unavailableReason =
      appState.cartesianJog.unavailable_reason ||
      appState.cartesianJog.blocked_reason ||
      appState.cartesianJog.message;
    el["jog-status-message"].textContent =
      unavailableReason || "将吸盘调为竖直向下，然后点击“开启 XYZ 微调”。";
  }
  updateCartesianJogControls();
  updateTask1PickControls();
  renderFixedAxisCalibration();
}

function cartesianJogMode(jog) {
  const raw = String(jog.mode || jog.state || "").trim();
  if (raw) {
    const labels = {
      disabled: "未启用",
      idle: "未捕获",
      ready: "姿态已记录 · 运动未启用",
      captured: "姿态已记录 · 运动未启用",
      armed: "微调已启用",
      enabled: "微调已启用",
      moving: "微调移动中",
      capturing: "捕获方向中",
      enabling: "启用中",
      restoring: "返回安全竖直位",
      busy: "左臂动作中",
      holding: "保持当前位置",
      disabling: "禁用中",
      error: "故障",
      fault: "故障",
    };
    return labels[raw.toLowerCase()] || raw;
  }
  if (cartesianJogEnabled(jog)) return "微调已启用";
  return cartesianJogOrientationCaptured(jog)
    ? "姿态已记录 · 运动未启用"
    : "未捕获";
}

function cartesianJogEnabled(jog) {
  if (typeof jog.enabled === "boolean") return jog.enabled;
  if (typeof jog.armed === "boolean") return jog.armed;
  if (typeof jog.active === "boolean") return jog.active;
  return ["enabled", "armed", "moving"].includes(
    String(jog.mode || jog.state || "").toLowerCase()
  );
}

function cartesianJogServerBusy(jog) {
  if (typeof jog.busy === "boolean") return jog.busy;
  return [
    "moving",
    "capturing",
    "enabling",
    "restoring",
    "holding",
    "disabling",
    "stopping",
  ].includes(String(jog.state || jog.mode || "").toLowerCase());
}

function cartesianJogQuaternion(jog) {
  const candidates = [
    jog.locked_quaternion_xyzw,
    jog.locked_quaternion,
    jog.orientation_quaternion,
    jog.captured_orientation,
    jog.locked_orientation,
    jog.quaternion,
    jog.pose?.orientation,
    jog.last_result?.locked_quaternion_xyzw,
    jog.last_result?.quaternion_xyzw,
  ];
  for (const value of candidates) {
    if (Array.isArray(value) && value.length >= 4) {
      const numbers = value.slice(0, 4).map(Number);
      if (numbers.every(Number.isFinite)) {
        return { x: numbers[0], y: numbers[1], z: numbers[2], w: numbers[3] };
      }
    }
    if (value && typeof value === "object") {
      const quaternion = {
        x: Number(value.x ?? value.qx),
        y: Number(value.y ?? value.qy),
        z: Number(value.z ?? value.qz),
        w: Number(value.w ?? value.qw),
      };
      if (Object.values(quaternion).every(Number.isFinite)) return quaternion;
    }
  }
  return null;
}

function cartesianJogOrientationCaptured(jog) {
  const explicit = [
    jog.orientation_captured,
    jog.has_locked_orientation,
    jog.orientation_locked,
    jog.captured,
  ].find((value) => typeof value === "boolean");
  return typeof explicit === "boolean"
    ? explicit
    : cartesianJogQuaternion(jog) !== null;
}

function cartesianJogPosition(jog) {
  const candidates = [
    [jog.current_position_m, 1],
    [jog.actual_position_m, 1],
    [jog.target_position_m, 1],
    [jog.position_m, 1],
    [jog.xyz_m, 1],
    [jog.base_xyz_m, 1],
    [jog.current_position, 1],
    [jog.position, 1],
    [jog.xyz, 1],
    [jog.pose?.position, 1],
    [jog.position_mm, 0.001],
    [jog.xyz_mm, 0.001],
    [jog.last_result?.actual_position_m, 1],
    [jog.last_result?.target_position_m, 1],
    [jog.last_result?.current_position_m, 1],
  ];
  for (const [value, scale] of candidates) {
    let position = null;
    if (Array.isArray(value) && value.length >= 3) {
      position = { x: Number(value[0]), y: Number(value[1]), z: Number(value[2]) };
    } else if (value && typeof value === "object") {
      position = {
        x: Number(value.x),
        y: Number(value.y),
        z: Number(value.z),
      };
    }
    if (position && Object.values(position).every(Number.isFinite)) {
      return {
        x: position.x * scale,
        y: position.y * scale,
        z: position.z * scale,
      };
    }
  }
  return null;
}

function jogErrorText(jog) {
  const value = jog.error || jog.fault || jog.last_error;
  if (!value) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") {
    return value.message || value.detail || value.code || JSON.stringify(value);
  }
  return String(value);
}

function cartesianJogTeleopBlocked() {
  const jog = appState.cartesianJog || {};
  const teleopState = String(appState.teleop?.state || "").toLowerCase();
  return (
    jog.teleop_running === true ||
    Boolean(jog.teleop_error) ||
    appState.teleopBusy ||
    appState.teleop?.running === true ||
    appState.teleop?.busy === true ||
    baseTrajectoryIsActive() ||
    ["starting", "stopping", "waiting-endpoints", "restarting"].includes(
      teleopState
    )
  );
}

function bindCartesianJogHoldButton(button) {
  const axis = String(button.dataset.jogAxis || "");
  const direction = Number(button.dataset.jogDirection);

  button.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || button.disabled) return;
    event.preventDefault();
    try {
      button.setPointerCapture(event.pointerId);
    } catch (_error) {
      // Pointer capture is an enhancement; window/page safety stops still apply.
    }
    startCartesianJogHold(button, axis, direction, event.pointerId);
  });

  const stopPointerHold = (event) => {
    event.preventDefault();
    stopCartesianJogHoldForPointer(event, "方向按钮已松开");
  };
  button.addEventListener("pointerup", stopPointerHold);
  button.addEventListener("pointercancel", stopPointerHold);
  button.addEventListener("lostpointercapture", stopPointerHold);

  button.addEventListener("keydown", (event) => {
    if (
      !["Enter", " "].includes(event.key) ||
      event.repeat ||
      button.disabled
    ) {
      return;
    }
    event.preventDefault();
    startCartesianJogHold(button, axis, direction, null);
  });
  button.addEventListener("keyup", (event) => {
    if (!["Enter", " "].includes(event.key)) return;
    event.preventDefault();
    const hold = appState.cartesianJogHold;
    if (hold?.button === button && hold.pointerId === null) {
      stopCartesianJogHold("方向按键已松开");
    }
  });

  // pointerdown/keydown owns movement; suppress the synthetic click so a
  // short press remains exactly one validated step.
  button.addEventListener("click", (event) => event.preventDefault());
  button.addEventListener("contextmenu", (event) => event.preventDefault());
}

function stopCartesianJogHoldForPointer(event, reason) {
  const hold = appState.cartesianJogHold;
  if (!hold || hold.pointerId !== event.pointerId) return;
  stopCartesianJogHold(reason);
}

function updateCartesianJogControls() {
  if (!el["jog-capture-orientation"]) return;
  const jog = appState.cartesianJog || {};
  const teleopBlocked = cartesianJogTeleopBlocked();
  const serverBusy = cartesianJogServerBusy(jog);
  const requestBusy = appState.cartesianJogBusy;
  const enabled = cartesianJogEnabled(jog);
  const captured = cartesianJogOrientationCaptured(jog);
  const explicitlyUnavailable = jog.available === false;
  const homeAvailable = jog.home_joint_pose?.available === true;
  const rightHomeReady = appState.rightArmHome?.ready === true;
  const safeVerticalAvailable = jog.safe_vertical_pose?.available === true;
  const safetyBlocked =
    !appState.serviceOnline ||
    !appState.cartesianJogSafetyContractValid ||
    teleopBlocked ||
    explicitlyUnavailable;
  const generallyBlocked = safetyBlocked || serverBusy || requestBusy;

  if (
    appState.cartesianJogHold?.active === true &&
    (safetyBlocked || !enabled || !captured)
  ) {
    stopCartesianJogHold("安全状态变化，连续微调已停止");
  }

  el["jog-capture-orientation"].disabled = true;
  el["jog-area-clear"].disabled = true;
  el["jog-estop-ready"].disabled = true;
  el["jog-suction-released"].disabled = true;
  el["jog-enable"].disabled =
    generallyBlocked || enabled;
  el["left-arm-reset-home"].disabled =
    generallyBlocked ||
    !homeAvailable ||
    recordingIsActive() ||
    baseTrajectoryIsActive();
  el["right-arm-reset-home"].disabled =
    generallyBlocked ||
    !rightHomeReady ||
    recordingIsActive() ||
    baseTrajectoryIsActive();
  if (el["jog-restore-safe"]) {
    el["jog-restore-safe"].disabled =
      generallyBlocked ||
      !safeVerticalAvailable ||
      enabled ||
      recordingIsActive() ||
      baseTrajectoryIsActive();
  }
  el.jogStepInputs.forEach((input) => {
    input.disabled = generallyBlocked || !enabled;
  });
  el.jogMoveButtons.forEach((button) => {
    const isActiveHoldButton =
      appState.cartesianJogHold?.active === true &&
      appState.cartesianJogHold.button === button;
    button.disabled =
      safetyBlocked ||
      !enabled ||
      !captured ||
      ((serverBusy || requestBusy) && !isActiveHoldButton);
    button.classList.toggle("is-holding", isActiveHoldButton);
    button.setAttribute("aria-pressed", isActiveHoldButton ? "true" : "false");
    button.title = "按住连续移动；松开后不再发送下一小步";
  });
  el["jog-hold-disable"].disabled =
    !appState.serviceOnline ||
    serverBusy ||
    requestBusy ||
    !enabled;
  updateTask1PickControls();
}

function setCartesianJogBusy(busy, message = "") {
  appState.cartesianJogBusy = busy;
  if (busy && message) {
    el["jog-status-message"].textContent = message;
  }
  el["jog-busy-state"].textContent = busy ? "忙 · 禁止重复操作" : "空闲";
  el["jog-busy-state"].classList.toggle("is-busy", busy);
  updateCartesianJogControls();
  updateBaseTrajectoryButtons();
  updateTask1PickControls();
}

async function captureCartesianJogOrientation() {
  if (el["jog-capture-orientation"].disabled) return;
  setCartesianJogBusy(true, "正在只读记录左臂当前位置与末端方向…");
  clearJogAlert(true);
  try {
    const payload = await requestJSON(API.cartesianJogCapture, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirm: "CAPTURE_LEFT_SUCTION_DOWN",
        vertical_down_confirmed: true,
      }),
    });
    await updateCartesianJogAfterMutation(payload);
    showToast("已只读记录左吸盘姿态；尚未启用任何运动。");
  } catch (error) {
    const message = `捕获左臂方向失败：${readableError(error)}`;
    showJogAlert(message);
    showToast(message, "error");
  } finally {
    setCartesianJogBusy(false);
  }
}

async function enableCartesianJog() {
  if (el["jog-enable"].disabled) return;
  setCartesianJogBusy(true, "正在读取当前姿态并开启 XYZ 微调…");
  clearJogAlert(true);
  try {
    const payload = await requestJSON(API.cartesianJogQuickEnable, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirm: "ENABLE_LEFT_CARTESIAN_JOG",
      }),
    });
    await updateCartesianJogAfterMutation(payload);
    showToast(
      appState.cartesianJog?.dry_run === true
        ? "干运行已启用；按住方向按钮会连续校验目标，不会驱动左臂。"
        : "任务一左臂实机微调已启用；按住方向按钮连续移动，松开停止续发。"
    );
  } catch (error) {
    const message = `无法开启 XYZ 微调：${readableError(error)}`;
    showJogAlert(message);
    showToast(message, "error");
  } finally {
    setCartesianJogBusy(false);
  }
}

async function restoreCartesianJogSafePose() {
  if (!el["jog-restore-safe"] || el["jog-restore-safe"].disabled) return;
  setCartesianJogBusy(true, "正在将左臂返回安全竖直位…");
  clearJogAlert(true);
  try {
    const payload = await requestJSON(
      API.cartesianJogRestoreSafe,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          confirm: "RESTORE_LEFT_SAFE_VERTICAL",
          area_clear: true,
          estop_ready: true,
          suction_released: true,
        }),
      },
      CARTESIAN_MOTION_REQUEST_TIMEOUT_MS
    );
    await updateCartesianJogAfterMutation(payload);
    const result = payload?.result || {};
    showToast(
      result.dry_run === true || result.executed === false
        ? "安全竖直位路径校验通过；干运行未移动左臂。"
        : "左臂已回到安全竖直位；方向已记录，微调仍未启用。"
    );
  } catch (error) {
    const message = `返回安全竖直位失败：${readableError(error)}`;
    showJogAlert(message);
    showToast(message, "error");
  } finally {
    setCartesianJogBusy(false);
  }
}

async function resetLeftArmHome() {
  if (
    el["left-arm-reset-home"].disabled ||
    appState.cartesianJogBusy
  ) {
    return;
  }
  setCartesianJogBusy(
    true,
    "左臂正在低速返回与右臂相同的初始关节位…"
  );
  clearJogAlert(true);
  try {
    const payload = await requestJSON(
      API.leftArmResetHomeSkill,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
      60000
    );
    if (!payload?.ok || payload?.skill?.id !== "left_arm.reset_home") {
      throw new Error(payload?.error || "左臂复位 Skill 返回异常");
    }
    await updateCartesianJogAfterMutation(payload);
    showToast(
      payload.result?.dry_run === true || payload.result?.executed === false
        ? "左臂初始位已校验；干运行未驱动机械臂。"
        : "左臂已复位到与右臂相同的初始关节位。"
    );
  } catch (error) {
    const message = `左臂复位未完成：${readableError(error)}`;
    showJogAlert(message);
    showToast(message, "error");
  } finally {
    setCartesianJogBusy(false);
  }
}

async function resetRightArmHome() {
  if (
    el["right-arm-reset-home"].disabled ||
    appState.cartesianJogBusy
  ) {
    return;
  }
  setCartesianJogBusy(true, "右臂正在低速返回初始关节位…");
  clearJogAlert(true);
  try {
    const payload = await requestJSON(
      API.rightArmResetHomeSkill,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      },
      60000
    );
    if (!payload?.ok || payload?.skill?.id !== "right_arm.reset_home") {
      throw new Error(payload?.error || "右臂复位 Skill 返回异常");
    }
    await Promise.allSettled([
      refreshCartesianJogStatus(),
      refreshRightArmHomeStatus(),
    ]);
    showToast(
      payload.result?.dry_run === true || payload.result?.executed === false
        ? "右臂初始位已校验；干运行未驱动机械臂。"
        : "右臂已复位到初始关节位。"
    );
  } catch (error) {
    const message = `右臂复位未完成：${readableError(error)}`;
    showJogAlert(message);
    showToast(message, "error");
  } finally {
    setCartesianJogBusy(false);
  }
}

function selectedCartesianJogStep() {
  const selectedStep = Number(
    el.jogStepInputs.find((input) => input.checked)?.value
  );
  return [1, 2, 5, 10].includes(selectedStep) ? selectedStep : null;
}

function effectiveCartesianJogStep(axis, direction, selectedStep) {
  if (axis !== "z" || direction >= 0) return selectedStep;
  const configuredLimit = Number(
    appState.cartesianJog?.max_downward_z_step_mm
  );
  const downwardLimit = [1, 2, 5, 10].includes(configuredLimit)
    ? configuredLimit
    : 5;
  return Math.min(selectedStep, downwardLimit);
}

function startCartesianJogHold(button, axis, direction, pointerId) {
  if (
    appState.cartesianJogHold !== null ||
    button.disabled ||
    !["x", "y", "z"].includes(axis) ||
    ![-1, 1].includes(direction) ||
    appState.cartesianJogBusy ||
    !cartesianJogEnabled(appState.cartesianJog || {})
  ) {
    return;
  }

  const selectedStepMm = selectedCartesianJogStep();
  if (selectedStepMm === null) {
    showJogAlert("微调步长无效；只能选择 1、2、5 或 10 mm。");
    return;
  }
  const stepMm = effectiveCartesianJogStep(axis, direction, selectedStepMm);

  const hold = {
    active: true,
    axis,
    button,
    completedSteps: 0,
    direction,
    pointerId,
    stepMm,
  };
  appState.cartesianJogHold = hold;
  clearJogAlert(true);
  updateCartesianJogControls();
  void runCartesianJogHold(hold);
}

function stopCartesianJogHold(reason = "") {
  const hold = appState.cartesianJogHold;
  if (!hold) return;

  hold.active = false;
  appState.cartesianJogHold = null;
  if (
    hold.pointerId !== null &&
    typeof hold.button.hasPointerCapture === "function" &&
    hold.button.hasPointerCapture(hold.pointerId)
  ) {
    try {
      hold.button.releasePointerCapture(hold.pointerId);
    } catch (_error) {
      // The browser may have released it already during pointerup/cancel.
    }
  }
  hold.button.classList.remove("is-holding");
  hold.button.setAttribute("aria-pressed", "false");
  if (reason && appState.cartesianJogBusy) {
    el["jog-status-message"].textContent =
      `${reason}；当前已下发的小步完成后停止。`;
  }
  updateCartesianJogControls();
  updateBaseTrajectoryButtons();
}

async function runCartesianJogHold(hold) {
  while (appState.cartesianJogHold === hold && hold.active) {
    const moved = await moveCartesianJog(hold.axis, hold.direction, {
      hold,
      stepMm: hold.stepMm,
    });
    if (!moved) {
      if (appState.cartesianJogHold === hold) {
        stopCartesianJogHold("本次微调未完成");
      }
      return;
    }
    hold.completedSteps += 1;
    if (appState.cartesianJogHold !== hold || !hold.active) return;
    await new Promise((resolve) => {
      window.setTimeout(resolve, CARTESIAN_JOG_HOLD_REPEAT_DELAY_MS);
    });
  }
}

async function moveCartesianJog(axis, direction, { hold = null, stepMm = null } = {}) {
  if (
    !["x", "y", "z"].includes(axis) ||
    ![-1, 1].includes(direction) ||
    appState.cartesianJogBusy ||
    !cartesianJogEnabled(appState.cartesianJog || {})
  ) {
    return false;
  }
  const selectedStep = stepMm ?? selectedCartesianJogStep();
  if (![1, 2, 5, 10].includes(selectedStep)) {
    showJogAlert("微调步长无效；只能选择 1、2、5 或 10 mm。");
    return false;
  }

  const label = `${axis.toUpperCase()}${direction > 0 ? "+" : "−"} ${selectedStep} mm`;
  const stepNumber = hold ? hold.completedSteps + 1 : 1;
  setCartesianJogBusy(
    true,
    hold
      ? `按住连续移动：正在执行左臂 ${label} · 第 ${stepNumber} 步…`
      : `正在执行左臂 ${label} 单步移动…`
  );
  clearJogAlert(true);
  try {
    const payload = await requestJSON(
      API.cartesianJogMove,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ axis, direction, step_mm: selectedStep }),
      },
      CARTESIAN_MOTION_REQUEST_TIMEOUT_MS
    );
    await updateCartesianJogAfterMutation(payload);
    const result = payload?.result || {};
    if (!hold) {
      showToast(
        result.dry_run === true || result.executed === false
          ? `干运行通过：左臂 ${label} 未实际执行。`
          : `左臂 ${label} 已完成。`
      );
    }
    return true;
  } catch (error) {
    const message = `左臂 ${label} 失败：${readableError(error)}`;
    showJogAlert(message);
    showToast(message, "error");
    return false;
  } finally {
    setCartesianJogBusy(false);
  }
}

async function disableCartesianJog() {
  if (el["jog-hold-disable"].disabled) return;
  setCartesianJogBusy(true, "正在禁用后续左臂微调请求…");
  clearJogAlert(true);
  try {
    const payload = await requestJSON(API.cartesianJogDisable, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    el["jog-area-clear"].checked = false;
    el["jog-estop-ready"].checked = false;
    el["jog-suction-released"].checked = false;
    await updateCartesianJogAfterMutation(payload);
    showToast("后续左臂微调请求已禁用。");
  } catch (error) {
    const message =
      `禁用微调失败：${readableError(error)}。` +
      "此按钮不能中断正在执行的动作；异常时请使用实体急停。";
    showJogAlert(message);
    showToast(message, "error");
    await refreshCartesianJogStatus().catch(() => {});
  } finally {
    setCartesianJogBusy(false);
  }
}

async function updateCartesianJogAfterMutation(payload) {
  const jog = extractCartesianJog(payload);
  const hasStatusFields =
    jog &&
    [
      "enabled",
      "armed",
      "active",
      "state",
      "mode",
      "orientation_captured",
      "locked_quaternion_xyzw",
      "locked_quaternion",
      "position",
      "position_m",
      "xyz",
      "xyz_m",
    ].some((key) => Object.hasOwn(jog, key));
  if (hasStatusFields) {
    renderCartesianJog(jog);
    return;
  }
  await refreshCartesianJogStatus();
}

function formatJogPosition(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(4)} m` : "—";
}

function showJogAlert(message, sticky = true) {
  appState.cartesianJogAlertSticky = sticky;
  el["jog-alert"].textContent = message;
  el["jog-alert"].classList.remove("is-hidden");
}

function clearJogAlert(force = false) {
  if (appState.cartesianJogAlertSticky && !force) return;
  appState.cartesianJogAlertSticky = false;
  el["jog-alert"].textContent = "";
  el["jog-alert"].classList.add("is-hidden");
}

async function refreshRecordingStatus() {
  try {
    const payload = await requestJSON(API.recordingStatus);
    if (!payload?.recording) throw new Error("录制状态接口返回异常");
    const previous = appState.recording;
    renderRecording(payload.recording);
    const finishedNow =
      previous &&
      ["starting", "recording", "stopping"].includes(previous.state) &&
      ["saved", "error"].includes(payload.recording.state);
    if (finishedNow) {
      await loadRecordings();
      showToast(
        payload.recording.state === "saved"
          ? "轨迹已安全保存。"
          : `录制失败：${payload.recording.error || "未知错误"}`,
        payload.recording.state === "saved" ? "info" : "error"
      );
      if (!recordingIsActive()) refreshFrame(false);
    }
    return payload;
  } catch (error) {
    if (recordingIsActive()) {
      el["recording-message"].textContent =
        `暂时无法读取录制状态：${readableError(error)}`;
    }
    throw error;
  }
}

async function loadRecordings() {
  try {
    const payload = await requestJSON(API.recordings);
    appState.recordings = Array.isArray(payload?.recordings)
      ? payload.recordings
      : [];
    renderRecordingList(appState.recordings);
    return payload;
  } catch (error) {
    renderRecordingList([]);
    throw error;
  }
}

function recordingIsActive() {
  return ["starting", "recording", "stopping"].includes(
    String(appState.recording?.state || "")
  );
}

function setRecordingBusy(busy) {
  appState.recordingBusy = busy;
  updateRecordingButtons();
  updateBaseTrajectoryButtons();
}

function updateRecordingButtons() {
  const active = recordingIsActive();
  const stopping = appState.recording?.state === "stopping";
  const teleopStarting =
    appState.teleopBusy || appState.teleop?.busy === true;
  const baseTrajectoryBlocked = baseTrajectoryIsActive();
  el["start-recording"].disabled =
    appState.recordingBusy ||
    active ||
    baseTrajectoryBlocked ||
    actRolloutIsActive() ||
    teleopStarting ||
    !appState.serviceOnline;
  el["stop-recording"].disabled =
    appState.recordingBusy || !active || stopping || !appState.serviceOnline;
  el["recording-label"].disabled = active || appState.recordingBusy || baseTrajectoryBlocked;
  el["recording-purpose"].disabled = active || appState.recordingBusy || baseTrajectoryBlocked;
  el["refresh-frame"].disabled = active;
  updateDetectionButton();
  updateTeleopButton();
  updateActRolloutButtons();
  updateTask1PickControls();
}

function renderRecording(recording) {
  const wasRecordingActive = recordingIsActive();
  appState.recording = recording || { state: "idle", active: false };
  const state = String(appState.recording.state || "idle");
  const labels = {
    idle: "空闲",
    starting: "正在连接",
    recording: "录制中",
    stopping: "正在保存",
    saved: "已保存",
    error: "失败",
  };
  el["recording-state"].textContent = labels[state] || state;
  el["recording-state"].className =
    `recording-state recording-state--${escapeHTML(state)}`;
  el["recording-duration"].textContent = formatDuration(
    appState.recording.duration_s || 0
  );
  el["recording-frames"].textContent = String(
    appState.recording.frame_count || 0
  );
  const isAct = String(appState.recording.purpose || "").startsWith("act_");
  const counts = isAct
    ? appState.recording.action_sample_counts || {}
    : appState.recording.arm_sample_counts || {};
  el["recording-left-samples"].textContent = String(counts.left || 0);
  el["recording-right-samples"].textContent = String(counts.right || 0);
  updateDisplayedRecordingPath();

  const messages = {
    idle: "请选择采集类型。标定采集会同步保存左右臂反馈。",
    starting: "正在只读连接相机与机械臂反馈；不会改变控制模式。",
    recording: (
      `正在录制 ${purposeLabel(appState.recording.purpose)}；` +
      "主视频保持原始帧率，页面显示约 5 FPS 的同源预览；药盒识别已暂停。"
    ),
    stopping: "正在关闭视频、补写时间戳并原子保存 episode…",
    saved: isAct
      ? "ACT episode 已封装并生成 READY，可执行数据校验。"
      : "本次数据已完成封装，可直接用于后续标定检查。",
    error: `录制失败：${appState.recording.error || "未知错误"}`,
  };
  el["recording-message"].textContent = messages[state] || state;
  updateRecordingButtons();
  if (
    wasRecordingActive !== recordingIsActive() &&
    appState.frameTimer !== null
  ) {
    scheduleFramePoll(0);
  }
}

function renderRecordingList(recordings) {
  if (!recordings.length) {
    el["recording-list"].innerHTML =
      '<li class="recording-empty">尚无本项目录制数据</li>';
    updateDisplayedRecordingPath();
    return;
  }
  el["recording-list"].innerHTML = recordings
    .map((item, index) => {
      const failed = item.status !== "completed";
      const duration = formatDuration(item.duration_s || 0);
      const frames = Number(item.frame_count || 0);
      const samples = String(item.purpose || "").startsWith("act_")
        ? item.action_sample_counts || {}
        : item.arm_sample_counts || {};
      const path = String(item.path || "");
      const replayable =
        !failed &&
        item.replay_ready === true &&
        String(item.id || "").startsWith("act_");
      const replaying =
        appState.replay?.active === true &&
        appState.replay?.recording_id === item.id;
      const deleting = appState.recordingDeleteId === item.id;
      const confirmingDelete = appState.recordingDeleteConfirmId === item.id;
      const deleteDisabled =
        recordingIsActive() ||
        appState.replay?.active === true ||
        appState.recordingDeleteId !== null;
      return `
        <li class="recording-item">
          <div>
            <strong>${escapeHTML(String(item.label || item.id || "recording"))}</strong>
            <div class="recording-item-path">
              <code title="${escapeHTML(path)}">${escapeHTML(path)}</code>
              <button
                class="recording-copy-button"
                type="button"
                data-copy-recording-index="${index}"
                ${path ? "" : "disabled"}
              >复制</button>
              ${replayable ? `
              <button
                class="recording-copy-button"
                type="button"
                data-replay-recording-index="${index}"
                ${appState.replayBusy || (appState.replay?.active && !replaying) ? "disabled" : ""}
              >${replaying ? "停止重放" : "重放轨迹"}</button>` : ""}
              <button
                class="recording-copy-button recording-delete-button${confirmingDelete ? " is-confirming" : ""}"
                type="button"
                data-delete-recording-index="${index}"
                ${deleteDisabled ? "disabled" : ""}
              >${deleting ? "删除中…" : confirmingDelete ? "确认删除" : "删除"}</button>
            </div>
          </div>
          <div>
            <strong>${escapeHTML(purposeLabel(item.purpose))}</strong>
            <small>${formatTime(item.created_at)} · ${duration}</small>
          </div>
          <div>
            <strong>${frames} 帧</strong>
            <small>L ${Number(samples.left || 0)} · R ${Number(samples.right || 0)}</small>
          </div>
          <span class="recording-item-status${failed ? " is-failed" : ""}">
            ${failed ? "FAILED" : "SAVED"}
          </span>
        </li>
      `;
    })
    .join("");
  updateDisplayedRecordingPath();
}

async function deleteRecording(item) {
  if (
    recordingIsActive() ||
    appState.replay?.active === true ||
    appState.recordingDeleteId !== null
  ) {
    showToast("请先停止录制或轨迹重放，再删除数据。", "error");
    return;
  }
  const recordingId = String(item.id || "").trim();
  if (!recordingId) return;
  const label = String(item.label || recordingId);
  if (appState.recordingDeleteConfirmId !== recordingId) {
    if (appState.recordingDeleteConfirmTimer !== null) {
      window.clearTimeout(appState.recordingDeleteConfirmTimer);
    }
    appState.recordingDeleteConfirmId = recordingId;
    renderRecordingList(appState.recordings);
    showToast(`请在 5 秒内再次点击“确认删除”以删除“${label}”。`);
    appState.recordingDeleteConfirmTimer = window.setTimeout(() => {
      if (appState.recordingDeleteConfirmId === recordingId) {
        appState.recordingDeleteConfirmId = null;
        appState.recordingDeleteConfirmTimer = null;
        renderRecordingList(appState.recordings);
      }
    }, 5000);
    return;
  }
  if (appState.recordingDeleteConfirmTimer !== null) {
    window.clearTimeout(appState.recordingDeleteConfirmTimer);
  }
  appState.recordingDeleteConfirmId = null;
  appState.recordingDeleteConfirmTimer = null;

  appState.recordingDeleteId = recordingId;
  renderRecordingList(appState.recordings);
  try {
    const payload = await requestJSON(API.recordingDelete, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recording_id: recordingId }),
    });
    if (!payload?.deletion?.deleted) {
      throw new Error("删除接口返回异常");
    }
    appState.recordings = Array.isArray(payload.recordings)
      ? payload.recordings
      : appState.recordings.filter((candidate) => candidate.id !== recordingId);
    await refreshRecordingStatus();
    showToast(`已删除“${label}”；如需恢复可从回收区找回。`);
  } catch (error) {
    showToast(`删除失败：${readableError(error)}`, "error");
  } finally {
    appState.recordingDeleteId = null;
    renderRecordingList(appState.recordings);
  }
}

function renderReplay(replay) {
  const previousState = String(appState.replay?.state || "");
  appState.replay = replay || { active: false, state: "idle" };
  if (appState.recordings.length) renderRecordingList(appState.recordings);
  const state = String(appState.replay.state || "idle");
  if (state !== previousState) {
    if (state === "completed") {
      showToast("轨迹原速重放完成；本次未控制吸盘。");
    } else if (state === "stopped") {
      showToast("轨迹重放已停止并保持当前位置。");
    } else if (state === "error") {
      showToast(`轨迹重放失败：${appState.replay.error || "未知错误"}`, "error");
    }
  }
}

async function startTrajectoryReplay(item) {
  if (appState.replayBusy || appState.replay?.active) return;
  const recordingId = String(item.id || "");
  appState.replayBusy = true;
  renderRecordingList(appState.recordings);
  try {
    const preflightPayload = await requestJSON(API.replayPreflight, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recording_id: recordingId }),
    });
    const preflight = preflightPayload?.preflight;
    if (!preflight) throw new Error("重放预检接口返回异常");
    const speedNotice =
      Number(preflight.speed_scale) > 1
        ? `按 ${Number(preflight.speed_scale).toFixed(1)}× 速度重放，保留${preflight.retained_frame_parity === "odd" ? "奇数" : "配置"}帧，`
        : "按录制原速重放，";
    const suctionNotice = preflight.suction_replayed
      ? `将在原始第 ${preflight.suction_release_frame_1_based} 帧自动停止吸力并释放物体。`
      : "不控制吸盘。";
    const confirmation = window.prompt(
      "系统将先低速回到第一帧，再" + speedNotice +
      "执行从臂实际关节与右夹爪轨迹，" +
      `预计 ${formatDuration(preflight.default_replay_duration_s)}。${suctionNotice}\n` +
      "请确认运动路径及工作区无人、夹爪内没有不应掉落的物体，然后完整输入录制 ID：\n" +
      recordingId,
      ""
    );
    if (confirmation === null) return;
    const payload = await requestJSON(API.replayStart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recording_id: recordingId,
        confirmation,
      }),
    });
    if (!payload?.replay) throw new Error("重放启动接口返回异常");
    renderReplay(payload.replay);
    showToast(
      preflight.suction_replayed
        ? `正在低速回到第一帧，随后以 ${Number(preflight.speed_scale).toFixed(1)}× 重放；原始第 ${preflight.suction_release_frame_1_based} 帧将自动释放吸盘。`
        : "正在低速回到第一帧，随后按录制原速重放从臂实际轨迹；可再次点击停止。"
    );
  } catch (error) {
    showToast(`无法开始重放：${readableError(error)}`, "error");
  } finally {
    appState.replayBusy = false;
    renderRecordingList(appState.recordings);
  }
}

async function stopTrajectoryReplay() {
  if (appState.replayBusy || appState.replay?.active !== true) return;
  appState.replayBusy = true;
  renderRecordingList(appState.recordings);
  try {
    const payload = await requestJSON(API.replayStop, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!payload?.replay) throw new Error("重放停止接口返回异常");
    renderReplay(payload.replay);
    showToast("正在停止轨迹重放并保持当前位置。");
  } catch (error) {
    showToast(`无法停止重放：${readableError(error)}`, "error");
  } finally {
    appState.replayBusy = false;
    renderRecordingList(appState.recordings);
  }
}

function baseTrajectoryIsActive() {
  const mode = String(appState.baseTrajectory?.mode || "");
  const replayState = String(appState.baseTrajectory?.replay?.state || "");
  return appState.baseTrajectory?.active === true ||
    ["recording", "replaying", "stopping"].includes(mode) ||
    ["replaying", "stopping"].includes(replayState);
}

function baseTrajectoryState(snapshot) {
  const state = String(snapshot?.mode || "");
  if (["disabled", "error", "recording", "replaying", "stopping"].includes(state)) {
    return state;
  }
  const replayState = String(snapshot?.replay?.state || "");
  if (state === "idle" && ["completed", "stopped", "error", "replaying", "stopping"].includes(replayState)) {
    return replayState;
  }
  if (["completed", "stopped", "error", "replaying", "stopping"].includes(replayState)) {
    return replayState;
  }
  if (snapshot?.last_saved || snapshot?.path) {
    return "saved";
  }
  return "idle";
}

function normalizeBaseTrajectorySnapshot(snapshot) {
  const normalized =
    snapshot && typeof snapshot === "object" ? { ...snapshot } : {};
  const replayState = String(normalized.state || "");
  if (
    !normalized.mode &&
    normalized.recording_id &&
    ["completed", "stopped", "error", "replaying", "stopping"].includes(replayState)
  ) {
    normalized.replay = normalized.replay || { ...normalized };
    normalized.mode = ["replaying", "stopping"].includes(replayState)
      ? replayState
      : replayState === "error"
        ? "error"
        : "idle";
    normalized.active = ["replaying", "stopping"].includes(replayState);
    normalized.message = normalized.message || `Base trajectory replay ${replayState}`;
  }
  if (
    !normalized.mode &&
    normalized.path &&
    typeof normalized.point_count !== "undefined"
  ) {
    normalized.mode = "idle";
    normalized.active = false;
    normalized.recording = normalized.recording || { ...normalized };
    normalized.last_saved = normalized.last_saved || { ...normalized };
    normalized.message = normalized.message || "Base trajectory recording saved";
  }
  return normalized;
}

function baseTrajectoryStateLabel(state) {
  return {
    disabled: "未启用",
    idle: "空闲",
    recording: "录制中",
    stopping: "正在保存",
    saved: "已保存",
    replaying: "回放中",
    completed: "已完成",
    stopped: "已停止",
    error: "失败",
  }[state] || state;
}

function baseTrajectoryStateClass(state) {
  if (state === "recording" || state === "replaying") return "recording-state--recording";
  if (state === "stopping") return "recording-state--stopping";
  if (state === "saved" || state === "completed" || state === "stopped") {
    return "recording-state--saved";
  }
  if (state === "error") return "recording-state--error";
  return "";
}

async function refreshBaseTrajectoryStatus() {
  try {
    const payload = await requestJSON(API.baseTrajectoryStatus);
    if (!payload?.base_trajectory) throw new Error("底盘轨迹状态接口返回异常");
    const previousState = baseTrajectoryState(appState.baseTrajectory);
    renderBaseTrajectory(payload.base_trajectory);
    const nextState = baseTrajectoryState(appState.baseTrajectory);
    if (previousState === "recording" && nextState === "saved") {
      await loadBaseTrajectories();
      showToast("底盘轨迹已保存。");
    } else if (previousState === "replaying" && nextState === "completed") {
      showToast("底盘轨迹回放已完成。");
    } else if (previousState === "replaying" && nextState === "stopped") {
      showToast("底盘轨迹回放已停止并保持当前位置。");
    } else if (previousState === "replaying" && nextState === "error") {
      showToast(
        `底盘轨迹回放失败：${appState.baseTrajectory?.error || "未知错误"}`,
        "error"
      );
    } else if (previousState === "recording" && nextState === "error") {
      showToast(
        `底盘轨迹录制失败：${appState.baseTrajectory?.error || "未知错误"}`,
        "error"
      );
    }
    return payload;
  } catch (error) {
    if (baseTrajectoryIsActive()) {
      el["base-trajectory-message"].textContent =
        `暂时无法读取底盘轨迹状态：${readableError(error)}`;
    }
    throw error;
  }
}

async function loadBaseTrajectories() {
  try {
    const payload = await requestJSON(API.baseTrajectories);
    appState.baseTrajectories = Array.isArray(payload?.base_trajectories)
      ? payload.base_trajectories
      : [];
    renderBaseTrajectoryList(appState.baseTrajectories);
    return payload;
  } catch (error) {
    appState.baseTrajectories = [];
    renderBaseTrajectoryList([]);
    throw error;
  }
}

function setBaseTrajectoryBusy(busy) {
  appState.baseTrajectoryBusy = busy;
  updateBaseTrajectoryButtons();
  updateRecordingButtons();
  updateTeleopButton();
  updateActRolloutButtons();
  updateCartesianJogControls();
}

function setBaseTrajectoryReplayBusy(busy) {
  appState.baseTrajectoryReplayBusy = busy;
  updateBaseTrajectoryButtons();
  updateRecordingButtons();
  updateTeleopButton();
  updateActRolloutButtons();
  updateCartesianJogControls();
}

function updateBaseTrajectoryButtons() {
  if (!el["base-trajectory-start"]) return;
  const state = baseTrajectoryState(appState.baseTrajectory);
  const active = baseTrajectoryIsActive();
  const conflicting =
    recordingIsActive() ||
    appState.replay?.active === true ||
    actRolloutIsActive() ||
    appState.teleopBusy ||
    appState.teleop?.running === true ||
    appState.teleop?.busy === true ||
    cartesianJogEnabled(appState.cartesianJog || {}) ||
    cartesianJogServerBusy(appState.cartesianJog || {});
  el["base-trajectory-start"].disabled =
    !appState.serviceOnline ||
    appState.baseTrajectoryBusy ||
    appState.baseTrajectoryReplayBusy ||
    active ||
    conflicting;
  el["base-trajectory-stop"].disabled =
    !appState.serviceOnline ||
    appState.baseTrajectoryBusy ||
    !["recording", "stopping"].includes(state);
  el["base-trajectory-label"].disabled =
    active ||
    appState.baseTrajectoryBusy ||
    appState.baseTrajectoryReplayBusy;
  if (el["base-trajectory-list"]) {
    const replayingId = String(appState.baseTrajectory?.replay?.recording_id || "");
    const replaying = String(appState.baseTrajectory?.replay?.state || "") === "replaying";
    el["base-trajectory-list"]
      .querySelectorAll("[data-replay-base-trajectory-index]")
      .forEach((button) => {
        const index = Number(button.dataset.replayBaseTrajectoryIndex);
        const item = appState.baseTrajectories[index];
        const isCurrentReplay = replaying && String(item?.id || "") === replayingId;
        button.disabled =
          appState.baseTrajectoryBusy ||
          appState.baseTrajectoryReplayBusy ||
          (active && !isCurrentReplay);
      });
  }
}

function renderBaseTrajectory(rawSnapshot) {
  const previousState = baseTrajectoryState(appState.baseTrajectory);
  const snapshot = normalizeBaseTrajectorySnapshot(rawSnapshot);
  appState.baseTrajectory = snapshot;
  const state = baseTrajectoryState(snapshot);
  const labels = {
    disabled: "未启用",
    idle: "空闲",
    recording: "录制中",
    stopping: "正在保存",
    saved: "已保存",
    replaying: "回放中",
    completed: "已完成",
    stopped: "已停止",
    error: "失败",
  };
  el["base-trajectory-state"].textContent = labels[state] || state;
  el["base-trajectory-state"].className =
    `recording-state ${baseTrajectoryStateClass(state)}`.trim();

  const recording = snapshot.recording || snapshot.last_saved || {};
  el["base-trajectory-duration"].textContent = formatDuration(
    Number(recording.duration_s || snapshot.duration_s || 0)
  );
  el["base-trajectory-points"].textContent = String(
    Number(recording.point_count || snapshot.point_count || 0)
  );
  el["base-trajectory-start-pose"].textContent = poseText(
    recording.start_pose || snapshot.start_pose
  );
  el["base-trajectory-end-pose"].textContent = poseText(
    recording.end_pose || snapshot.end_pose
  );
  updateDisplayedBaseTrajectoryPath();
  updateBaseTrajectoryButtons();
  renderBaseTrajectoryList(appState.baseTrajectories);

  const messages = {
    disabled: "底盘轨迹功能未启用。",
    idle: "录的是底盘位姿轨迹，不是遥操 episode。先填名称，再开始录制。",
    recording: "正在记录底盘的位置、朝向和速度；录制结束后可直接重放。",
    stopping: "正在停止并原子保存底盘轨迹…",
    saved: "底盘轨迹已保存，可在下方列表里重放。",
    replaying: "正在按录制轨迹控制底盘，请保持路径清空。",
    completed: "底盘轨迹回放已完成。",
    stopped: "底盘轨迹回放已停止并保持当前位置。",
    error: `底盘轨迹失败：${snapshot.error || "未知错误"}`,
  };
  el["base-trajectory-message"].textContent = messages[state] || state;

  if (previousState !== state && state === "completed") {
    renderBaseTrajectoryList(appState.baseTrajectories);
  }
}

function renderBaseTrajectoryList(items) {
  if (!items.length) {
    el["base-trajectory-list"].innerHTML =
      '<li class="recording-empty">尚无底盘轨迹数据</li>';
    updateDisplayedBaseTrajectoryPath();
    return;
  }
  const replayingId = String(appState.baseTrajectory?.replay?.recording_id || "");
  const replaying = String(appState.baseTrajectory?.replay?.state || "") === "replaying";
  el["base-trajectory-list"].innerHTML = items
    .map((item, index) => {
      const path = String(item.path || "");
      const isReplaying = replaying && replayingId === String(item.id || "");
      const replayDisabled =
        appState.baseTrajectoryBusy ||
        appState.baseTrajectoryReplayBusy ||
        (baseTrajectoryIsActive() && !isReplaying);
      return `
        <li class="recording-item">
          <div>
            <strong>${escapeHTML(String(item.label || item.id || "base-trajectory"))}</strong>
            <div class="recording-item-path">
              <code title="${escapeHTML(path)}">${escapeHTML(path)}</code>
              <button
                class="recording-copy-button"
                type="button"
                data-copy-base-trajectory-index="${index}"
                ${path ? "" : "disabled"}
              >复制</button>
              <button
                class="recording-copy-button"
                type="button"
                data-replay-base-trajectory-index="${index}"
                ${replayDisabled ? "disabled" : ""}
              >${isReplaying ? "停止回放" : "回放轨迹"}</button>
            </div>
            <small>${formatTime(item.created_at)} · 起点 ${poseText(item.start_pose)} · 终点 ${poseText(item.end_pose)}</small>
          </div>
          <div>
            <strong>${formatDuration(item.duration_s || 0)}</strong>
            <small>${Number(item.point_count || 0)} 点</small>
          </div>
          <div>
            <strong>${isReplaying ? "运行中" : "可回放"}</strong>
            <small>${isReplaying ? "当前底盘轨迹正在执行" : "确认串：REPLAY_BASE_TRAJECTORY"}</small>
          </div>
          <span class="recording-item-status">
            ${isReplaying ? "REPLAYING" : "READY"}
          </span>
        </li>
      `;
    })
    .join("");
  updateDisplayedBaseTrajectoryPath();
}

async function startBaseTrajectoryRecording() {
  if (
    !appState.serviceOnline ||
    appState.baseTrajectoryBusy ||
    baseTrajectoryIsActive() ||
    recordingIsActive() ||
    appState.replay?.active === true ||
    actRolloutIsActive() ||
    appState.teleopBusy ||
    appState.teleop?.running === true ||
    appState.teleop?.busy === true ||
    cartesianJogEnabled(appState.cartesianJog || {}) ||
    cartesianJogServerBusy(appState.cartesianJog || {})
  ) {
    return;
  }
  const label =
    String(el["base-trajectory-label"].value || "").trim() || "base-trajectory";
  setBaseTrajectoryBusy(true);
  try {
    const payload = await requestJSON(API.baseTrajectoryRecordStart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!payload?.base_trajectory) {
      throw new Error("底盘轨迹录制启动接口返回异常");
    }
    renderBaseTrajectory(payload.base_trajectory);
    showToast("底盘轨迹开始录制；记录的是底盘轨迹，不是遥操 episode。");
  } catch (error) {
    showToast(`无法开始底盘轨迹录制：${readableError(error)}`, "error");
    await refreshBaseTrajectoryStatus().catch(() => {});
  } finally {
    setBaseTrajectoryBusy(false);
  }
}

async function stopBaseTrajectoryRecording() {
  if (!baseTrajectoryIsActive() || appState.baseTrajectoryBusy) return;
  setBaseTrajectoryBusy(true);
  try {
    const payload = await requestJSON(API.baseTrajectoryRecordStop, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!payload?.base_trajectory) {
      throw new Error("底盘轨迹录制停止接口返回异常");
    }
    renderBaseTrajectory(payload.base_trajectory);
    showToast("底盘轨迹正在停止并保存。");
  } catch (error) {
    showToast(`无法停止底盘轨迹录制：${readableError(error)}`, "error");
    await refreshBaseTrajectoryStatus().catch(() => {});
  } finally {
    setBaseTrajectoryBusy(false);
  }
}

async function startBaseTrajectoryReplay(item) {
  if (
    appState.baseTrajectoryReplayBusy ||
    appState.baseTrajectoryBusy ||
    baseTrajectoryIsActive() ||
    recordingIsActive() ||
    appState.replay?.active === true ||
    actRolloutIsActive() ||
    appState.teleopBusy ||
    appState.teleop?.running === true ||
    appState.teleop?.busy === true ||
    cartesianJogEnabled(appState.cartesianJog || {}) ||
    cartesianJogServerBusy(appState.cartesianJog || {})
  ) {
    return;
  }
  const recordingId = String(item.id || "").trim();
  if (!recordingId) return;
  setBaseTrajectoryReplayBusy(true);
  try {
    const preflightPayload = await requestJSON(API.baseTrajectoryReplayPreflight, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recording_id: recordingId }),
    });
    const preflight = preflightPayload?.preflight;
    if (!preflight) throw new Error("底盘轨迹预检接口返回异常");
    if (preflight.ready !== true) {
      const blockerText = Array.isArray(preflight.blockers) && preflight.blockers.length
        ? preflight.blockers.join("；")
        : "预检未通过";
      showToast(`底盘轨迹预检未通过：${blockerText}`, "error");
      return;
    }
    const confirmation = window.prompt(
      `将按底盘轨迹重放“${String(item.label || recordingId)}”，预计 ${formatDuration(preflight.recording?.duration_s || item.duration_s || 0)}。\n` +
        "请确认路径清空，并完整输入确认串：\nREPLAY_BASE_TRAJECTORY",
      ""
    );
    if (confirmation === null) return;
    const payload = await requestJSON(API.baseTrajectoryReplayStart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recording_id: recordingId,
        confirmation,
      }),
    });
    if (!payload?.base_trajectory) {
      throw new Error("底盘轨迹回放启动接口返回异常");
    }
    renderBaseTrajectory(payload.base_trajectory);
    showToast("底盘轨迹回放已启动。");
  } catch (error) {
    showToast(`无法开始底盘轨迹回放：${readableError(error)}`, "error");
    await refreshBaseTrajectoryStatus().catch(() => {});
  } finally {
    setBaseTrajectoryReplayBusy(false);
  }
}

async function stopBaseTrajectoryReplay() {
  if (appState.baseTrajectoryReplayBusy || baseTrajectoryState(appState.baseTrajectory) !== "replaying") return;
  setBaseTrajectoryReplayBusy(true);
  try {
    const payload = await requestJSON(API.baseTrajectoryReplayStop, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!payload?.base_trajectory) {
      throw new Error("底盘轨迹回放停止接口返回异常");
    }
    renderBaseTrajectory(payload.base_trajectory);
    showToast("底盘轨迹回放已停止并保持当前位置。");
  } catch (error) {
    showToast(`无法停止底盘轨迹回放：${readableError(error)}`, "error");
    await refreshBaseTrajectoryStatus().catch(() => {});
  } finally {
    setBaseTrajectoryReplayBusy(false);
  }
}

function updateDisplayedRecordingPath() {
  const recording = appState.recording || {};
  const active = recordingIsActive();
  const savedPath = String(recording.saved_path || "");
  const targetPath = active ? String(recording.target_path || "") : "";
  const latestPath = !active
    ? String(appState.recordings[0]?.path || "")
    : "";
  const displayPath =
    savedPath || targetPath || latestPath || "尚无已保存轨迹";
  const copyablePath = savedPath || latestPath;
  el["recording-path"].textContent = displayPath;
  el["recording-path"].title = displayPath;
  el["copy-recording-path"].dataset.copyPath = copyablePath;
  el["copy-recording-path"].disabled = !copyablePath;
}

function updateDisplayedBaseTrajectoryPath() {
  const baseTrajectory = appState.baseTrajectory || {};
  const displayPath =
    String(baseTrajectory.last_saved?.path || baseTrajectory.recording?.path || baseTrajectory.path || "");
  const active = baseTrajectoryIsActive();
  const copyablePath = displayPath && !active
    ? displayPath
    : String(baseTrajectory.last_saved?.path || baseTrajectory.recording?.path || "");
  el["base-trajectory-path"].textContent = active && !displayPath
    ? "底盘轨迹录制中，停止后生成保存路径"
    : displayPath || "尚无已保存底盘轨迹";
  el["base-trajectory-path"].title = el["base-trajectory-path"].textContent;
  el["copy-base-trajectory-path"].dataset.copyPath = copyablePath;
  el["copy-base-trajectory-path"].disabled = !copyablePath;
}

async function copyPath(path) {
  const value = String(path || "").trim();
  if (!value) {
    showToast("当前还没有可复制的保存路径。", "error");
    return;
  }
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.readOnly = true;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("浏览器拒绝复制");
    }
    showToast("完整录制路径已复制。");
  } catch (error) {
    showToast(`复制失败：${readableError(error)}`, "error");
  }
}

function renderProfile(camera) {
  const color = camera.color || {};
  const depth = camera.depth || {};
  const approved = camera.profile_approved === true;

  el["camera-name"].textContent = camera.name || "front";
  el["camera-serial"].textContent = `S/N ${camera.serial || "—"}`;
  el["camera-mode"].textContent = humanizeCameraMode(camera.mode);
  el["camera-resolution"].textContent =
    color.width && color.height ? `${color.width} × ${color.height}` : "— × —";
  el["camera-fps"].textContent = color.fps ? `${formatNumber(color.fps, 0)} FPS` : "— FPS";
  el["camera-alignment"].textContent =
    camera.aligned_to ? `Depth → ${camera.aligned_to}` : "Depth alignment —";

  if (el["profile-status"]) {
    el["profile-color"].textContent = streamLabel(color);
    el["profile-depth"].textContent = streamLabel(depth);
    el["profile-aligned"].textContent = camera.aligned_to || "未声明";
    el["profile-intrinsics"].textContent = intrinsicsLabel(camera.intrinsics);
    el["profile-status"].textContent = approved ? "配置已批准" : "配置未批准";
    el["profile-status"].classList.toggle("is-approved", approved);
    el["profile-status"].classList.toggle("is-rejected", !approved);
    el["profile-note"].textContent = camera.error
      ? camera.error
      : approved
        ? "设备、分辨率、内参与深度对齐满足当前感知契约。"
        : "配置未批准时只能预览，不能把坐标交给后续运动规划。";
  }
}

function renderProfileOffline(detail) {
  if (!el["profile-status"]) return;
  el["profile-status"].textContent = "接口离线";
  el["profile-status"].classList.remove("is-approved");
  el["profile-status"].classList.add("is-rejected");
  el["profile-note"].textContent = detail || "无法读取相机配置。";
}

function renderGates(gates) {
  if (!el["gate-list"]) return;
  if (!gates.length) {
    el["gate-list"].innerHTML = gateMarkup({
      label: "尚未配置门禁",
      detail: "服务没有返回 gates",
      passed: false,
      pending: true,
    });
    setGateSummary(0, 0, true);
    return;
  }

  const passed = gates.filter((gate) => gate.passed === true).length;
  el["gate-list"].innerHTML = gates
    .map((gate) =>
      gateMarkup({
        label: gate.label || gate.id || "未命名门禁",
        detail: gate.detail || gate.id || "—",
        passed: gate.passed === true,
        pending: gate.passed == null,
      })
    )
    .join("");
  setGateSummary(passed, gates.length, false);
}

function renderGatesOffline() {
  if (!el["gate-list"]) return;
  el["gate-list"].innerHTML = gateMarkup({
    label: "状态服务离线",
    detail: "无法读取 /api/status",
    passed: false,
    pending: true,
  });
  el["gate-summary"].textContent = "未知";
  el["gate-summary"].classList.remove("is-ready");
  el["gate-summary"].classList.add("is-blocked");
}

function gateMarkup({ label, detail, passed, pending }) {
  const stateClass = pending
    ? "gate-item--pending"
    : passed
      ? "gate-item--passed"
      : "gate-item--failed";
  const stateLabel = pending ? "未知" : passed ? "通过" : "阻塞";
  return `
    <li class="gate-item ${stateClass}">
      <span class="gate-icon" aria-hidden="true"></span>
      <div>
        <strong>${escapeHTML(String(label))}</strong>
        <small title="${escapeHTML(String(detail))}">${escapeHTML(String(detail))}</small>
      </div>
      <span>${stateLabel}</span>
    </li>
  `;
}

function setGateSummary(passed, total, pending) {
  if (!el["gate-summary"]) return;
  const allPassed = total > 0 && passed === total;
  el["gate-summary"].textContent = pending
    ? "等待状态"
    : allPassed
      ? `${passed}/${total} 已通过`
      : `${passed}/${total} 已通过`;
  el["gate-summary"].classList.toggle("is-ready", allPassed);
  el["gate-summary"].classList.toggle("is-blocked", !allPassed && !pending);
}

function renderDetection(detection) {
  const detected2d = detection?.detected_2d === true;
  const targetReady = detection?.target_ready === true;
  const candidate = detection?.candidate || null;
  const blockers = Array.isArray(detection?.blockers) ? detection.blockers : [];
  const isShippingBox = detection?.type === "task2_shipping_box_opening";
  const count =
    detection?.recognized_count ??
    detection?.count ??
    detection?.instance_count ??
    (Array.isArray(detection?.candidates)
      ? detection.candidates.length
      : detected2d
        ? 1
        : 0);
  const captureDetail = detectionCaptureDetail(detection);
  const layerEstimate = detection?.layer_estimate || null;
  const layerDetail = layerEstimate?.valid === true
    ? `；估计 ${Number(layerEstimate.layer)} 层（顶面距桌面 ${formatNumber(Number(layerEstimate.measured_height_m) * 1000, 1)} mm）`
    : "";
  const consensus = detection?.temporal_consensus || null;
  const consensusDetail = consensus?.valid === true
    ? `；多帧一致（${Number(consensus.required || 1)} 帧${Number.isFinite(Number(consensus.matched_distance_m)) ? `，偏差 ${formatNumber(Number(consensus.matched_distance_m) * 1000, 1)} mm` : ""}）`
    : "";
  const observed = Number(detection?.layout?.observed_instance_count);
  const safeGraspCount = Number(detection?.safe_grasp_candidate_count);
  const instanceDetail = Number.isFinite(observed)
    ? appState.activeTask === "task1"
      ? `；识别 ${observed} 个${Number.isFinite(safeGraspCount) ? `，本帧安全候选 ${safeGraspCount} 个` : ""}`
      : `；当前任务布局 ${observed} 个单盒实例`
    : "";

  if (isShippingBox) {
    renderFixedAxis(null);
    el["detection-pixel-label"].textContent = "开口中心像素";
    el["detection-count-label"].textContent = "纸箱候选数量";
    el["detection-angle-label"].textContent = "纸箱长轴方向";
    el["candidate-count"].textContent = String(count);
    if (!detected2d || !candidate) {
      setDetectionState("未发现纸箱", "error");
      el["base-coordinate"].textContent = "X —   Y —   Z —";
      el["suction-pixel"].textContent = "—, —";
      el["detection-score"].textContent = "—";
      el["surface-tilt"].textContent = "—";
      el["detection-message"].textContent = blockers.length
        ? `纸箱识别失败：${blockers.map(humanizeBlocker).join("；")}${captureDetail}`
        : `搜索区域内没有找到满足门限的开放纸箱。${captureDetail}`;
      drawDetectionOverlay();
      return;
    }
    const pointBase =
      candidate.opening_center_left_base_m || detection.point_left_base_m || null;
    const centerPixel = candidate.center_px || candidate.suction_px || null;
    const score = candidate.score ?? detection.score;
    const yaw = candidate.yaw_left_base_deg ?? detection.yaw_left_base_deg;
    const openingSize = candidate.opening_size_m || detection.opening_size_m;
    const cavityDepth = Number(candidate.cavity_depth_m ?? detection.cavity_depth_m);
    const rimZ = Number(candidate.rim_z_m ?? detection.rim_z_m);
    const bottomZ = Number(candidate.bottom_z_m ?? detection.bottom_z_m);
    setDetectionState(
      targetReady ? "纸箱开口可用" : "纸箱已发现 / 阻塞",
      targetReady ? "success" : "error"
    );
    el["base-coordinate"].textContent = pointBase
      ? `X ${formatMeters(pointBase[0])}  Y ${formatMeters(pointBase[1])}  Z ${formatMeters(pointBase[2])}`
      : "X —   Y —   Z —";
    el["suction-pixel"].textContent = centerPixel
      ? `${formatNumber(centerPixel[0], 0)}, ${formatNumber(centerPixel[1], 0)}`
      : "—, —";
    el["detection-score"].textContent = Number.isFinite(Number(score))
      ? formatNumber(Number(score), 3)
      : "—";
    el["surface-tilt"].textContent = Number.isFinite(Number(yaw))
      ? `${formatNumber(Number(yaw), 1)}°`
      : "—";
    const sizeDetail =
      Array.isArray(openingSize) && openingSize.length >= 2
        ? `；开口 ${formatNumber(Number(openingSize[0]) * 1000, 0)} × ${formatNumber(Number(openingSize[1]) * 1000, 0)} mm`
        : "";
    const depthDetail =
      Number.isFinite(rimZ) && Number.isFinite(bottomZ) && Number.isFinite(cavityDepth)
        ? `；箱沿 Z ${formatNumber(rimZ * 1000, 1)} mm，箱底 Z ${formatNumber(bottomZ * 1000, 1)} mm，深度 ${formatNumber(cavityDepth * 1000, 1)} mm`
        : "";
    el["detection-message"].textContent = blockers.length
      ? `纸箱已找到，但安全门禁阻塞：${blockers.map(humanizeBlocker).join("；")}${captureDetail}`
      : targetReady
        ? `四个内箱沿通过 RGB-D 门限${sizeDetail}${depthDetail}${consensusDetail}；已缓存开口中心，本步骤未移动机械臂、未操作吸盘。${captureDetail}`
        : `纸箱已找到，但没有形成稳定三维开口目标。${captureDetail}`;
    drawDetectionOverlay();
    return;
  }

  renderFixedAxis(detection);

  el["candidate-count"].textContent = String(count);

  if (!detected2d || !candidate) {
    setDetectionState("未发现", "error");
    resetDetectionMetrics();
    el["candidate-count"].textContent = String(count);
    el["detection-message"].textContent = blockers.length
      ? `阻塞原因：${blockers.map(humanizeBlocker).join("；")}${captureDetail}`
      : `画面中没有满足二维几何门限的药盒。${captureDetail}`;
    drawDetectionOverlay();
    return;
  }

  const pointBase =
    candidate.point_left_base_m ||
    detection.point_left_base_m ||
    candidate.base_point_m ||
    null;
  const dualSuction =
    detection.dual_suction_target || candidate.dual_suction || null;
  const suctionPixel =
    dualSuction?.midpoint_px ||
    candidate.suction_px ||
    candidate.suction_pixel ||
    detection.suction_px ||
    null;
  const score = candidate.score ?? detection.score;
  const tilt = candidate.surface_tilt_deg ?? detection.surface_tilt_deg;

  setDetectionState(
    targetReady ? "目标可用" : "二维已发现 / 阻塞",
    targetReady ? "success" : "error"
  );
  el["base-coordinate"].textContent = pointBase
    ? `X ${formatMeters(pointBase[0])}  Y ${formatMeters(pointBase[1])}  Z ${formatMeters(pointBase[2])}`
    : "X —   Y —   Z —";
  el["suction-pixel"].textContent = suctionPixel
    ? `${formatNumber(suctionPixel[0], 0)}, ${formatNumber(suctionPixel[1], 0)}`
    : "—, —";
  el["detection-score"].textContent =
    Number.isFinite(Number(score)) ? formatNumber(Number(score), 3) : "—";
  el["surface-tilt"].textContent =
    Number.isFinite(Number(tilt)) ? `${formatNumber(Number(tilt), 1)}°` : "—";
  el["detection-message"].textContent = blockers.length
    ? `二维目标已找到，但仍被门禁阻塞：${blockers.map(humanizeBlocker).join("；")}${captureDetail}`
    : targetReady
      ? `目标通过当前感知门限${layerDetail}${consensusDetail}${instanceDetail}；这里只输出坐标，不执行抓取。${captureDetail}`
      : `二维目标已找到，但未形成可执行三维目标。${captureDetail}`;

  drawDetectionOverlay();
}

function renderDetectionError(message) {
  setDetectionState("接口错误", "error");
  el["detection-message"].textContent = message;
  renderFixedAxis(null);
}

function resetDetectionMetrics() {
  el["detection-pixel-label"].textContent = "吸取像素";
  el["detection-count-label"].textContent = "单盒实例数量";
  el["detection-angle-label"].textContent = "表面倾角（未开放）";
  el["base-coordinate"].textContent = "X —   Y —   Z —";
  el["suction-pixel"].textContent = "—, —";
  el["detection-score"].textContent = "—";
  el["surface-tilt"].textContent = "—";
}

function renderFixedAxis(detection) {
  const profile = TASK_PROFILES[appState.activeTask] || TASK_PROFILES.task1;
  const calibration = appState.fixedSuctionAxis || {};
  const candidate = detection?.candidate || null;
  const dualSuction =
    detection?.dual_suction_target || candidate?.dual_suction || null;
  const isFixedProjection = dualSuction?.alignment === "fixed_tool_axis";
  const cups = dualSuction?.cup_centers_px;
  const hasCups =
    Array.isArray(cups) &&
    cups.length === 2 &&
    cups.every((point) => Array.isArray(point) && point.length >= 2);

  if (!profile.available) {
    setFixedAxisState(
      calibration.ready === true ? "工具轴已标定" : "工具轴待标定",
      calibration.ready === true ? "success" : "idle"
    );
    el["fixed-axis-cup-a"].textContent = "—, —";
    el["fixed-axis-cup-b"].textContent = "—, —";
    el["fixed-axis-angle"].textContent = "—";
    el["fixed-axis-clearance"].textContent = "未计算";
    el["fixed-axis-message"].textContent =
      calibration.ready === true
        ? "工具轴标定可复用；先建立该任务的药盒识别 profile，再运行真实杯心投影。"
        : "先建立该任务的药盒识别 profile；真实固定工具轴仍共用同一套杯心标定。";
    return;
  }

  if (!candidate || !hasCups) {
    setFixedAxisState(
      calibration.ready === true ? "已标定 · 待检测" : "待标定",
      calibration.ready === true ? "success" : "idle"
    );
    el["fixed-axis-cup-a"].textContent = "—, —";
    el["fixed-axis-cup-b"].textContent = "—, —";
    el["fixed-axis-angle"].textContent = "—";
    el["fixed-axis-clearance"].textContent = "—";
    el["fixed-axis-message"].textContent =
      calibration.ready === true
        ? "真实工具轴参数已就绪；下一次检测将运行固定杯心投影与同面校验。"
        : "当前显示的是视觉规划杯心；完成一次真实杯心轴标定后，再把固定工具轴投影作为执行门禁。";
    return;
  }

  const valid2d = dualSuction.valid_2d === true;
  const depthSupport = dualSuction.depth_support || {};
  const depthAvailable = depthSupport.available === true;
  const depthValid = depthSupport.valid === true;
  const margins = dualSuction.raw_edge_margins_mm || {};
  const longMargin = Number(margins.long_end);
  const shortMargin = Number(margins.short_side);
  const marginValues = [longMargin, shortMargin].filter(Number.isFinite);
  const minimumMargin = marginValues.length ? Math.min(...marginValues) : null;
  const angle = Number(dualSuction.axis_angle_deg);

  el["fixed-axis-cup-a"].textContent =
    `${formatNumber(cups[0][0], 0)}, ${formatNumber(cups[0][1], 0)}`;
  el["fixed-axis-cup-b"].textContent =
    `${formatNumber(cups[1][0], 0)}, ${formatNumber(cups[1][1], 0)}`;
  el["fixed-axis-angle"].textContent = Number.isFinite(angle)
    ? `${formatNumber(angle, 1)}°`
    : "—";
  el["fixed-axis-clearance"].textContent = valid2d
    ? depthAvailable
      ? depthValid
        ? `同面通过${minimumMargin == null ? "" : ` · 余量 ${formatNumber(minimumMargin, 1)} mm`}`
        : "深度同面阻塞"
      : "二维通过 · 深度待取"
    : "杯面越界";

  if (valid2d && (!depthAvailable || depthValid)) {
    setFixedAxisState(
      isFixedProjection ? "真实杯心通过" : "规划杯心通过",
      "success"
    );
  } else {
    setFixedAxisState("规划杯心阻塞", "error");
  }
  el["fixed-axis-message"].textContent = isFixedProjection
    ? "当前为已标定的真实固定工具轴投影；机械臂保持锁定姿态，不跟随药盒旋转框。"
    : "这里仍是沿候选药盒长轴生成的规划结果，不代表腕部固定轴；真实执行门禁需用标定后的两个杯心投影替换。";
}

function setFixedAxisState(label, state) {
  el["fixed-axis-state"].textContent = label;
  el["fixed-axis-state"].className = `result-pill result-pill--${state || "idle"}`;
}

function setDetectionBusy(busy) {
  appState.detectionBusy = busy;
  updateDetectionButton();
  el["run-detection"].classList.toggle("is-loading", busy);
  updateTask1PickControls();
  if (busy) {
    setDetectionState("识别中", "busy");
    el["detection-message"].textContent =
      "正在获取同步 RGB-D 并计算吸取点…";
  }
}

function setDetectionState(label, state) {
  el["detection-state"].textContent = label;
  el["detection-state"].className = `result-pill result-pill--${state || "idle"}`;
}

function drawDetectionOverlay() {
  const canvas = el["detection-overlay"];
  const stage = el["camera-stage"];
  const image = el["camera-frame"];
  if (!canvas || !stage) return;

  const width = stage.clientWidth;
  const height = stage.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  // The backend-rendered overlay is tied to the exact captured frame. Never
  // draw its old coordinates over a later preview frame.
  if (appState.detection?.overlay_url) return;

  const candidate = appState.detection?.candidate;
  if (
    !candidate ||
    appState.detection?.detected_2d !== true ||
    !image.naturalWidth
  ) {
    return;
  }

  const sourceWidth =
    appState.profile?.color?.width || image.naturalWidth;
  const sourceHeight =
    appState.profile?.color?.height || image.naturalHeight;
  const sourceRatio = sourceWidth / sourceHeight;
  const stageRatio = width / height;
  let displayWidth;
  let displayHeight;
  let offsetX;
  let offsetY;

  if (sourceRatio > stageRatio) {
    displayWidth = width;
    displayHeight = width / sourceRatio;
    offsetX = 0;
    offsetY = (height - displayHeight) / 2;
  } else {
    displayHeight = height;
    displayWidth = height * sourceRatio;
    offsetX = (width - displayWidth) / 2;
    offsetY = 0;
  }

  const transformPoint = (point) => [
    offsetX + (Number(point[0]) / sourceWidth) * displayWidth,
    offsetY + (Number(point[1]) / sourceHeight) * displayHeight,
  ];

  const polygon = candidate.polygon_px || candidate.polygon || [];
  if (Array.isArray(polygon) && polygon.length >= 3) {
    context.beginPath();
    polygon.forEach((point, index) => {
      const [x, y] = transformPoint(point);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.closePath();
    context.fillStyle = "rgba(93, 242, 179, 0.10)";
    context.strokeStyle = "#5df2b3";
    context.lineWidth = 1.5;
    context.fill();
    context.stroke();
  }

  const dualSuction =
    appState.detection?.dual_suction_target || candidate.dual_suction || null;
  const cups = dualSuction?.cup_centers_px;
  if (
    Array.isArray(cups) &&
    cups.length === 2 &&
    cups.every((point) => Array.isArray(point) && point.length >= 2)
  ) {
    const [cupA, cupB] = cups.map(transformPoint);
    const midpoint = transformPoint(
      dualSuction.midpoint_px || [
        (Number(cups[0][0]) + Number(cups[1][0])) / 2,
        (Number(cups[0][1]) + Number(cups[1][1])) / 2,
      ]
    );
    const sourceRadius = Number(dualSuction.projected_cup_radius_px);
    const displayScale = 0.5 * (
      displayWidth / sourceWidth + displayHeight / sourceHeight
    );
    const radius = Number.isFinite(sourceRadius)
      ? Math.max(4, sourceRadius * displayScale)
      : 12;
    context.strokeStyle = dualSuction.valid_2d === true ? "#5df2b3" : "#f0bd68";
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(cupA[0], cupA[1]);
    context.lineTo(cupB[0], cupB[1]);
    for (const cup of [cupA, cupB]) {
      context.moveTo(cup[0] + radius, cup[1]);
      context.arc(cup[0], cup[1], radius, 0, Math.PI * 2);
    }
    context.moveTo(midpoint[0] - 8, midpoint[1]);
    context.lineTo(midpoint[0] + 8, midpoint[1]);
    context.moveTo(midpoint[0], midpoint[1] - 8);
    context.lineTo(midpoint[0], midpoint[1] + 8);
    context.stroke();

    context.fillStyle = "rgba(3, 9, 7, 0.85)";
    context.fillRect(midpoint[0] + 12, midpoint[1] - 19, 78, 17);
    context.fillStyle = context.strokeStyle;
    context.font = `700 9px ${getComputedStyle(document.documentElement).getPropertyValue("--mono")}`;
    context.fillText("DUAL · PLAN", midpoint[0] + 17, midpoint[1] - 7);
  } else {
    const suction = candidate.suction_px || candidate.suction_pixel;
    if (!Array.isArray(suction) || suction.length < 2) return;
    const [x, y] = transformPoint(suction);
    context.strokeStyle = "#f0bd68";
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(x, y, 12, 0, Math.PI * 2);
    context.moveTo(x - 18, y);
    context.lineTo(x + 18, y);
    context.moveTo(x, y - 18);
    context.lineTo(x, y + 18);
    context.stroke();

    context.fillStyle = "rgba(3, 9, 7, 0.85)";
    context.fillRect(x + 15, y - 19, 58, 17);
    context.fillStyle = "#f0bd68";
    context.font = `700 9px ${getComputedStyle(document.documentElement).getPropertyValue("--mono")}`;
    context.fillText("SUCTION", x + 20, y - 7);
  }
}

function setServiceAvailability(online, detail) {
  appState.serviceOnline = online;
  el["connection-state"].classList.toggle("is-online", online);
  el["connection-state"].classList.toggle("is-offline", !online);
  el["connection-label"].textContent = online ? "服务在线" : "服务离线";
  el["connection-state"].title = detail || "";
  updateDetectionButton();
  updateRecordingButtons();
  updateTeleopButton();
  updateCartesianJogControls();
  updateTask1PickControls();
}

function updateDetectionButton() {
  const profile = TASK_PROFILES[appState.activeTask];
  el["run-detection"].disabled =
    profile?.available !== true ||
    appState.detectionBusy ||
    recordingIsActive() ||
    !appState.serviceOnline;
}

function showDetectionFrame(detection) {
  const overlayUrl = String(detection?.overlay_url || "");
  if (!overlayUrl.startsWith("/api/camera/frame.jpg")) {
    releaseDetectionFrame();
    refreshFrame(false);
    return;
  }
  appState.displayingDetectionFrame = true;
  appState.detectionFramePinnedUntil = Date.now() + DETECTION_FRAME_HOLD_MS;
  requestCameraFrame(overlayUrl, "检测画面请求超时");
}

function releaseDetectionFrame() {
  appState.displayingDetectionFrame = false;
  appState.detectionFramePinnedUntil = 0;
}

function setCameraAvailability(online, detail) {
  appState.cameraOnline = online;
  const cameraMode = String(appState.profile?.mode || "").toLowerCase();
  const isDirectLiveFeed = online && cameraMode === "realsense";
  const isSharedLiveFeed =
    online &&
    (cameraMode === "shared" ||
      cameraMode === "shared_memory" ||
      cameraMode === "web_console");
  const isLiveFeed = isDirectLiveFeed || isSharedLiveFeed;
  el["camera-frame"].classList.toggle("is-visible", online);
  el["camera-unavailable"].classList.toggle("is-hidden", online);
  el["camera-live-badge"].classList.toggle("is-live", isLiveFeed);
  el["camera-live-badge"].lastChild.textContent = online
    ? isDirectLiveFeed
      ? " LIVE"
      : isSharedLiveFeed
        ? " SHARED LIVE"
        : " OFFLINE PREVIEW"
    : " OFFLINE";
  if (!online) {
    el["camera-offline-detail"].textContent = detail || "等待前置相机画面";
    el["frame-time"].textContent = "尚无画面";
  }
  renderFixedAxisCalibration();
}

async function requestJSON(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      cache: "no-store",
      credentials: "same-origin",
      ...options,
      signal: controller.signal,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(
        payload?.error ||
        payload?.message ||
        `${response.status} ${response.statusText}`.trim()
      );
    }
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求超时");
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast${type === "error" ? " toast--error" : ""}`;
  toast.textContent = message;
  el["toast-region"].appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function formatMeters(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(3)} m` : "—";
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function formatTime(value) {
  const normalized =
    typeof value === "number" && value > 0 && value < 10_000_000_000
      ? value * 1000
      : value;
  const date = normalized instanceof Date ? normalized : new Date(normalized);
  if (Number.isNaN(date.getTime())) return String(value || "—");
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatDuration(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder
    .toFixed(1)
    .padStart(4, "0")}`;
}

function poseText(pose) {
  if (!pose || typeof pose !== "object") return "—";
  const x = Number(pose.x);
  const y = Number(pose.y);
  const yaw = Number(pose.yaw);
  if (![x, y, yaw].every(Number.isFinite)) return "—";
  return `X ${x.toFixed(3)} · Y ${y.toFixed(3)} · yaw ${yaw.toFixed(3)}`;
}

function purposeLabel(value) {
  const labels = {
    act_bimanual: "ACT 双臂示教",
    calibration_left: "左臂手眼标定",
    calibration_right: "右臂手眼标定",
    projection_validation: "双臂投影验证",
    trajectory_both: "普通双臂轨迹",
    trajectory_left: "仅左臂轨迹",
    trajectory_right: "仅右臂轨迹",
  };
  return labels[String(value || "")] || String(value || "未指定");
}

function detectionCaptureDetail(detection) {
  const capturedAt = Number(detection?.captured_at);
  if (!Number.isFinite(capturedAt) || capturedAt <= 0) return "";
  const capturedMs = capturedAt < 10_000_000_000 ? capturedAt * 1000 : capturedAt;
  const ageSeconds = Math.max(0, (Date.now() - capturedMs) / 1000);
  const ageLabel =
    ageSeconds < 1 ? "刚刚" : ageSeconds < 60 ? `${ageSeconds.toFixed(1)} 秒前` : "超过 1 分钟前";
  return `；检测帧 ${formatTime(capturedAt)}（${ageLabel}）`;
}

function streamLabel(stream) {
  if (!stream || !stream.width || !stream.height) return "未配置";
  const fps = stream.fps ? ` @ ${formatNumber(stream.fps, 0)} FPS` : "";
  const format = stream.format ? ` · ${stream.format}` : "";
  return `${stream.width}×${stream.height}${fps}${format}`;
}

function intrinsicsLabel(intrinsics) {
  if (!Array.isArray(intrinsics) || !intrinsics.length) return "未配置";
  if (Array.isArray(intrinsics[0])) {
    const fx = Number(intrinsics[0]?.[0]);
    const fy = Number(intrinsics[1]?.[1]);
    if (Number.isFinite(fx) && Number.isFinite(fy)) {
      return `fx ${fx.toFixed(1)} · fy ${fy.toFixed(1)}`;
    }
  }
  const fx = Number(intrinsics.fx);
  const fy = Number(intrinsics.fy);
  return Number.isFinite(fx) && Number.isFinite(fy)
    ? `fx ${fx.toFixed(1)} · fy ${fy.toFixed(1)}`
    : "已配置";
}

function humanizeCameraMode(mode) {
  const labels = {
    readonly: "只读采集",
    read_only: "只读采集",
    live: "实时采集",
    realsense: "实时 RGB-D",
    offline: "离线",
  };
  return labels[String(mode || "").toLowerCase()] || mode || "只读采集";
}

function humanizeBlocker(value) {
  const labels = {
    no_candidate: "未找到候选药盒",
    missing_depth: "目标深度缺失",
    stale_depth: "深度帧过期",
    surface_tilt: "表面倾角超限",
    plane_residual: "表面不够平整",
    physical_size: "物理尺寸不符",
    physical_size_long_mismatch: "候选长边不是单个药盒尺寸",
    physical_size_short_mismatch: "候选短边不是单个药盒尺寸",
    physical_size_depth_support_low: "物理尺寸测量区域深度不足",
    task_surface_height_mismatch: "药盒表面高度与当前任务不符",
    target_not_temporally_consistent: "多帧识别结果不一致",
    unreachable: "目标超出左臂工作区",
    tcp_not_calibrated: "吸盘 TCP 未标定",
    camera_profile_unapproved: "相机配置未批准",
    dual_suction_dimensions_inconsistent: "双吸盘尺寸配置不一致",
    dual_suction_face_not_allowed: "当前不是可吸附的大面",
    candidate_not_graspable: "候选药盒未通过二维抓取门禁",
    dual_suction_geometry_unavailable: "双吸盘几何信息不足",
    dual_suction_margin_low: "吸盘边缘安全余量不足",
    dual_suction_outside_image: "吸盘接触区域超出画面",
    shipping_box_not_found: "搜索区域内未找到开放纸箱",
    shipping_box_depth_unavailable: "纸箱 RGB-D 深度不可用",
    shipping_box_depth_invalid: "纸箱深度帧无效",
    shipping_box_intrinsics_unavailable: "相机内参不可用",
    shipping_box_handeye_unavailable: "相机到左臂基座标定不可用",
    shipping_box_rim_depth_incomplete: "四个箱沿深度不完整",
    shipping_box_bottom_depth_invalid: "箱底深度不足",
    shipping_box_cavity_depth_invalid: "箱沿与箱底高度差不符合开放纸箱",
    shipping_box_opening_size_invalid: "纸箱开口尺寸超出门限",
    shipping_box_outside_configured_workspace: "纸箱开口超出配置工作区",
    shipping_box_score_low: "纸箱二维置信度不足",
    shipping_box_target_not_temporally_consistent: "纸箱开口多帧坐标不一致",
    no_surface_in_staged_top_height_band: "固定工位高度范围内没有检测到顶部平面",
    no_staged_carton_top_passed_rgbd_geometry: "固定工位目标未通过顶部窄面几何门禁",
    staged_top_barcode_not_found: "顶部窄面右侧未检测到条码平行条纹",
    staged_top_area_mismatch: "顶部窄面像素面积不符",
    staged_top_long_size_mismatch: "顶部窄面长边不是约130毫米",
    staged_top_short_size_mismatch: "顶部窄面短边不是约25毫米",
    staged_top_surface_not_planar: "顶部窄面高度起伏过大",
    staged_top_not_temporally_consistent: "顶部窄面多帧三维中心不一致",
  };
  const key = String(value || "").toLowerCase();
  return labels[key] || String(value || "未知阻塞");
}

function readableError(error) {
  if (!error) return "未知错误";
  return error.message || String(error);
}

function escapeHTML(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
