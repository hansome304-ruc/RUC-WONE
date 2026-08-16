# Medicine Agentic Skills

## `task1.pick_carton`

Pick one folded medicine carton with the left dual-suction tool.

- Descriptor: `GET /api/skills/task1/pick-carton`
- Execute: `POST /api/skills/task1/pick-carton`
- Request content type: `application/json`
- Request body: `{}`
- Selection policy: highest valid layer, then nearest left-arm-base XY distance
- Success postcondition: one carton is held 20 mm above its contact height

The skill performs these stages atomically:

1. Read and lock the current left-tool orientation if no session is active.
2. Verify that orientation against the calibrated downward suction pose.
3. Capture synchronized RGB-D and split a touching carton array into cells.
4. Select the highest-layer, nearest reachable carton.
5. Approach at fixed orientation, engage both suction channels, and lift 20 mm.

Example:

```bash
curl -X POST http://127.0.0.1:8899/api/skills/task1/pick-carton \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Important preconditions:

- The physical work area is clear and the emergency stop is available.
- Teleoperation and trajectory recording are stopped.
- The left suction tool is already pointing vertically downward.
- Both suction cups are released and the arm is idle.

XY targets are not clipped by a software workspace; reachability and joint
limits are decided by the AIRBOT IK/planner. The configured flange-Z minimum
remains active as the table-contact guard.

The response contains `skill.stages`, the fresh `detection`, selected layer and
coordinates, executed approach trajectory, suction command result, and lift
result. A failed precondition returns a non-2xx response before target motion.

## `left_arm.reset_home`

Return only the left arm to the same shared AIRBOT initial joint pose used by
the existing dual-arm home implementation.

- Descriptor: `GET /api/skills/left-arm/reset-home`
- Execute: `POST /api/skills/left-arm/reset-home`
- Request content type: `application/json`
- Request body: `{}`
- Affected hardware: left arm only (`localhost:50051`)
- Target joints (rad): `[0, 0, 0, 1.5, 0, -1.5]`
- Motion profile: verified `SLOW` profile in `PLANNING_POS`
- Completion: live six-joint feedback settles within `0.12 rad`

The skill clears the old Cartesian-jog capture before the motion. It refuses
to start while teleoperation or trajectory recording is active. It never
commands the right arm, chassis, gripper, or suction channels.

```bash
curl -X POST http://127.0.0.1:8899/api/skills/left-arm/reset-home \
  -H 'Content-Type: application/json' \
  -d '{}'
```
