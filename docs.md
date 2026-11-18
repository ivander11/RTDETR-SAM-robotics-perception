# Temporary Lite6 Learning Checkpoint

> Local working notes. Do not commit yet.

## Current focus

The current implementation to understand is:

- `src/control/lite6_env/reach_env.py`

It defines a Gymnasium environment in which a DDPG policy commands relative
Cartesian end-effector movement and PyBullet inverse kinematics converts that
movement into six Lite6 joint targets.

## Current environment contract

### Action

The action is a normalized three-value vector:

```text
[dx, dy, dz], each in [-1, 1]
```

The base dense environment scales it by `0.02 m` per component. The HER wrapper
changes the scale to `0.025 m`.

### Observation

The dense environment returns nine values:

```text
[end-effector XYZ, end-effector linear velocity XYZ, goal XYZ]
```

### Motion pipeline

```text
policy action [dx, dy, dz]
    -> clip normalized action
    -> multiply by action_scale
    -> add displacement to current end-effector position
    -> clip target position to the workspace
    -> solve inverse kinematics
    -> command six joint positions
    -> advance PyBullet for 24 steps at 240 Hz
    -> read the resulting end-effector state
```

One policy action therefore advances approximately `0.1 s` of simulated time.

### Task and reward

- Dense reward: negative Euclidean distance to the goal.
- Success: end effector is within `0.01 m` of the goal.
- Natural termination: success.
- Time-limit truncation: 50 actions without success.
- Fixed and seeded random goal modes are available.

## Responsibilities of each function

- `__init__`: validates configuration, defines spaces and limits, connects to
  PyBullet, loads the URDF, discovers the joints, reads joint limits and creates
  the goal marker.
- `_find_revolute_joints`: finds the six controllable revolute joints in the
  URDF.
- `_create_goal_marker`: creates a non-colliding green sphere at the target.
- `reset`: resets six joints to zero, selects the episode goal, moves the goal
  marker and returns the initial observation.
- `step`: converts one Cartesian action into an IK solution, commands the
  joints, advances physics, calculates distance and reward, and returns the
  Gymnasium transition.
- `_command_joint_positions`: sends six targets to PyBullet position
  controllers.
- `_get_ee_state`: reads the world position and linear velocity of `link6`.
- `_get_observation`: concatenates position, velocity and goal into the
  nine-value observation.
- `_distance_to_goal`: calculates Euclidean distance between the end effector
  and target.
- `render`: satisfies the Gymnasium API; the PyBullet GUI performs rendering.
- `close`: disconnects the PyBullet client safely.

## What has been checked

- The official standalone Lite6 URDF loads successfully.
- The URDF exposes six revolute joints from `joint1` through `joint6`.
- The environment currently uses `link6` as the end effector.
- Gymnasium's environment checker passes.
- The observation shape is `(9,)`.
- Cartesian actions produce simulated movement through inverse kinematics.
- The episode limit works.
- Random goals are reproducible when reset with the same seed.

## Current simplifications

- The scene has no plane or table.
- The robot base is at the URDF origin.
- The goal is a visual marker without collision geometry.
- IK controls end-effector position without an explicit target orientation.
- `link6` is treated as the end effector; there is no modeled gripper or
  tool-center-point offset.
- The environment uses PyBullet position control and URDF effort limits rather
  than a calibrated model of physical Lite6 motor behavior.
- There are no grasping actions, object contact, collision-avoidance objective,
  camera observation or physical-arm feedback.
- `reach_env.py` currently combines robot simulation/control with the basic
  Gymnasium observation, reward and episode interface.

## Next review: closer to the paper and archived implementation

Before adding realism, distinguish paper reproduction from general simulation
upgrades.

Known archive-alignment topics to examine are:

1. Add the plane and table used by the archived scene.
2. Match the archived robot base position and coordinate frame.
3. Match the goal sampler and reachable goal region.
4. Set the archived fixed IK end-effector orientation.
5. Use the archived `0.025 m` Cartesian action scale.
6. Match the goal-conditioned observation, HER replay, reward condition and
   training configuration.

Possible realism upgrades that require separate justification are:

- Model a gripper and define its tool-center point.
- Add collision geometry and physical target objects.
- Add contact-aware rewards or collision penalties.
- Model joint velocity, acceleration, latency and controller limits closer to
  the physical Lite6.
- Calibrate scene dimensions, transforms and motor behavior against the real
  setup.

These realism upgrades are not automatically required to reproduce the
paper's coordinate-reaching experiment. Each change should be classified as
paper alignment, archive alignment, or a new real-world-oriented improvement
before implementation.
