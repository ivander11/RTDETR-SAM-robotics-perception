# Lite6 model provenance

- Upstream: https://github.com/xArm-Developer/xarm_ros2
- Branch at import: humble
- Commit: ac736c92b02de7ca7e0be6f737d77d46707464d8
- Import date: 2026-10-24
- License: BSD-3-Clause; see LICENSE.

The selected files in upstream/xarm_description are unmodified copies:
the Lite6 URDF macro, common link and material macros, Lite6 joint limits,
default Lite6 kinematics, xarm6_type9_HT_BR2 inertial parameters, and Lite6
visual and collision meshes. The inertial filename is selected by the
upstream Lite6 macro; it is not an accidental xArm6 model substitution.

Our wrapper instantiates the bare six-joint arm, selects STL visual meshes,
and emits relative mesh paths. The exporter replaces three ROS package
lookups with an explicit directory argument in a temporary copy only.
No joint geometry or physical parameters are intentionally changed.

The generated model has seven links and six revolute joints. Its base must
be fixed when loaded. It has no gripper, world joint, or additional tool
frame; link6 is the last link, not a calibrated gripper TCP.

Rebuild from the repository root using Python with xacro==2.1.1 installed:

```bash
env -u PYTHONPATH venv/bin/python scripts/simulation/export_lite6.py
env -u PYTHONPATH venv/bin/python scripts/simulation/load_lite6.py
```

Add --gui to the loader for a ten-second visual inspection. The loader
checks successful loading and finite joint states, not physical accuracy,
reachability, collision safety, or RL task correctness.
