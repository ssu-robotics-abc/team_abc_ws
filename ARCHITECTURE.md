# Architecture

This workspace is a ROS 2 system for voice-guided item handling: speech input is interpreted into item requests, camera perception detects objects, and manipulation nodes command a Doosan M0609 arm and OnRobot gripper to pick, scan, and place items.

## Directory Structure

```text
src/
  abc_bringup/              Launch-only package for starting cameras, robot bringup, and grouped demos.
  abc_interfaces/           Shared ROS messages, services, and actions.
    msg/                    Detection and purchase-item message definitions.
    srv/                    Speech, inventory, and user-request service definitions.
    action/                 Pick, scan-barcode, and place action definitions.
  abc_perception/           YOLO-based image perception node and model asset.
    abc_perception/         Python node implementation.
    models/                 Installed YOLO model file.
    launch/                 Camera plus perception launch file.
  abc_speech/               Speech-to-text and text-to-speech nodes.
    abc_speech/             Python node implementations.
    launch/                 Speech launch file.
  vlm_select/               VLM/order interpretation and experimental calibration code.
    vlm_select/             VLM, YOLO helper, and calibration/tutorial scripts.
    launch/                 VLM launch file.
  abc_manipulation/         Task planning, picking, barcode scanning, placing, camera helpers, and gripper helpers.
    abc_manipulation/       Python action servers/clients and hardware utilities.
    config/                 MoveIt Python configuration.
    launch/                 Manipulation launch file.
  dsr_moveit_config_m0609/  MoveIt2 configuration for the Doosan M0609 robot.
    config/                 URDF/Xacro, SRDF, planning, kinematics, controller, and sensor config.
    launch/                 MoveIt demo/start launch files.
```

## Module Responsibilities

- `abc_bringup`: Composes existing launch files. `demo.launch.py` starts RealSense and Doosan RViz bringup; `test.launch.py` starts speech and VLM flows.
- `abc_interfaces`: Owns cross-package contracts. Change these definitions carefully because generated message/service/action types are imported by other packages.
- `abc_perception`: Runs `yolo_node`, subscribes to RealSense color images, publishes `/detections`, and serves `/vlm_request` using `abc_interfaces/UserRequest`.
- `abc_speech`: Runs `stt_node` and `tts_node`. STT listens for `/stt_start` and sends recognized text to `/stt_results`; TTS exposes `/vlm_to_tts` and can pause/resume STT through `/stt_start`.
- `vlm_select`: Interprets speech/user requests with a VLM and external stock API, then routes requests either to TTS for insufficient stock or to perception for available items. `Calibration_Tutorial/` contains development/calibration scripts rather than core runtime code.
- `abc_manipulation`: Runs the manipulation pipeline. `task_planner` consumes detections and calls the `pick_item`, `scan_barcode`, and `place_item` action servers. `onrobot.py` wraps gripper access, and `realsense.py` contains image/depth helper logic.
- `dsr_moveit_config_m0609`: Provides robot model and MoveIt configuration consumed by manipulation and robot bringup. Treat this as robot configuration data, not application logic.

## Important Data Flows

### Runtime Request Flow

```text
User speech
  -> abc_speech/stt_node
  -> /stt_results service
  -> vlm_select/vlm_node
  -> stock check at http://127.0.0.1:8000
  -> if insufficient: /vlm_to_tts service -> abc_speech/tts_node
  -> if sufficient: /vlm_request service -> abc_perception/yolo_node
```

### Perception Flow

```text
RealSense color image: /camera/camera/color/image_raw
  -> abc_perception/yolo_node
  -> detection boxes: /detections
  -> abc_manipulation/task_planner
```

### Manipulation Flow

```text
/detections
  -> abc_manipulation/task_planner
  -> PickItem action: pick_item
  -> ScanBarcode action: scan_barcode
  -> PlaceItem action: place_item
  -> MoveIt/Doosan robot + OnRobot gripper hardware
```

### Shared Interface Types

- Messages: `DetectedObject`, `DetectionArray`, `PurchaseItem`
- Services: `Stt`, `SttStart`, `UserRequest`, `CheckInventory`
- Actions: `PickItem`, `ScanBarcode`, `PlaceItem`

## Build Dependencies

- Build system:
  - Python packages use `ament_python`: `abc_bringup`, `abc_perception`, `abc_speech`, `vlm_select`, `abc_manipulation`.
  - Interface/config packages use `ament_cmake`: `abc_interfaces`, `dsr_moveit_config_m0609`.
- Core ROS dependencies:
  - `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`
  - `abc_interfaces` for speech, perception, and manipulation contracts
  - `rosidl_default_generators` and `rosidl_default_runtime` for interface generation
- Perception/VLM dependencies:
  - `cv_bridge`, OpenCV, YOLO/Ultralytics model assets, `requests`, `google-generativeai`
- Manipulation dependencies:
  - MoveIt/MoveItPy, `moveit_msgs`, Doosan packages such as `dsr_bringup2`, `dsr_msgs2`, robot state publishers, `tf2_ros`, and `xacro`
  - Python numeric libraries: NumPy and SciPy
- Hardware/runtime dependencies:
  - RealSense ROS driver package `realsense2_camera`
  - Doosan M0609 controller/network availability
  - OnRobot gripper connectivity

Build from the workspace root with the normal ROS 2 flow:

```bash
colcon build
source install/setup.bash
```

## Coding Boundaries

- Keep interface changes in `abc_interfaces` deliberate and coordinated. Any `.msg`, `.srv`, or `.action` change can require updates in multiple packages.
- Keep launch composition in `abc_bringup`; package-specific launch files should stay with the package they start.
- Keep perception responsibilities in `abc_perception`: image subscription, model loading, detection publication, and perception request handling.
- Keep speech responsibilities in `abc_speech`: microphone/STT control and TTS playback. Do not put order logic there.
- Keep order/VLM/business logic in `vlm_select`: speech-result interpretation, inventory checks, and routing to TTS or perception.
- Keep robot motion and gripper control in `abc_manipulation`; avoid putting hardware commands in perception or VLM code.
- Keep robot model, planning, controller, and sensor configuration in `dsr_moveit_config_m0609`; application nodes should consume these settings rather than duplicating them.
- `vlm_select/Calibration_Tutorial/` appears to be experimental calibration code. Avoid depending on it from runtime launch paths unless it is promoted into the main package structure.
- Check launch files and `setup.py` entry points together when adding or renaming nodes. Some current `vlm_select` launch/entry-point names appear out of sync, so keep new changes explicit and verified.
