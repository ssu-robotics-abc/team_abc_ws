# Team ABC Robotics Workspace

> 2026-1 COOP Robotics Workspace

음성 명령을 기반으로 상품을 인식하고, 로봇팔이 상품을 집어 바코드를 스캔한 뒤 지정 위치에 배치하는 ROS 2 기반 로보틱스 워크스페이스입니다.

이 프로젝트는 음성 인식, VLM 기반 주문 해석, 객체 인식, 로봇팔 제어를 하나의 파이프라인으로 통합하는 것을 목표로 합니다.

<img width="2560" height="1600" alt="스크린샷(3)" src="https://github.com/user-attachments/assets/52d2afc1-56fc-44d2-ad84-f2c9b33c7986" />

---

## Package Responsibilities

| Package                   | Responsibility                                     |
| ------------------------- | -------------------------------------------------- |
| `abc_bringup`             | 전체 시스템 실행을 위한 launch 관리                            |
| `abc_interfaces`          | 공통 message, service, action 정의                     |
| `abc_speech`              | 사용자 음성 인식 STT와 음성 응답 TTS 처리                        |
| `vlm_select`              | 사용자 주문 해석, 재고 확인, VLM 기반 요청 처리                     |
| `abc_perception`          | RealSense 이미지 입력, YOLO 객체 탐지, 탐지 결과 발행             |
| `abc_manipulation`        | 로봇팔 task planning, pick, barcode scan, place 동작 수행 |
| `dsr_moveit_config_m0609` | Doosan M0609 로봇 모델, MoveIt2, controller 설정         |

---

## Requirements

### System

* Ubuntu
* ROS 2
* Python 3.10
* `colcon`
* `rosdep`
* MoveIt2 / MoveItPy
* Intel RealSense ROS driver
* Doosan Robotics ROS 2 packages
* OnRobot RG2 gripper connection

### Python Dependencies

주요 Python 의존성은 다음과 같습니다.

```text
numpy
opencv-python
pandas
pyyaml
requests
torch
torchvision
ultralytics
pyrealsense2
scikit-learn
scipy
matplotlib
pymodbus
pyserial
gtts
speechrecognition
pygame
```

정확한 의존성은 프로젝트 설정 파일과 각 패키지의 `package.xml`을 함께 확인하세요.

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ssu-robotics-abc/team_abc_ws.git
cd team_abc_ws
```

### 2. Source ROS 2

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

예시:

```bash
source /opt/ros/humble/setup.bash
```

### 3. Install ROS Dependencies

```bash
rosdep update
rosdep install --from-paths src -y --ignore-src
```

### 4. Build Workspace

```bash
colcon build --symlink-install
source install/setup.bash
```

새 터미널을 열었다면 다음 명령을 다시 실행해야 합니다.

```bash
source install/setup.bash
```

---

## Quick Start

### Integrated Demo

전체 데모 실행은 `abc_bringup`의 launch 파일을 사용합니다.

```bash
ros2 launch abc_bringup abc.launch.py
```

현재 TTS 노드는 별도 터미널에서 실행합니다.

```bash
ros2 run abc_speech tts_node
```

`abc.launch.py`는 RealSense, perception, manipulation, robot bringup 등 데모에 필요한 핵심 노드를 실행하기 위한 진입점입니다.

---

## Run Individual Nodes

필요한 경우 각 노드를 개별적으로 실행할 수 있습니다.

### Speech

```bash
ros2 run abc_speech stt_node
ros2 run abc_speech tts_node
```

### VLM

```bash
ros2 run vlm_select vlm_node
```

### Perception

```bash
ros2 run abc_perception yolo_node
```

### Manipulation

```bash
ros2 run abc_manipulation task_planner
ros2 run abc_manipulation pick_item
ros2 run abc_manipulation scan_barcode
ros2 run abc_manipulation place_item
```

---

## Troubleshooting

### `package not found` 오류가 발생하는 경우

빌드 후 workspace setup 파일을 source 했는지 확인합니다.

```bash
source install/setup.bash
```

새 터미널을 열었다면 위 명령을 다시 실행해야 합니다.

### RealSense image topic이 보이지 않는 경우

```bash
ros2 topic list | grep camera
```

RealSense launch가 정상 실행되었는지 확인합니다.

### Detection 결과가 나오지 않는 경우

```bash
ros2 topic echo /detections
```

다음 항목을 확인합니다.

* RealSense color image topic 입력 여부
* YOLO model 파일 설치 여부
* `abc_perception` 빌드 여부
* camera topic name 일치 여부
