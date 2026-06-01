## 1. abc_interfaces

> ROS2 패키지 간 통신에 사용할 커스텀 메시지, 서비스, 액션을 정의


## 2. abc_bringup

> 전체 시스템을 실행하기 위한 launch/config 전용 패키지

### Quick Start

```bash
ros2_ws
ws_moveit
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real model:=m0609 host:=192.168.1.100
ros2 launch realsense2_camera rs_align_depth_launch.py depth_module.depth_profile:=640x480x30 rgb_camera.color_profile:=640x480x30 initial_reset:=true align_depth.enable:=true
```
대신

```bash
ros2_ws
ws_moveit
ros2 launch abc_bringup demo.launch.py
```

로 실행 가능  
  



필요 역할:  
- 전체 노드 실행 관리  
- RealSense 실행  
- YOLO perception 노드 실행  
- task planner 실행  
- manipulation 노드 실행  
- logger 실행  
- 시연용 launch 파일 관리  
- 파라미터 yaml 관리  

## 3. abc_speech

> 음성 인식 담당 패키지

주요 역할:
- 마이크 입력 수집하고 STT를 수행하여
- 음성 명령을 텍스트 주문으로 변환
- 주문 결과를 특정 데이터 형태로 publish
- 상품 재고 부족 시 재주문 요청


## 4. abc_perception

> RealSense와 YOLO 등을 이용한 비전 인식 패키지

## 5. abc_calibration

> 카메라 좌표계를 로봇 베이스 좌표계로 변환하는 패키지

필요 역할:
- Eye-in-hand camera transform 관리
- Hand-Eye calibration 결과 적용
- camera frame → robot base frame 좌표 변환
- TF publish
- 목표 물체 pose 변환


## 6. abc_task_planner

> 음성 명령과 비전 인식 결과를 바탕으로 로봇 행동을 결정하는 패키지


## 7. abc_manipulation

> 로봇팔 제어와 그리퍼 제어를 담당하는 패키지

필요 역할:
- 목표 pose 수신
- MoveIt2 기반 경로 계획
- 두산 M0609 로봇 이동 명령
- OnRobot RG2 그리퍼 open/close 제어
- 파지 성공 여부 판단
- 충돌 및 비상 정지 처리


## 8. abc_logger

> 실험 결과와 시스템 상태를 기록하는 패키지

필요 역할:
- STT 결과 기록
- VLA 추론 시간 기록
- 물체 탐지 결과 기록
- 파지 성공 여부 기록
- 작업 시간 기록
- CSV 또는 SQLite 저장
