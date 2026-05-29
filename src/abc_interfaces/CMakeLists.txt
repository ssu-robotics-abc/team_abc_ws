cmake_minimum_required(VERSION 3.8)
project(abc_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PurchaseItem.msg"
  "msg/DetectedObject.msg"
  "msg/DetectionArray.msg"
  "action/PickItem.action"
  "action/ScanBarcode.action"
  "action/PlaceItem.action"
  "srv/CheckInventory.srv"
  "srv/Tts.srv"
  "srv/Stt.srv"
  "srv/SttStart.srv"
  "srv/UserRequest.srv"
  DEPENDENCIES 
    std_msgs 
    action_msgs
    geometry_msgs
    sensor_msgs
    builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()