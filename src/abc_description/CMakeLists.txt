cmake_minimum_required(VERSION 3.8)
project(abc_description)

find_package(ament_cmake REQUIRED)

find_program(XACRO_EXECUTABLE xacro REQUIRED)


install(DIRECTORY
  urdf
  meshes
  config
  DESTINATION share/${PROJECT_NAME}
)



ament_package()
