cmake_minimum_required(VERSION 3.8)
project(abc_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    urdf
    meshes
    licenses
    config
    scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()