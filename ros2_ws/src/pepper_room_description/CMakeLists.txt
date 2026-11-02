cmake_minimum_required(VERSION 3.5)
project(pepper_room_description)

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install directories.
#
# `models` must land directly under share/${PROJECT_NAME} (not in a subdirectory) because
# the environment hooks below put exactly that path on IGN_GAZEBO_RESOURCE_PATH, and
# Ignition resolves `model://room_photogrammetry` by looking for
# <resource path entry>/room_photogrammetry/model.config. Nesting it deeper breaks the
# lookup silently -- the world loads with the model simply absent.
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME})
install(DIRECTORY worlds DESTINATION share/${PROJECT_NAME})

# Put share/${PROJECT_NAME} on IGN_GAZEBO_RESOURCE_PATH. This is the whole mechanism by
# which `model://` URIs resolve; pepper_robot_description does the same thing for the
# robot itself. Without these two lines the worlds below reference models Ignition
# cannot find.
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in")
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.sh.in")

ament_package()
