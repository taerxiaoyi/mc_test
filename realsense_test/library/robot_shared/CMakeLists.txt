# ========== robot_shared utility (header-only) ==========
add_library(robot_shared_utility INTERFACE)
target_include_directories(robot_shared_utility INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ========== Eigen (header-only) ==========
add_library(robot_shared_eigen INTERFACE)
target_include_directories(robot_shared_eigen INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/Eigen
)

# ========== Mujoco ==========
add_library(robot_shared_mujoco INTERFACE)
target_include_directories(robot_shared_mujoco INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/mujoco/include
)
target_link_directories(robot_shared_mujoco INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/mujoco/lib/${CURRENT_PLATFORM}
)
target_link_libraries(robot_shared_mujoco INTERFACE mujoco)

# ========== CycloneDDS(Unitree SDK2) ==========
add_library(robot_shared_cyclonedds INTERFACE)
target_include_directories(robot_shared_cyclonedds INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/unitree_sdk2/thirdparty/include
    ${CMAKE_CURRENT_SOURCE_DIR}/library/unitree_sdk2/thirdparty/include/ddscxx
)
target_link_directories(robot_shared_cyclonedds INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/unitree_sdk2/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}
)
target_link_libraries(robot_shared_cyclonedds INTERFACE ddscxx ddsc)

# ========== Unitree SDK2 ==========
add_library(robot_shared_unitree_sdk2 INTERFACE)
target_include_directories(robot_shared_unitree_sdk2 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/unitree_sdk2/include
)
target_link_directories(robot_shared_unitree_sdk2 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/unitree_sdk2/lib/${CMAKE_SYSTEM_PROCESSOR}
)
target_link_libraries(robot_shared_unitree_sdk2 INTERFACE unitree_sdk2)

# ========== Westlake Cyclone SDK ==========
add_library(robot_shared_westlake_cyclonedds_sdk INTERFACE)
target_include_directories(robot_shared_westlake_cyclonedds_sdk INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/library/wlrobot_cyclone_sdk/include
)

