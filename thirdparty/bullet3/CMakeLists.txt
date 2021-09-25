include(external)

download_external(
    bullet3
    GIT_REPOSITORY https://github.com/bulletphysics/bullet3.git
    # GIT_TAG        3.17
)

external(
  bullet3
    INSTALL OFF
    TARGETS
      # Bullet3SoftBody
      # Bullet3Common
      BulletDynamics
      BulletCollision
      LinearMath
    PROJECT_OPTIONS
      -DBUILD_CPU_DEMOS=OFF
      -DUSE_GLUT=OFF
      -DBUILD_EXTRAS=OFF
      -DBUILD_UNIT_TESTS=OFF
      -DBUILD_BULLET2_DEMOS=OFF
      -DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON
)

target_include_directories(bullet3_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/bullet3/src/src)
# target_link_libraries(bullet3_external INTERFACE BulletSoftBody BulletDynamics BulletCollision LinearMath)
# target_link_libraries(bullet3_external INTERFACE Bullet3Dynamics)