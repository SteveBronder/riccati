# Install script for directory: /mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/AdolcForward"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/AlignedVector3"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/ArpackSupport"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/AutoDiff"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/BVH"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/EulerAngles"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/FFT"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/IterativeSolvers"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/KroneckerProduct"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/LevenbergMarquardt"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/MatrixFunctions"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/MoreVectorization"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/MPRealSupport"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/NonLinearOptimization"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/NumericalDiff"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/OpenGLSupport"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/Polynomials"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/Skyline"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/SparseExtra"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/SpecialFunctions"
    "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-src/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/mnt/home/sbronder/opensource/riccati/build/_deps/eigen-build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

