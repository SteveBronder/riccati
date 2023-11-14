# CMake generated Testfile for 
# Source directory: /mnt/home/sbronder/opensource/riccati
# Build directory: /mnt/home/sbronder/opensource/riccati/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[riccati_tests_cpp]=] "/mnt/home/sbronder/opensource/riccati/build/riccati_tests_cpp")
set_tests_properties([=[riccati_tests_cpp]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/home/sbronder/opensource/riccati/CMakeLists.txt;139;add_test;/mnt/home/sbronder/opensource/riccati/CMakeLists.txt;152;add_gtest_grouped_test;/mnt/home/sbronder/opensource/riccati/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("_deps/eigen-build")
