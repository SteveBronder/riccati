# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/home/sbronder/opensource/riccati

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/home/sbronder/opensource/riccati/build

# Include any dependencies generated for this target.
include CMakeFiles/riccati_tests_cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/riccati_tests_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/riccati_tests_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/riccati_tests_cpp.dir/flags.make

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o: CMakeFiles/riccati_tests_cpp.dir/flags.make
CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o: ../riccati/tests/cpp/chebyshev_test.cpp
CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o: CMakeFiles/riccati_tests_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/home/sbronder/opensource/riccati/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o -MF CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o.d -o CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o -c /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/chebyshev_test.cpp

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.i"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/chebyshev_test.cpp > CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.i

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.s"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/chebyshev_test.cpp -o CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.s

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o: CMakeFiles/riccati_tests_cpp.dir/flags.make
CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o: ../riccati/tests/cpp/step_test.cpp
CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o: CMakeFiles/riccati_tests_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/home/sbronder/opensource/riccati/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o -MF CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o.d -o CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o -c /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/step_test.cpp

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.i"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/step_test.cpp > CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.i

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.s"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/step_test.cpp -o CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.s

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o: CMakeFiles/riccati_tests_cpp.dir/flags.make
CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o: ../riccati/tests/cpp/stepsize_test.cpp
CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o: CMakeFiles/riccati_tests_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/home/sbronder/opensource/riccati/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o -MF CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o.d -o CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o -c /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/stepsize_test.cpp

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.i"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/stepsize_test.cpp > CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.i

CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.s"
	/usr/lib64/ccache/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/home/sbronder/opensource/riccati/riccati/tests/cpp/stepsize_test.cpp -o CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.s

# Object files for target riccati_tests_cpp
riccati_tests_cpp_OBJECTS = \
"CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o" \
"CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o" \
"CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o"

# External object files for target riccati_tests_cpp
riccati_tests_cpp_EXTERNAL_OBJECTS =

riccati_tests_cpp: CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/chebyshev_test.cpp.o
riccati_tests_cpp: CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/step_test.cpp.o
riccati_tests_cpp: CMakeFiles/riccati_tests_cpp.dir/riccati/tests/cpp/stepsize_test.cpp.o
riccati_tests_cpp: CMakeFiles/riccati_tests_cpp.dir/build.make
riccati_tests_cpp: lib/libgtest_main.a
riccati_tests_cpp: lib/libgtest.a
riccati_tests_cpp: CMakeFiles/riccati_tests_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/home/sbronder/opensource/riccati/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable riccati_tests_cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/riccati_tests_cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/riccati_tests_cpp.dir/build: riccati_tests_cpp
.PHONY : CMakeFiles/riccati_tests_cpp.dir/build

CMakeFiles/riccati_tests_cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/riccati_tests_cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/riccati_tests_cpp.dir/clean

CMakeFiles/riccati_tests_cpp.dir/depend:
	cd /mnt/home/sbronder/opensource/riccati/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/home/sbronder/opensource/riccati /mnt/home/sbronder/opensource/riccati /mnt/home/sbronder/opensource/riccati/build /mnt/home/sbronder/opensource/riccati/build /mnt/home/sbronder/opensource/riccati/build/CMakeFiles/riccati_tests_cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/riccati_tests_cpp.dir/depend
