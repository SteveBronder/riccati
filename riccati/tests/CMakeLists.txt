# Define a function to simplify adding grouped tests
function(add_gtest_grouped_test test_directory)
  # Create a test target name based on the directory
  file(RELATIVE_PATH TEST_DIRECTORY_RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${test_directory})
  string(REPLACE "/" "_" TEST_TARGET_NAME ${TEST_DIRECTORY_RELATIVE})

  # Collect all .cpp files in the directory
  file(GLOB TEST_SOURCES "${TEST_DIRECTORY_RELATIVE}/*.cpp")
  if(TEST_SOURCES)
  # Add an executable for the grouped tests
  add_executable(${TEST_TARGET_NAME} ${TEST_SOURCES})

  # Add include directories, link libraries, and register tests as before
  target_include_directories(${TEST_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(${TEST_TARGET_NAME} gtest_main riccati)
  add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})
  endif()
endfunction()
message(STATUS "Got here")
# List all subdirectories in the tests folder that contain test files
file(GLOB_RECURSE TEST_DIRECTORIES LIST_DIRECTORIES true cpp/*)

# Create a grouped test executable for each directory
foreach(TEST_DIRECTORY ${TEST_DIRECTORIES})
  if(IS_DIRECTORY ${TEST_DIRECTORY})
    message(STATUS "Adding test dir: ${TEST_DIRECTORY}")
    add_gtest_grouped_test(${TEST_DIRECTORY})
  endif()
endforeach()
