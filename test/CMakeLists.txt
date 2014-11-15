add_executable(run_tests.bin EXCLUDE_FROM_ALL
  asymmetric_laplace.cpp
  asymmetric_normal.cpp
  discrete.cpp
  laplace.cpp
  log_number.cpp
  mixture.cpp
  normal.cpp
)

target_link_libraries(run_tests.bin gtest gtest_main
)

add_custom_target(test COMMAND run_tests.bin
                       DEPENDS run_tests.bin)
