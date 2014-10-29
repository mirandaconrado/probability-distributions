add_executable(run_tests.bin EXCLUDE_FROM_ALL
  discrete.cpp
  laplace.cpp
  mixture.cpp
  normal.cpp
)

target_link_libraries(run_tests.bin gtest gtest_main
)

add_custom_target(test COMMAND run_tests.bin
                       DEPENDS run_tests.bin)
