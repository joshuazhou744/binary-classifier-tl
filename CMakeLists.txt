cmake_minimum_required(VERSION 3.16)
project(binary_classifier_tl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  torchlite
  GIT_REPOSITORY https://github.com/joshuazhou744/torchlite-cpp
  GIT_TAG        main
)
FetchContent_MakeAvailable(torchlite)

add_executable(snore_classifier src/main.cpp)
target_link_libraries(snore_classifier PRIVATE torchlite)
