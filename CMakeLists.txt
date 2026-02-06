cmake_minimum_required(VERSION 3.10)
project(casadi_cpp_tutorial)

set(CMAKE_CXX_STANDARD 14)

# 1. CasADi가 설치된 경로 설정 (프로젝트 루트의 external/casadi 기준)
set(CASADI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/casadi-main/install")

# 2. 헤더 파일 경로 추가
include_directories(${CASADI_ROOT}/include)

# 3. 라이브러리 경로 설정 및 찾기
link_directories(${CASADI_ROOT}/lib)

# 4. 실행 파일 생성
add_executable(cruise_control_mpc src/main.cpp)

# 5. CasADi 라이브러리 링크
target_link_libraries(cruise_control_mpc casadi)

# 만약 런타임에 .so 파일을 못 찾는다면 아래 설정 추가 (Linux 전용)
set_target_properties(cruise_control_mpc PROPERTIES INSTALL_RPATH "${CASADI_ROOT}/lib")
set_target_properties(cruise_control_mpc PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
