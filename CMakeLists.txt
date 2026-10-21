cmake_minimum_required(VERSION 3.25)

project(histocache VERSION 0.1.0 LANGUAGES CXX CUDA)

option(HISTOCACHE_BUILD_TESTS "Build the standalone GPU validation executable" ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

find_package(CCCL CONFIG REQUIRED COMPONENTS CUB)

add_library(histocache INTERFACE)
add_library(histocache::histocache ALIAS histocache)

# Histocache is an out-of-tree header overlay. The overlay precedes the selected
# CCCL installation, while all unchanged CUB/libcu++ headers continue to come
# from CCCL::CUB.
target_include_directories(histocache BEFORE INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/overlay>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(histocache INTERFACE CCCL::CUB)
target_compile_features(histocache INTERFACE cxx_std_17)

install(TARGETS histocache EXPORT histocacheTargets)
install(DIRECTORY overlay/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT histocacheTargets
  FILE histocacheTargets.cmake
  NAMESPACE histocache::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/histocache)
configure_package_config_file(
  cmake/histocacheConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/histocacheConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/histocache)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/histocacheConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/histocacheConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/histocacheConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/histocache)

if(HISTOCACHE_BUILD_TESTS)
  enable_testing()
  add_executable(histocache.smoke tests/device_histogram_smoke.cu)
  target_link_libraries(histocache.smoke PRIVATE histocache::histocache)
  set_target_properties(histocache.smoke PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES)
  add_test(NAME histocache.smoke COMMAND histocache.smoke)
endif()
