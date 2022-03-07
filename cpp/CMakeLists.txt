cmake_minimum_required( VERSION 3.10 )
project( demo )

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

# Include lib.facecorr
add_subdirectory(lib.facecorr)

# Include OpenCV
find_package( OpenCV REQUIRED COMPONENTS core imgproc videoio highgui )

add_executable( ${PROJECT_NAME} demo.cpp )

set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 11)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories( ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( ${PROJECT_NAME} lib.facecorr ${OpenCV_LIBS} )

# set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
# set(INSTALL_BIN_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Path where exe and dll will be installed")
# install(
#     TARGETS ${PROJECT_NAME}
#     CONFIGURATIONS Release
#     RUNTIME DESTINATION bin
# )