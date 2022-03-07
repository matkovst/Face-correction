set(CMAKE_DEBUG_POSTFIX d)

cmake_minimum_required(VERSION 3.10)
project( lib.facecorr )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_CONSTANT_MACROS")
if(MSVC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif(MSVC)

file(GLOB_RECURSE HEADERS "*.hpp")
file(GLOB_RECURSE SOURCES "*.cpp")

# OpenMP
find_package( OpenMP )
if(OpenMP_CXX_FOUND)
    add_compile_definitions(WITH_OPENMP)
endif()

add_library( ${PROJECT_NAME} ${HEADERS} ${SOURCES} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${PROJECT_NAME} PRIVATE OMP_THREADS=4)
endif()