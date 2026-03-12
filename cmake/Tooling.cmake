if(NOT PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  return()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SHAFFT_ENABLE_CLANG_TIDY "Run clang-tidy during builds" OFF)
if(SHAFFT_ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else()
    message(WARNING "SHAFFT_ENABLE_CLANG_TIDY=ON but clang-tidy not found")
  endif()
endif()

file(
  GLOB_RECURSE
  SHAFFT_FORMAT_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/*.c
  ${CMAKE_SOURCE_DIR}/src/*.cc
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/*.cxx
  ${CMAKE_SOURCE_DIR}/src/*.h
  ${CMAKE_SOURCE_DIR}/src/*.hh
  ${CMAKE_SOURCE_DIR}/src/*.hpp
  ${CMAKE_SOURCE_DIR}/src/*.hxx
  ${CMAKE_SOURCE_DIR}/include/*.h
  ${CMAKE_SOURCE_DIR}/include/*.hh
  ${CMAKE_SOURCE_DIR}/include/*.hpp
  ${CMAKE_SOURCE_DIR}/include/*.hxx
  ${CMAKE_SOURCE_DIR}/src/c/*.c
  ${CMAKE_SOURCE_DIR}/src/c/*.h
)

file(
  GLOB_RECURSE
  SHAFFT_TIDY_SRCS
  RELATIVE ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/*.c
  ${CMAKE_SOURCE_DIR}/src/*.cc
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/*.cxx
  ${CMAKE_SOURCE_DIR}/include/*.h
  ${CMAKE_SOURCE_DIR}/include/*.hh
  ${CMAKE_SOURCE_DIR}/include/*.hpp
  ${CMAKE_SOURCE_DIR}/include/*.hxx
  ${CMAKE_SOURCE_DIR}/src/c/*.c
  ${CMAKE_SOURCE_DIR}/src/c/*.h
)

# Exclude generated/vendor trees
list(FILTER SHAFFT_FORMAT_SRCS EXCLUDE REGEX "^build/")
list(FILTER SHAFFT_FORMAT_SRCS EXCLUDE REGEX "^\\.deps/")
list(FILTER SHAFFT_FORMAT_SRCS EXCLUDE REGEX "^_deps/")
list(FILTER SHAFFT_TIDY_SRCS   EXCLUDE REGEX "^build/")
list(FILTER SHAFFT_TIDY_SRCS   EXCLUDE REGEX "^\\.deps/")
list(FILTER SHAFFT_TIDY_SRCS   EXCLUDE REGEX "^_deps/")

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_EXE} -i ${SHAFFT_FORMAT_SRCS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Run clang-format on C/C++ sources"
  )
endif()

find_program(FPRETTIFY_EXE NAMES fprettify)
if(FPRETTIFY_EXE)
  add_custom_target(
    format_fortran
    COMMAND
      ${FPRETTIFY_EXE} --config ${CMAKE_SOURCE_DIR}/.fprettify.rc -r
      ${CMAKE_SOURCE_DIR}/src/f03 ${CMAKE_SOURCE_DIR}/examples ${CMAKE_SOURCE_DIR}/tests
      ${CMAKE_SOURCE_DIR}/validate
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Run fprettify on Fortran sources"
  )
endif()

if(CLANG_TIDY_EXE)
  add_custom_target(
    tidy
    COMMAND
      ${CLANG_TIDY_EXE}
      -p ${CMAKE_BINARY_DIR}
      --header-filter=${CMAKE_SOURCE_DIR}/include/.*
      ${SHAFFT_TIDY_SRCS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Run clang-tidy on C/C++ sources"
  )
endif()
