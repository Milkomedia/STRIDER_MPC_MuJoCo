cmake_minimum_required(VERSION 3.16)
project(strider LANGUAGES CXX)

# -------------------------
# C++ standard
# -------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------
# Options / paths
# -------------------------
option(STRIDER_BUILD_APP "Build strider executable" ON)
set(STRIDER_PY_DIR "${CMAKE_SOURCE_DIR}/resources" CACHE PATH "Python root dir (contains mpc_py/)")
set(STRIDER_ACADOS_GEN_DIR "${CMAKE_SOURCE_DIR}/resources/mpc_py/generated" CACHE PATH "acados generated dir")

# -------------------------
# Dependencies
# -------------------------
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Prefer Conda Python if available
if(DEFINED ENV{CONDA_PREFIX})
  set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Conda env root" FORCE)
  set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python3" CACHE FILEPATH "Conda python" FORCE)
endif()
find_package(Python3 COMPONENTS Interpreter Development.Embed REQUIRED)

# -------------------------
# pybind11
# -------------------------
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE _PYBIND11_CMAKEDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT _PYBIND11_CMAKEDIR)
  message(FATAL_ERROR "pybind11 not found in this Python env (python -m pybind11 --cmakedir failed)")
endif()

find_package(pybind11 CONFIG REQUIRED
  PATHS "${_PYBIND11_CMAKEDIR}"
  NO_DEFAULT_PATH
)

# -------------------------
# MuJoCo
# -------------------------

# (A) Locate Python package dir (optional; used as fallback)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)"
  OUTPUT_VARIABLE MUJOCO_PY_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(MUJOCO_PY_DIR)
  message(STATUS "MuJoCo python dir: ${MUJOCO_PY_DIR}")
else()
  message(STATUS "MuJoCo python dir: (not found via import mujoco) - will rely on conda/system")
endif()

# (B) Include dir: prefer conda, then python layouts
set(_MUJOCO_INCLUDE_HINTS "")
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _MUJOCO_INCLUDE_HINTS "$ENV{CONDA_PREFIX}/include")
endif()
if(MUJOCO_PY_DIR)
  list(APPEND _MUJOCO_INCLUDE_HINTS
    "${MUJOCO_PY_DIR}/include"
    "${MUJOCO_PY_DIR}/../include"
    "${MUJOCO_PY_DIR}/mujoco.framework/Headers"
  )
endif()

find_path(MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  HINTS ${_MUJOCO_INCLUDE_HINTS}
)

# (C) Shared lib: prefer conda, else python wheel layout (your original behavior)
set(MUJOCO_LIB "")

if(DEFINED ENV{CONDA_PREFIX})
  find_library(_MUJOCO_LIB_CONDA
    NAMES mujoco libmujoco
    HINTS "$ENV{CONDA_PREFIX}/lib"
    NO_DEFAULT_PATH
  )
  if(_MUJOCO_LIB_CONDA)
    set(MUJOCO_LIB "${_MUJOCO_LIB_CONDA}")
  endif()
endif()

if(NOT MUJOCO_LIB AND MUJOCO_PY_DIR)
  # Original-style fallback (python package directories)
  if(APPLE)
    file(GLOB _MUJOCO_LIB_CANDIDATES
      "${MUJOCO_PY_DIR}/libmujoco*.dylib"
      "${MUJOCO_PY_DIR}/lib/libmujoco*.dylib"
      "${MUJOCO_PY_DIR}/mujoco.libs/libmujoco*.dylib"
      "${MUJOCO_PY_DIR}/_lib/libmujoco*.dylib"
      "${MUJOCO_PY_DIR}/mujoco.framework/Versions/*/libmujoco*.dylib"
    )
  else()
    file(GLOB _MUJOCO_LIB_CANDIDATES
      "${MUJOCO_PY_DIR}/libmujoco*.so*"
      "${MUJOCO_PY_DIR}/lib/libmujoco*.so*"
      "${MUJOCO_PY_DIR}/mujoco.libs/libmujoco*.so*"
      "${MUJOCO_PY_DIR}/_lib/libmujoco*.so*"
    )
  endif()

  list(LENGTH _MUJOCO_LIB_CANDIDATES _MJ_N)
  if(_MJ_N GREATER 0)
    list(GET _MUJOCO_LIB_CANDIDATES 0 MUJOCO_LIB)
  endif()
endif()

# (D) Validate
if(NOT MUJOCO_INCLUDE_DIR)
  message(FATAL_ERROR
    "Could not find MuJoCo header mujoco/mujoco.h. "
    "Tried CONDA_PREFIX/include and Python package layouts. "
    "CONDA_PREFIX=$ENV{CONDA_PREFIX}, MUJOCO_PY_DIR=${MUJOCO_PY_DIR}"
  )
endif()

if(NOT MUJOCO_LIB)
  message(FATAL_ERROR
    "Could not find MuJoCo shared library (libmujoco). "
    "Tried CONDA_PREFIX/lib and Python package layouts. "
    "CONDA_PREFIX=$ENV{CONDA_PREFIX}, MUJOCO_PY_DIR=${MUJOCO_PY_DIR}"
  )
endif()

message(STATUS "MuJoCo include dir: ${MUJOCO_INCLUDE_DIR}")
message(STATUS "MuJoCo lib: ${MUJOCO_LIB}")

add_library(mujoco SHARED IMPORTED GLOBAL)
set_target_properties(mujoco PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}"
)

# -------------------------
# GLFW (CONFIG first, fallback pkg-config)
# -------------------------
find_package(glfw3 CONFIG QUIET)

if(glfw3_FOUND)
  # Different distros/exporters can name the target differently
  if(TARGET glfw)
    set(STRIDER_GLFW_TARGET glfw)
  elseif(TARGET glfw::glfw)
    set(STRIDER_GLFW_TARGET glfw::glfw)
  elseif(TARGET glfw3)
    set(STRIDER_GLFW_TARGET glfw3)
  elseif(TARGET glfw3::glfw)
    set(STRIDER_GLFW_TARGET glfw3::glfw)
  else()
    message(FATAL_ERROR "glfw3_FOUND but no known GLFW target (glfw / glfw::glfw / glfw3 / glfw3::glfw).")
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW3 REQUIRED glfw3)

  add_library(glfw3_pc INTERFACE)
  target_include_directories(glfw3_pc INTERFACE ${GLFW3_INCLUDE_DIRS})
  target_link_directories(glfw3_pc INTERFACE ${GLFW3_LIBRARY_DIRS})
  target_link_libraries(glfw3_pc INTERFACE ${GLFW3_LIBRARIES})

  set(STRIDER_GLFW_TARGET glfw3_pc)
endif()

# ============================================================
# modules/geometry_ctrl
# ============================================================
add_library(geometry_ctrl STATIC
  resources/geometry_ctrl/src/fdcl_control.cpp
  resources/geometry_ctrl/src/fdcl_matrix_utils.cpp
)

target_include_directories(geometry_ctrl PUBLIC
  ${CMAKE_SOURCE_DIR}/resources/geometry_ctrl/include
  ${CMAKE_SOURCE_DIR}/include
  ${MUJOCO_INCLUDE_DIR}  # for <mujoco/mujoco.h>
)

target_link_libraries(geometry_ctrl PUBLIC
  Eigen3::Eigen
)

# ============================================================
# strider core
# ============================================================
add_library(strider_core STATIC
  src/mj_viewer_helper.cpp
  src/mpc_wrapper.cpp
)

target_include_directories(strider_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(strider_core PUBLIC
  mujoco
  ${STRIDER_GLFW_TARGET}
  Threads::Threads
  Eigen3::Eigen
  pybind11::embed
  Python3::Python
)

# On Linux, dl is commonly needed for Python embed / shared loading
if(UNIX AND NOT APPLE)
  target_link_libraries(strider_core PUBLIC dl)
endif()

# ============================================================
# App
# ============================================================
if(STRIDER_BUILD_APP)
  add_executable(strider apps/main.cpp)
  target_link_libraries(strider PRIVATE
    strider_core
    geometry_ctrl
  )

  # Runtime search paths
  if(APPLE)
    set(_ORIGIN "@loader_path")
  else()
    set(_ORIGIN "$ORIGIN")
  endif()

  get_filename_component(MJ_RPATH_DIR "${MUJOCO_LIB}" DIRECTORY)

  set_target_properties(strider PROPERTIES
    BUILD_RPATH   "${_ORIGIN};${MJ_RPATH_DIR};${STRIDER_ACADOS_GEN_DIR}"
    INSTALL_RPATH "${_ORIGIN};${MJ_RPATH_DIR};${STRIDER_ACADOS_GEN_DIR}"
  )

  # macOS-only: optional framework-style symlink (safe to omit on Linux)
  if(APPLE)
    get_filename_component(MUJOCO_BASENAME "${MUJOCO_LIB}" NAME)
    add_custom_command(TARGET strider POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
              "$<TARGET_FILE_DIR:strider>/mujoco.framework/Versions/A"
      COMMAND ${CMAKE_COMMAND} -E rm -f
              "$<TARGET_FILE_DIR:strider>/mujoco.framework/Versions/A/${MUJOCO_BASENAME}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "${MUJOCO_LIB}"
              "$<TARGET_FILE_DIR:strider>/mujoco.framework/Versions/A/${MUJOCO_BASENAME}"
      VERBATIM
    )
  endif()

  # Convenience target: run with proper env vars
  if(APPLE)
    set(_LDVAR "DYLD_LIBRARY_PATH")
  else()
    set(_LDVAR "LD_LIBRARY_PATH")
  endif()

  add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${STRIDER_PY_DIR}"
            "${_LDVAR}=${MJ_RPATH_DIR}:${STRIDER_ACADOS_GEN_DIR}:$ENV{${_LDVAR}}"
            "$<TARGET_FILE:strider>"
    DEPENDS strider
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()