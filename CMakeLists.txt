cmake_minimum_required(VERSION 3.20)
project(timesift LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The binning and the reduction are one implementation, compiled into the R package by R itself and
# into the Python extension here. Both read src/ directly, which is why this file sits beside
# DESCRIPTION rather than under python/: a Python source distribution cannot reach above its own
# project directory, so the project directory is the repository.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
# The penalised fit runs the folds of a cross-validation on threads through `std::thread`, which
# on a toolchain whose standard library reaches pthreads has to be linked against.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

nanobind_add_module(_core NB_STATIC
  src/ts_core.cpp
  src/ts_calendar.cpp
  src/ts_penalised.cpp
  python/ts_py.cpp)

target_include_directories(_core PRIVATE src)
target_link_libraries(_core PRIVATE Threads::Threads)
install(TARGETS _core LIBRARY DESTINATION timesift)
