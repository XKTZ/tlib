cmake_minimum_required(VERSION 3.20)
project(tlib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(OPEN_BLAS "static")
set(TLIB_VISION "")

MACRO(subdir_list result curdir)
    FILE(GLOB children ${curdir}/*) # This was changed
    SET(dirlist "")
    FOREACH (child ${children})
        IF (IS_DIRECTORY ${child}) # This was changed
            LIST(APPEND dirlist ${child})
        ENDIF ()
    ENDFOREACH ()
    SET(${result} ${dirlist})
ENDMACRO()

MACRO(compile_definition_true_if_non_empty definition variable)
    string(STRIP "${variable}" variable_stripped)
    if (NOT ${variable_stripped} STREQUAL "")
        add_compile_definitions(${definition}=1)
    else ()
        add_compile_definitions(${definition}=0)
    endif ()
ENDMACRO()

subdir_list(INCLUDE_DIRECTORIES includes)

MESSAGE(STATUS "Include Directories: ${INCLUDE_DIRECTORIES}")

# ==================== LIBRARY ANALYSIS ====================
set(LIBRARY_DIRECTORY ./lib)
set(LIBRARIES "")

MACRO(tlib_import_library libname variable)
    if ("${variable}" STREQUAL "static")
        set(LIBRARIES "${LIBRARIES} ${libname}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    elseif ("${variable}" STREQUAL "shared")
        set(LIBRARIES "${LIBRARIES} ${libname}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif ()
ENDMACRO()

tlib_import_library(z "${TLIB_VISION}")
tlib_import_library(png "${TLIB_VISION}")
tlib_import_library(openblas "${OPEN_BLAS}")

string(STRIP ${LIBRARIES} LIBRARIES)

message(STATUS "LIBRARIES: ${LIBRARIES}")

# ================ TENSORLIB ==================

add_compile_options(-fPIC -Wall -Wsuggest-override)
add_compile_options(-g)
compile_definition_true_if_non_empty(_TLIB_CONFIG_VISION "${TLIB_VISION}")
add_compile_definitions(_TLIB_CONFIG_BOUND_CHECK=false)
add_compile_definitions(_TLIB_CONFIG_SHAPE_CHECK=true)
compile_definition_true_if_non_empty(_TLIB_OPTIMIZE_CPU_BLAS "${OPEN_BLAS}")
add_compile_definitions(_TLIB_OPTIMIZE_CPU_BLAS_CONTIGUOUS=1<<10)

set(TLIB_HEADER tlib/header)
set(TLIB_SOURCE tlib/src)

file(GLOB_RECURSE TLIB_SOURCE_FILE ${TLIB_SOURCE}/*.cc)
add_library(tensorlib STATIC ${TLIB_SOURCE_FILE})
target_include_directories(tensorlib PRIVATE ${TLIB_HEADER})
target_include_directories(tensorlib PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(tensorlib PUBLIC "${LIBRARY_DIRECTORY}")
target_link_libraries(tensorlib PRIVATE ${LIBRARIES})

# =============== DEMOS ===================

MACRO(add_target_executable target source)
    set(SOURCE_${target} ${source})
    file(GLOB_RECURSE SOURCE_${source} ${source}/*.cc)

    MESSAGE(STATUS "${target} using files: ${SOURCE_${source}}")

    add_executable(${target} ${SOURCE_${source}})
    target_include_directories(${target} PRIVATE ${TLIB_HEADER})
    target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES})
    target_link_libraries(${target} PRIVATE tensorlib)

    set_target_properties(
            ${target}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "./"
    )
ENDMACRO()

add_target_executable(test ./src/test)

add_target_executable(classify ./src/classify)

add_target_executable(generate ./src/generate)
