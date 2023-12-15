cmake_minimum_required(VERSION 3.20)
project(tlib)


MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children ${curdir}/*) # This was changed
    SET(dirlist "")
    FOREACH (child ${children})
        IF (IS_DIRECTORY ${child}) # This was changed
            LIST(APPEND dirlist ${child})
        ENDIF ()
    ENDFOREACH ()
    SET(${result} ${dirlist})
ENDMACRO()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(OPEN_BLAS 1)

add_compile_options(-fPIC -Wall -Wsuggest-override)
add_compile_options(-O3)
add_compile_definitions(_TLIB_CONFIG_BOUND_CHECK=false)
add_compile_definitions(_TLIB_CONFIG_SHAPE_CHECK=true)
add_compile_definitions(_TLIB_OPTIMIZE_CPU_BLAS=${OPEN_BLAS})
add_compile_definitions(_TLIB_OPTIMIZE_CPU_BLAS_CONTIGUOUS=1<<10)

# HEADERS

SUBDIRLIST(INCLUDE_DIRECTORIES includes)

MESSAGE(STATUS "Include Directories: ${INCLUDE_DIRECTORIES}")

# LIBRARIES

set(LIBRARY_DIRECTORY lib)
set(LIBRARIES png z)

if (${OPEN_BLAS})
    set(LIBRARIES ${LIBRARIES} openblas)
endif ()

# TENSORLIB SOURCE

set(TLIB_SOURCE ./tlib)

MACRO(add_target_executable target source)
    set(SOURCE_${target} ${source})
    file(GLOB_RECURSE SOURCE_${source} ${source}/*.cc)

    MESSAGE(STATUS "${target} using files: ${SOURCE_${source}}")

    add_executable(${target} ${SOURCE_${source}})
    target_include_directories(${target} PRIVATE ${INCLUDE_DIRECTORIES})
    target_include_directories(${target} PUBLIC ${TLIB_SOURCE})

    target_link_directories(${target} PRIVATE ${LIBRARY_DIRECTORY})
    target_link_libraries(${target} ${LIBRARIES})

    set_target_properties(
            ${target}
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "./"
    )
ENDMACRO()

add_target_executable(test ./src/test)

add_target_executable(classify ./src/classify)

add_target_executable(generate ./src/generate)

add_target_executable(optimdemo ./src/optimdemo)
