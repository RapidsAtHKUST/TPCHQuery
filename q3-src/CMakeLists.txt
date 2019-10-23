cmake_minimum_required(VERSION 3.6)
project(TPCH-Q3)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug") # enable assertions

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_options(-Wall -O3 -g -march=native)

option(ENABLE_GPU "enable GPU" ON)
option(ENABLE_LOG "enable log" ON)
if (ENABLE_LOG)
    add_definitions(-DUSE_LOG)
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

#add_executable(tpch-q3 tpch_q3_main.cpp util/log.cpp file_input_indexing.cpp)
#add_executable(tpch-q3-submit tpch_q3_main_submit.cpp util/log.cpp file_input_indexing.cpp)

find_package(CUDA)
if (CUDA_FOUND)
    message("--- CUDA Found, Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}/")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc) #for helper_cuda.h
    message(${CMAKE_CXX_COMPILER})
    set(CUDA_NVCC_FLAGS "-ccbin ${CMAKE_CXX_COMPILER}")

    if (USE_ARCH)
        message(STATUS "User has specified ARCH: ${USE_ARCH}")
        set(ARCH -arch=${USE_ARCH})
    else() # checking the default arch of the current platform
        set(OUTPUTFILE ./cuda_script)
        set(CUDAFILE ${CMAKE_SOURCE_DIR}/check_cuda.cu)
        execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
        execute_process(COMMAND ${OUTPUTFILE}
                RESULT_VARIABLE CUDA_RETURN_CODE
                OUTPUT_VARIABLE ARCH)

        if(${CUDA_RETURN_CODE} EQUAL 0)
            set(CUDA_SUCCESS "TRUE")
        else()
            set(CUDA_SUCCESS "FALSE")
        endif()

        if (${CUDA_SUCCESS})
            message(STATUS "CUDA Default Architecture: ${ARCH}")
            message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
            message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
            message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
            message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")
        else()
            message(WARNING ${ARCH})
        endif()
    endif ()

    message(STATUS "Architecture specified: ${ARCH}")

    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()
    cuda_add_compile_options("-std=c++11 -O3 -g -lineinfo ${ARCH}")
#    cuda_add_compile_options("--compiler-bindir /usr/local/bin/gcc6")

    cuda_add_compile_options("--expt-extended-lambda -Wno-deprecated-gpu-targets")

    if (${ENABLE_LOG})
        cuda_add_compile_options("-DUSE_LOG")
    endif()

    cuda_add_compile_options("-Xcompiler=\"-fopenmp\"")
    cuda_add_compile_options("-Xcompiler=\"-Wno-deprecated\"")
    #    cuda_add_compile_options("-Xptxas -v")         #list the usage of registers and memory

    if(${ENABLE_GPU})
        cuda_add_compile_options("-DUSE_GPU=1")           #use GPU for evaluation
    endif()

    set(CUDA_SOURCE_FILES tpch_q3_main_submit.cu util/log.cpp file_input_indexing.cu)
    cuda_add_executable(tpch-q3-submit ${CUDA_SOURCE_FILES})

endif()

#set(CPP_SOURCE_FILES tpch_q3_main_submit.cu util/log.cpp file_input_indexing.cu)
#add_executable(tpch-q3-submit-cpu ${CPP_SOURCE_FILES})

install(TARGETS tpch-q3-submit DESTINATION .)


