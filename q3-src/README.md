# TPC-H Q3

## Compile 

see [../compile.sh](../compile.sh)

## Run

see [../run.sh](../run.sh)

## 目录结构

Main File: [tpch_q3_main_submit.cpp](tpch_q3_main_submit.cpp)

### Parsing and Indexing

Files/Folders | Description
--- | ---
[file_loader.h](file_loader.h) | Input File Loading
[file_parser.h](file_parser.h), [parsing_util.h](parsing_util.h) | Parsing
[file_input_helper.h](file_input_helper.h), [file_input_helper.cpp](file_input_helper.cpp) | Indexing
[lock_free_table.h](lock_free_table.h) | For the customer category mapping

### Querying

Files/Folders | Description
--- | ---
[index_query_helper.cpp](index_query_helper.cpp), [index_query_helper.h](index_query_helper.h) | Query Logic
[query_offloading_gpu.cu](query_offloading_gpu.cu) | GPU Query Acceleration

### Utils

Files/Folders | Description
--- | ---
[cuda/cub](cuda/cub), [cuda/primitives.cuh](cuda/primitives.cuh) | GPU primitives
[cuda/CUDAStat.cuh](cuda/CUDAStat.cuh), [cuda/cuda_base.cuh](cuda/cuda_base.cuh) | CUDA helpers
[util/primitives](util/primitives) | 并行有关的primitives
[util/archive.h](util/archive.h) | 序列化meta文件的util
[util/timer.h](util/timer.h), [util/util.h](util/util.h), [util/log.cpp](util/log.cpp) | timer, loggers

