# GBSI: GPU-Accelerated B+Tree Secondary Index

## Overview

GBSI is a high-performance secondary index engine that leverages GPU-accelerated B+Tree structures for fast data insertion and query. The project supports both CPU and GPU-based index management, and is designed for integration with LevelDB as the underlying storage engine.

## Features

- **GPU-accelerated B+Tree** for secondary index management
- **LevelDB** integration for persistent storage
- **Batch insertion** and **range/point queries**
- **Multi-threaded** flush and conversion pipeline
- **Benchmark tools** for insertion and query performance evaluation

## Prerequisites

- **CMake** (>= 3.19)
- **CUDA Toolkit** (for GPU acceleration, tested with CUDA 11+)
- **C++17** compatible compiler
- **LevelDB** (install via your package manager or from [LevelDB GitHub](https://github.com/google/leveldb))
- **Snappy** (compression library, required by LevelDB)
- **NVIDIA GPU** (for GPU acceleration; CPU-only mode is not the main target)

### Ubuntu Example

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libsnappy-dev libleveldb-dev
# CUDA Toolkit: follow NVIDIA's official instructions for your GPU/OS
```

## Build Instructions

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone <https://github.com/Nick-jie/GBSI.git>
    cd <GBSI>
    ```

2. **Build the project:**

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

3. **Run benchmarks:**

    - **Insertion benchmark:**
      ```bash
      ./benchmark_insert
      ```
    - **Query benchmark:**
      ```bash
      ./benchmark_query
      ```

## Project Structure

- `memtable/`  
  Core source code for secondary index management, GPU/CPU index managers, and benchmarks.
- `src/`  
  GPU B+Tree implementation, memory allocators, and kernel code.
- `CMakeLists.txt`  
  Top-level build configuration.

## Notes

- **LevelDB** must be installed and discoverable by CMake. If installed in a non-standard location, set `LEVELDB_ROOT` or adjust `CMAKE_PREFIX_PATH`.
- The benchmarks will create and use LevelDB databases in `/opt/Leveldb_DB_DOC/` by default. Make sure you have write permissions or adjust the paths in the code.
- For best performance, run on a machine with a recent NVIDIA GPU and sufficient memory.

