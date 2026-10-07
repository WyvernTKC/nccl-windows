# NCCL

Optimized primitives for inter-GPU communication.

This repository is NVIDIA NCCL 2.32.3 with two small patches for Windows. It builds with CUDA 12.x, which Volta (V100) and older GPUs need because CUDA 13 dropped them, and it no longer sleeps for milliseconds in its Windows wait loops. See [Build on Windows](#build-on-windows).

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Build on Windows

Requirements:

- Visual Studio 2022 with the C++ x64 build tools (the free Build Tools edition is enough)
- CUDA Toolkit 12.x. Volta (V100) and older GPUs need 12.x, because CUDA 13 cannot target them.
- CMake 3.25 or newer, Ninja, and Python 3

From `cmd.exe`, in the repository root:

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_INSTALL_PREFIX=install
cmake --build build --parallel 16 --target install
```

Change the `vcvarsall.bat` path to match your Visual Studio edition. Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability of your GPUs, separated by semicolons (for example `70;86`). 70 is V100 and 75 is T4. If you leave it unset, the build targets a long default list of architectures, which takes much longer.

The build installs:

- `install\bin\nccl.dll`: the library
- `install\lib\nccl.lib`: the import library for `nccl.dll`
- `install\lib\nccl_static.lib` and `install\lib\nccl_device.lib`: the static library, which needs both files
- `install\include\`: the headers, including `nccl.h`
- `install\lib\cmake\NCCL\`: a CMake package for `find_package(NCCL)`

To use it, put `install\bin` on `PATH` or copy `nccl.dll` next to your executable. This build has been tested as a drop-in replacement for an application built against NCCL 2.29.7.

### Tested configuration

- Windows Server 2022, Visual Studio 2022 Build Tools (MSVC 14.44), CUDA 12.9, CMake 4.3, Ninja 1.13
- 4x Tesla V100-SXM2 connected by NVLink, in TCC mode. WDDM mode is untested.
- Single-process AllReduce, AllGather and Broadcast, with every result checked. AllReduce reaches about 118 GB/s bus bandwidth at 256 MB.

### Changes from NVIDIA NCCL 2.32.3

- **Builds with CUDA 12.x.** The build uses C++14 when CUDA is older than 13, but two `static_assert`s used a form that needs C++17.
- **No oversleeping on Windows.** NCCL's async job waits asked for a 1 µs sleep, which Windows rounds up to 1-16 ms. They now yield for the first 50 ms of a wait before falling back to sleeping.

### Windows notes

- Without an RDMA network adapter, NCCL logs `Failed to initialize NET plugin NetworkDirect` and falls back to TCP sockets. This is expected. GPUs in the same machine talk over NVLink or PCIe, not the network.
- Leave `NCCL_ENQUEUE_REARCH_ENABLE` and `NCCL_SINGLE_PROC_MEM_REG_ENABLE` unset. With several GPUs in one process, both create a thread per GPU on every call, which costs about 1 ms per collective on Windows.

## Build

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## Install

To install NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

Python wheel :
```shell
$ # Install uv to create the Python wheel (uv manages Python deps in a venv)
$ # See: https://docs.astral.sh/uv/getting-started/installation/
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ # Build NCCL Python wheel (this also builds the .txz archive as an intermediate)
$ make pkg.python_wheel.build
$ ls build/pkg/python_wheel/
```

## Tests

Tests for NCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
