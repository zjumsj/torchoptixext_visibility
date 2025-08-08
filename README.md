# Pytorch Extension using OptiX for Visibility 

This project is an efficient and easy-to-use PyTorch library that provides GPU-accelerated visibility computation powered by NVIDIA OptiX, only for **research purposes**.

This library facilitates visibility calculations for differentiable rendering applications. Data is passed directly through torch tensors residing on GPU, eliminating unnecessary CPU-GPU transfer overhead.

The library specializes in **visibility and depth computation** for computer graphics and vision research, providing a focused interface for these specific operations. For general-purpose OptiX Python bindings, consider [PyOptiX](https://github.com/ozen/PyOptiX) or [otk-pyoptix](https://github.com/NVIDIA/otk-pyoptix).

### Supported Platforms
Both **Windows** and **Linux** platforms.

### Supported OptiX Versions
- ✅ **OptiX 7.1+** (current support)
- ⬜ **OptiX 8.x.x** (future plan)

## Installation
### Prerequisites
- CUDA-capable GPU
- Python environment with **CUDA-supported PyTorch**
- C/C++ Compiler:  
  - Linux: GCC/G++  
  - Windows: Visual Studio (MSVC)
- NVIDIA OptiX

### Setup
1. Install CUDA (verify installation with `nvcc --version`)
2. Get OptiX:
   - Download from [NVIDIA Developer](https://developer.nvidia.com/designworks/optix/download)
   - **Windows**: Run installer  
   - **Linux**: Run self-extracting script
3. Compile the demo (example below)
```shell
# Linux example
bash NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64.sh
cd NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/SDK

# Windows example (Command Prompt)
cd /d "C:/ProgramData/NVIDIA Corporation/Optix SDK 7.4.0/SDK"

# Common steps for both platforms
mkdir build
cd build
cmake ..

# Linux (using Make)
make -j$(nproc)

# Windows (using Visual Studio)
# Open the generated solution file in Visual Studio and build
# OR use command line:
cmake --build . --config Release
```
4. Set environment variables
```shell
# Linux example
export OPTIX_PATH="/path/to/optix"
export OPTIX_VERSION="7.4.0"
export CUDA_PATH="/usr/local/cuda-11.8"

# Windows example (Command Prompt)
set OPTIX_PATH="C:\ProgramData\NVIDIA Corporation\Optix SDK 7.4.0"
set OPTIX_VERSION="7.4.0"
set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
```

5. Clone and install
```shell
git clone https://github.com/zjumsj/torchoptixext_visibility.git
cd torchoptixext_visibility
python setup.py install
```

## Demo

Running the demo requires some additional packages:
```shell
pip install "numpy<2.0" opencv-python 
```
### Quick test
Run a simple triangle rendering example to verify proper installation.
```shell
python demo/helloWorldTriangle.py
```

### Depth map

```shell
# render with pinhole camera
python demo/renderDepthMap.py
# render with envmap camera
python demo/renderDepthMap.py --camera_type envmap_camera -H 512 -W 1024
```
As unoccluded rays return inf, this also enables visibility map computation.

### Visibility bitfield

For compute-intensive applications such as real-time computation of environment map visibility for 10k light probes, each with 16×32 directions, we implemented additional optimizations. The visibility of each ray is encoded into a compact bitfield to reduce GPU memory access overhead.

```shell
# for static scene
python demo/visibilityFieldStatic.py
# for dynamic scene
python demo/visibilityFieldDynamic.py
# test fps for rendering 10k light probe. It achieves 900 FPS on my RTX 2080 Ti for this test case.
python demo/visibilityFieldDynamic.py --testFPS --row 100 --col 100 --envmap_H 16 --envmap_W 32
```
## Cite
Please kindly cite our repository and preceding paper if you find our software or algorithm useful for your research.
```
@misc{ma2025torchextforvis,
    title={Pytorch Extension using OptiX for Visibility},
    author={Ma, Shengjie},
    year={2025},
    mouth={aug},
    url={https://github.com/zjumsj/torchoptixext_visibility}
}
```
```
@inproceedings{ma2025relightable,
    title={Relightable Gaussian Blendshapes for Head Avatar Animation},
    author={Ma, Shengjie and Zheng, Youyi and Weng, Yanlin and Zhou, Kun},
    booktitle={International Conference on Computer-Aided Design and Compute Graphics},
    year={2025}
}
```

