import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from version import Version, replace_blackslashes, strip_quotes, fill_template, find_nvrtc_dll

_src_path = os.path.dirname(os.path.abspath(__file__))

## set path
optix_path = os.environ.get('OPTIX_PATH', None)
optix_version = os.environ.get('OPTIX_VERSION', None)
cuda_path = os.environ.get('CUDA_PATH', None)
if optix_path is None:
    print('Please set OPTIX_PATH environment variable. Installation aborted.')
    exit(0)
if optix_version is None:
    print('Please set OPTIX_VERSION environment variable. Installation aborted.')
    print('Example: OPTIX_VERSION=\"7.4.0\"')
    exit(0)
if cuda_path is None:
    print('Please set CUDA_PATH environment variable. Installation aborted.')
    exit(0)

optix_path = strip_quotes(optix_path)
optix_version = strip_quotes(optix_version)
cuda_path = strip_quotes(cuda_path)

version = Version(optix_version)
version_7_1 = Version("7.1.0")
version_7_2 = Version("7.2.0")
version_8_0 = Version("8.0.0")

assert version >= version_7_1 and version < version_8_0, "Only support Optix 7.1<=version<8"

## sources list
sources = [
    'binding1.cpp',
    'PyOptixTriangle.cpp',
    'PyOptixDist.cpp',
    'PyOptixEnvmapVisibility.cpp',
    'myTools.cpp',
]

nvcc_flags = [
    '-O3', '-std=c++14',
    '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
]

if version <= version_7_2:
    sources.append('OptixUtil.cpp')

if os.name == 'posix': # Linux
    replace_dict = {
        "LINUX_OPTIX_LIB_PATH" : replace_blackslashes(os.path.join(optix_path, "SDK/build/lib/libsutil_7_sdk.so"))
    }
elif os.name == 'nt': # Win
    nvrtc_dll = find_nvrtc_dll(cuda_path)
    replace_dict = {
        "WIN_OPTIX_LIB_PATH" : replace_blackslashes(os.path.join(optix_path, "SDK/build/bin/Release")),
        "WIN_NVRTC_LIB_PATH" : replace_blackslashes(nvrtc_dll)
    }
else:
    raise NotImplementedError("Unsupported system. Currently only Linux and Windows are supported.")

# Automatically fill in PATH
fill_template("torchoptixext_visibility/init.template", "torchoptixext_visibility/__init__.py", replace_dict)

replace_dict = {
    "OPTIX_INSTALL_PATH_SDK" : replace_blackslashes(os.path.join(optix_path, "SDK")),
    "OPTIX_INSTALL_PATH_INCLUDE" : replace_blackslashes(os.path.join(optix_path, "include")),
    "CUDA_INSTALL_PATH_INCLUDE" : replace_blackslashes(os.path.join(cuda_path, "include"))
}

fill_template("myTools.template", "myTools.cpp", replace_dict)

if os.name == "posix": # Linux
    c_flags = ['-O3', '-std=c++14']

    setup(
        name='torchoptixext_visibility', # package name, import this to use python API
        ext_modules=[
            CUDAExtension(
                name='_torchoptixext_visibility', # extension name, import this to use CUDA API
                sources=sources,
                extra_compile_args={
                    'cxx': c_flags,
                    'nvcc': nvcc_flags,
                },
                include_dirs = [ 
                    os.path.join(optix_path,"include"),
                    os.path.join(optix_path,"SDK"),
                    os.path.join(optix_path,"SDK/build")
                ],
                library_dirs = [
                    os.path.join(optix_path,"SDK/build/lib")
                ],
                libraries = [
                    'sutil_7_sdk','nvrtc'
                ]
            ),
        ],
        packages=['torchoptixext_visibility'],
        package_data={
          'torchoptixext_visibility': [
              'cu/*.cu', 'cu/*.h', 'cu/*.cuh'
          ]
        },
        cmdclass={
            'build_ext': BuildExtension,
        }
    )

elif os.name == "nt": # Win
    c_flags = ['/O2', '/std:c++17']
    c_flags += ['-DNOMINMAX']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

    setup(
        name='torchoptixext_visibility', # package name, import this to use python API
        ext_modules=[
            CUDAExtension(
                name='_torchoptixext_visibility', # extension name, import this to use CUDA API
                sources=sources,
                extra_compile_args={
                    'cxx': c_flags,
                    'nvcc': nvcc_flags,
                },
                include_dirs = [ 
                    os.path.join(optix_path,"include"),
                    os.path.join(optix_path,"SDK"),
                    os.path.join(optix_path,"SDK/build")
                ],
                library_dirs = [
                    os.path.join(optix_path, "SDK/build/lib/Release")
                ],
                libraries = [
                    'sutil_7_sdk','nvrtc','advapi32'
                ]
            ),
        ],
        packages=['torchoptixext_visibility'],
        package_data={
            'torchoptixext_visibility': [
                'cu/*.cu', 'cu/*.h', 'cu/*.cuh'
            ]
        },
        cmdclass={
            'build_ext': BuildExtension,
        }
    )
