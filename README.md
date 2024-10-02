# SUPR C++

This repository contains a working C++ implementation using libtorch for the SUPR model.  
The implementation is as close as possible to the original.  
For the official python implementation please visit [SUPR](https://github.com/ahmedosman/SUPR).


## Setup

1. You need a current g++, cmake and anaconda or miniconda.

2. Setup the environment: 
You can use the provided `setup_env.sh`.  
This will download all the requirements for the following steps.  
The script assumes that you anaconda is installed at `~/anaconda3`, if that is not the case adjust `CONDA_PATH` inside the script.

3. Download the SUPR model files from [here](https://supr.is.tue.mpg.de/), note that the model files are subject to their license.  
These files must be converted using the `supr_convert.py` due, because they are not directly compatible with C++.  
Usage:  
```
//First activate environment
conda activate supr
//Actual conversion
python supr_convert.py -p "./EXAMPLE_PATH/model.npy"
```
 
## Build
CMake adjust the `PREFIX_PATH` if you anaconda environment is save somewhere else

Build SUPR library only:
```
cmake -S . -B build -Wno-dev -DCMAKE_PREFIX_PATH="~/anaconda3/envs/supr/;~/anaconda3/envs/supr/lib/python3.10/site-packages/torch/" && cmake --build build --parallel
```
Build with example:
```
cmake -S . -B build -Wno-dev -DCMAKE_PREFIX_PATH="~/anaconda3/envs/supr/;~/anaconda3/envs/supr/lib/python3.10/site-packages/torch/" -DSUPR_BUILD_EXAMPLES=ON && cmake --build build --parallel
```
You also need to set the path to the npz file in `example.cpp`.
   
Build without cuda:
```
cmake -S . -B build -Wno-dev -DCMAKE_PREFIX_PATH="~/anaconda3/envs/supr/;~/anaconda3/envs/supr/lib/python3.10/site-packages/torch/" -DBUILD_WITH_CUDA=OFF && cmake --build build --parallel
```
