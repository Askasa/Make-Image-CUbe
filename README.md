# MICU
## About

## Getting Started
### Utils Downloading
```PowerShell
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libflann-dev \
    libmetis-dev
```
```PowerShell
sudo apt-get install libatlas-base-dev libsuitesparse-dev
sudo apt-get install libeigen3-dev libgoogle-glog-dev libgflags-dev
```
```PowerShell
pip install ninja
```
[CERES-SOLVER](https://ceres-solver.googlesource.com/ceres-solver)

[COLMAP](https://github.com/colmap/colmap)

[LLFF](https://github.com/Fyusion/LLFF)

### Data Prepare

- Sythetic data
- Real world data

```Python
dir = "{your data forder}/{folder contain image.jpg}"
training_data, testing_data = sample_tt(dir, 15, 5)
with open('training_data.pkl','wb') as f: pickle.dump(training_data.numpy(), f)
with open('testing_data.pkl','wb') as f: pickle.dump(testing_data.numpy(), f)
```

### Rendering result

```cmd
python train.py --num_epochs 2 --fine_stage 1 --device 'cpu'
```