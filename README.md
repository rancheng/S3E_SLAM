### S3E-SLAM

Sparse Spatial Scene Embedding powered SLAM system.

#### Dependencies

Here is the dependencies

for g2o:

```shell
sudo apt install libsuitesparse-dev\
                  qtdeclarative5-dev\
                  qt5-qmake\
                  libqglviewer-dev\
                  cmake\
                  libeigen3-dev
```

for pangolin:

```shell
sudo apt install libgl1-mesa-dev\
                  libglew-dev\
                  ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev\
pip install pyopengl
```


```shell
conda install -c dglteam dgl-cuda10.2
```



#### Build Instruction

Build the `g2o` and `pangolin` from source

 - g2o
```shell
cd third_party
cd g2opy
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make -j12
```

 - pangolin
```shell
cd third_party
cd pangolin
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make -j8
cd ..
python setup.py install 
```

#### Dataset Prepare

Please follow the following steps to build the dataset

- run the Vins-RGBD to generate the pose graph and point cloud map data
- run the dataset preprocess script to cook the dataset (crop the patch, calculate ground truth iou)

#### Credit

The project is based on multiple open-source projects including:

 - [rgbd_ptam](https://github.com/uoip/rgbd_ptam)
 - [vins_rgbd](https://github.com/STAR-Center/VINS-RGBD)
