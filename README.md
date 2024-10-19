# SurrogateCalib
Official Implementation of Iterative Camera-Lidar Calibration via Surrogate Diffusion Models
# Dependencies
|Pytorch|CUDA|Python|
|---|---|---|
|1.13.1|11.7|3.8.17|
# Build Packages
* Build csrc package for our method
```bash
cd models/tools/csrc/
python setup.py install
```
* Copy `.so` files into `models/tools/csrc/`
```bash
cp lib.linux-x86_64-cpython-38/* .
```
* Build correlation_cuda package for LCCNet
```bash
cd models/lccnet/correlation_package/
python setup.py install
```
The `correlation_cuda` package may be incompatible with CUDA >= 12.0. The failure of building this package only affects implementation of our baseline, LCCNet.
# Link KITTI Dataset to the root
* download KITTI dataset from [https://www.cvlibs.net/datasets/kitti/eval_odometry.php](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). (RGB, Veloydne and Calib data are all required)
* link the `dataset` filefolder as follows:
```bash
mkdir data
cd data
ln -s /path/to/kitti/ kitti
cd ..
```
# Train
* You can download our [pretrained models](https://github.com/gitouni/SurrogateCalib/releases/download/0.0/large_ckpt.zip) trained on KITTI Odometry Dataset or train them following the instructions.
