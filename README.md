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
<details>
  <summary>Troubleshooting</summary>
  The `correlation_cuda` package may be incompatible with CUDA >= 12.0. The failure of building this package only affects implementation of our baseline, LCCNet. If you have CUDA >= 12.0 and still want to implement LCCNET, it would be easy to use correlation pacakge in csrc to re-implement it. To try our best to reproduce LCCNet's performance, we utilize their own correlation package.
</details>

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
* You can download our [pretrained models](https://github.com/gitouni/SurrogateCalib/releases/download/0.1/LSD_chkpt.zip) trained on KITTI Odometry Dataset or train them following the instructions.
* Train a surrogate model (dataset config + model config)
```bash
python train.py --dataset_config cfg/dataset/kitti_large.yml --model_config cfg/unipc_model/main.yml
```
Change `main.yml` to other configs in `cfg/unipc_model` to train baselines, and the same applies to the following commands.
* Train a vae for RGGNet (note that this should be run before training RGGNet)
```bash
python train_vae.py --dataset_config cfg/dataset/kitti_vae_large.yml --model_config cfg/model/vae.yml
```
# Test
* one-step mode
```bash
python test.py --dataset_config cfg/dataset/kitti_large.yml --model_config cfg/unipc_model/main.yml --model_type iterative --iters 1
```
* naive iterative method (NFE=10)
```bash
python test.py --dataset_config cfg/dataset/kitti_large.yml --model_config cfg/unipc_model/main.yml --model_type iterative --iters 10
```
* Linear Surrogate Diffusion Model
```bash
python test.py --dataset_config cfg/dataset/kitti_large.yml --model_config cfg/unipc_model/main.yml --model_type diffusion
```
* Non-Linear Surrogate Diffusion Model
```bash
python test_se3_diff.py --dataset_config cfg/dataset/kitti_large.yml --model_config cfg/unipc_model/main_sd.yml --model_type se3_diffusion
```
# Acknowledgements
Thanks authors of [CamLiFLow](https://github.com/MCG-NJU/CamLiFlow), [DPM-Solver](https://github.com/LuChengTHU/dpm-solver), [UniPC](https://github.com/wl-zhao/UniPC) and [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
