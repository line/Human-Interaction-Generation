## Installation

Please download `SMPL_NEUTRAL.pkl` from [this link](https://smpl.is.tue.mpg.de/) and put it in `./smpl_models/smpl`.  
Please setup as follows.

```
conda create -n render python=3.9 -y
source activate render
pip install tqdm h5py trimesh chumpy smplx
pip uninstall -y numpy
pip install numpy==1.23.1

## install osmesa
sudo apt update
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -f

## install pyrender
pip install git+https://github.com/mmatl/pyopengl.git
pip install pyrender
```

## Visualization

```
python render_smpl.py --file_name ../vis_data/pit/0_1/generated.npy
```