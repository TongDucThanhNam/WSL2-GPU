# Cách cài đặt GPU cho AI, ML, DP  - WSL2
- Các thư viện hỗ trợ GPU cho tính toán:
  - Tensorflow (Keras)
  - Pytorch (Torchvision, Torchtext, Torchaudio)
  - Cudf - thay cho Pandas
  - CuPy - thay cho Numpy và Scipy
  - Hummingbird - cải tiến Scikit-learn để dụng tourch-GPU
  - Caffe
  - Caffe2
  - MXNet
  - CNTK : 
  - Theano
  - Chainer

## Cài đặt GPU cho WSL2
### Cài đặt WSL2
```bash
# Cài đặt WSL2
wsl --install
# Cài đặt Ubuntu 20.04
wsl --install -d Ubuntu-20.04
```

### Thiết lập WSL2 Ubuntu 20.04
- Cài đặt miniconda
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

- Restart WSL2
```bash
source ~/.bashrc
```

- Cài đặt không mở base conda
```bash
conda config --set auto_activate_base false
```

- Tạo môi trường conda
```bash
conda create --name tf python=3.10
conda activate tf
```
- Kiểm tra đã cài đặt NVIDIA GPU chưa
```bash
nvidia-smi
```


- Cài đặt CUDA and cuDNN
```bash
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
```

- Thêm đường dẫn CUDA vào PATH

[//]: # (&#40;export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}&#41;)

[//]: # (&#40;export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}&#41;)
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
- check cuda path
```bash
conda activate tf
echo $LD_LIBRARY_PATH
```

- Install tensorflow
```bash
pip install tensorflow==2.10
```
- Cài thêm các thư viện khi bị báo lỗi
```bash
pip install --force-reinstall charset-normalizer==3.1.0
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
```

- Kiểm tra tensorflow GPU
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
- Nếu có hiển thị GPU thì đã cài đặt thành công
- Cài đặt Jupiter notebook

```bash
conda install -c anaconda jupyter
```
 - Chỉnh sửa file jupyter notebook
```bash
jupyter notebook --generate-config
```
 - Tìm đến file jupyter_notebook_config.py
```bash
cd ~/.jupyter
```
 - Thêm đoạn code sau vào file jupyter_notebook_config.py
```python
import os
c = get_config()
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-8.0/lib64:usr/local/cuda-8.0/lib64/libcudart.so.8.0'
c.Spawner.env.update('LD_LIBRARY_PATH')
```

 - Code kiểm tra gpu trong jupyter notebook
```python
import tensorflow as tf

tf.test.is_gpu_available()
```

- Cài đặt Pytorch
```bash
conda install pytorch torchvision torchaudio
```
- Kiểm tra Pytorch GPU
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device)
```

- Cài đặt Hummingbird
```bash
pip install hummingbird-ml
```




