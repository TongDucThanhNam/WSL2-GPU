# Cách cài đặt GPU cho AI, ML, DP  - WSL2
- Các thư viện hỗ trợ GPU cho tính toán:
  - Tensorflow (Keras)
  - Pytorch (Torchvision, Torchtext, Torchaudio)
  - Cudf - thay cho Pandas
  - CuPy - thay cho Numpy và Scipy


## Cài đặt GPU cho WSL2
### Cài đặt WSL2
```bash
# Cài đặt WSL2
wsl --install
# Cài đặt Ubuntu
wsl --install -d Ubuntu
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

- Thêm conda vào PATH
```
# export PATH="/path/to/your/conda/bin:$PATH"
# /home/terasumi/miniconda3
export PATH="/home/terasumi/miniconda3/bin:$PATH"
```
- Check conda
```bash
conda --version
```


- Cài đặt không mở base conda
```bash
conda config --set auto_activate_base false
```

- Tạo môi trường conda
```bash
conda create --name tf python=3.10
```

- Reload bashrc
```bash
source ~/.bashrc
```

- Kích hoạt môi trường conda tf
```bash
conda activate tf
```
- Kiểm tra đã cài đặt NVIDIA GPU chưa
```bash
nvidia-smi
# Hiển thị thông tin GPU
```


- Install tensorflow (Giờ khi tải tensorflơw bằng lệnh bên dưới sẽ tự động cài đặt các gói Cuda cần thiết)
```bash
python3 -m pip install tensorflow[and-cuda]
```
- Cài thêm các thư viện khi bị báo lỗi
```bash
pip install --force-reinstall charset-normalizer==3.1.0
pip install nvidia-pyindex
pip install chardet
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

- Mở file jupyter_notebook_config.py
```bash
sudo nano jupyter_notebook_config.py
```

 - Thêm đoạn code sau vào file jupyter_notebook_config.py
```python
import os
c = get_config()
#/home/terasumi/miniconda3/envs/tf/lib/:/home/terasumi/miniconda3/envs/tf/lib/python3.10/site-packages/nvidia/cudnn/lib
os.environ['LD_LIBRARY_PATH'] = "/home/terasumi/miniconda3/envs/tf/lib/:/home/terasumi/miniconda3/envs/tf/lib/python3.10/site-packages/nvidia/cudnn/lib"
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
