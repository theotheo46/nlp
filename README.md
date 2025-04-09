## nlp karpov project

### Selectel host theo_nlp

```
ssh root@176.114.66.250
```

### Gitlab

```
ssh-keygen -t rsa -b 2048 -C "aiwatcher"
git clone git@gitlab.com:theotheo46/nlp.git
```

### Check nvidia driver

```
nvidia-smi
nvcc --version
```

### Проверка куды

```
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

### wandb

```
wandb login cc603ae0565bbbfce5cc5b068a9bddabb8950920
```

### Some commands

```
du -h . | sort -n -r | head -n 20
tar -cvzf - /path/to/local/directory | ssh root@195.142.145.66 'tar -xzf - -C /path/to/remote/directory'
tar tvf file.tar --wildcards '*.png' 
du -hs
```

## Необязательные шаги - на Selectel DS compatible server делать не надо
#### Как устанавливать torch под нужную куду

- For CUDA 11
```
pip install ... /whl/cu118
```

- For CUDA 12

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
#### Перед первой тренировкой модели установить зависимости для сборки CUDA библиотек

```
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit
sudo apt-get install -y build-essential python3-dev
```
#### перезапуск Jupyter kernel

#### install NVidia driver

```
sudo apt update && sudo apt upgrade -y && sudo apt autoclean
sudo apt remove nvidia -y
for kernel in $(linux-version list); do apt install -y "linux-headers-${kernel}"; done
sudo apt install -y nvidia-driver-450-server
```
