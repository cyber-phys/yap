Clone a voice: `uv run yap clone ivy_2.wav`

libcudnn8_8 fix on lamda stack
```
uv run wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
uv run https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
uv run sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
uv run sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
uv run sudo apt-get install -f  # resolve dependency errors you saw earlier
```
