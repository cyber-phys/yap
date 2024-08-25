# Yap TTS
## Usage
Clone a voice:
```
nix develop
uv run yap clone example.wav
```

Query a voice:
```
nix develop
uv run yap -n voice_name
```

## Notes
fix for libcudnn8_8 issues with [lamda stack](https://lambdalabs.com/lambda-stack-deep-learning-software)
```
uv run wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
uv run https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
uv run sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
uv run sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
uv run sudo apt-get install -f
```
