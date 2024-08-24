import torch
from TTS.tts.models.xtts import Xtts
from yap.xtts_streaming.config_loader import ConfigLoader
from huggingface_hub import hf_hub_download
from yap.xtts_streaming.huggingface_utils import HuggingfaceUtils
from huggingface_hub import cached_assets_path
from TTS.tts.configs.xtts_config import XttsConfig


class ModelManager:
    def __init__(self, hf_token):
        self.current_model = None
        self.current_speaker_embedding = None
        self.current_gpt_cond_latent = None
        self.hf_utils = HuggingfaceUtils(hf_token)

    def load_model(self, config_path, checkpoint_dir, use_deepspeed=False):
        # config = ConfigLoader.load_config(config_path)
        # print(config)
        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed)
        model.cpu()
        # model.cuda()
        self.current_model = model

    def set_conditioning(self, audio_path):
        gpt_cond_latent, speaker_embedding = self.current_model.get_conditioning_latents(audio_path=[audio_path])
        self.current_gpt_cond_latent = gpt_cond_latent
        self.current_speaker_embedding = speaker_embedding

    def load_model_from_hf(self, model_id, config_filename="ready/config.json", checkpoint_dirname="ready/model.pth", sample_audio_filename="ready/reference.wav"):
        config_path = hf_hub_download(repo_id=model_id, filename=config_filename)
        checkpoint_dir = hf_hub_download(repo_id=model_id, filename=checkpoint_dirname)
        sample_audio_path = hf_hub_download(repo_id=model_id, filename=sample_audio_filename)
        self.load_model(config_path, checkpoint_dir)
        self.set_conditioning(sample_audio_path)

    def upload_model_to_hf(self, model_id, local_dir):
        self.hf_utils.upload_model_files(model_id, local_dir)

    def download_model_from_hf(self, model_id, local_dir):
        self.hf_utils.download_model_files(model_id, local_dir)

    def get_path(self, voice_name):
        cache_path = cached_assets_path(library_name="xtts", namespace="voices", subfolder=voice_name)
        return cache_path
