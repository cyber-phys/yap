import time
import torch
import torchaudio
import numpy as np
from threading import Thread
from queue import Queue
from yap.xtts_streaming.audio_module import AudioPlayer

class InferenceEngine:
    def __init__(self, model_manager):
        self.model_manager = model_manager


    def infer(self, text, language="en", play=True, streaming=False):
        t0 = time.time()
        chunks = self.model_manager.current_model.inference_stream(
            text,
            language,
            self.model_manager.current_gpt_cond_latent,
            self.model_manager.current_speaker_embedding
        )
        wav_chunks = []
        for i, chunk in enumerate(chunks):
            if streaming:
                yield chunk
            if i == 0:
                print(f"Time to first chunk: {time.time() - t0}")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

        if not streaming:
            wav = torch.cat(wav_chunks, dim=0)
            return wav

    def save_wav(self, wav, filepath, sample_rate=24000):
        torchaudio.save(filepath, wav.squeeze().unsqueeze(0).cpu(), sample_rate)
