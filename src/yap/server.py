from __future__ import unicode_literals
from sanic import Sanic, response
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time
import numpy as np
import os
from typing import Union
import json
import wave
import base64
import torch
import uuid
import sqlite3
import asyncio
import curses
import sys
import threading
from yap.xtts_streaming import model_manager
from yap.xtts_streaming.model_manager import ModelManager
from yap.xtts_streaming.finetune import train_model, preprocess_dataset, optimize_model
from yap.xtts_streaming.inference import InferenceEngine

app = Sanic("TTS_Streaming_Server")

hf_token = os.getenv("HF_TOKEN")
model_manager = ModelManager(hf_token=hf_token)

def generate_voice(tts_text, voice_name, model_manager):
    finetuned_path = model_manager.get_path(voice_name)
    config_path = f"{finetuned_path}/ready/config.json"
    checkpoint_dir = f"{finetuned_path}/ready"
    sample_audio_path = f"{finetuned_path}/ready/reference.wav"
    print(config_path)

    model_manager.load_model(config_path, checkpoint_dir)
    model_manager.set_conditioning(sample_audio_path)
    inference_engine = InferenceEngine(model_manager)

    # Use the streaming option in the infer method
    for chunk in inference_engine.infer(tts_text, streaming=True):
        yield chunk

@app.route("/api/tts-stream")
async def tts_stream(request):
    text = request.args.get("text", "")
    voice_id = request.args.get("voice_id", "")
    language = request.args.get("language", "en")
    print(f"text: {text}")

    # Initialize the response as a stream, setting the appropriate content type for audio
    response = await request.respond(content_type="audio/wav")

    # Perform the streaming inference
    t0 = time.time()
    chunks = generate_voice(text, voice_id, model_manager)

    # Stream the audio chunks as they are generated
    first_chunk = True
    for i, chunk in enumerate(chunks):
        if first_chunk:
            first_chunk_time = time.time() - t0
            metrics_text = f"Latency to first audio chunk: {round(first_chunk_time*1000)} milliseconds\n"
            print(metrics_text)
            first_chunk = False

        chunk = chunk.detach().cpu().numpy().squeeze()
        chunk = (chunk * 32767).astype(np.int16)
        await response.send(chunk.tobytes())

    # End the stream
    await response.eof()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6666)

def main():
    host = input("Enter the host (default is 0.0.0.0): ") or "0.0.0.0"
    while True:
        try:
            port = int(input("Enter the port number (default is 6666): ") or "6666")
            break
        except ValueError:
            print("Please enter a valid port number.")

    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port)
