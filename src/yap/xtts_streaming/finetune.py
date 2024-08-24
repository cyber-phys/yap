import os
import tempfile
import torch
import torchaudio
import shutil
from pathlib import Path
from yap.xtts_streaming.config_loader import ConfigLoader
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        model.to(device)
    return model

def preprocess_dataset(audio_path, language, whisper_model, out_path, train_csv, eval_csv, progress=None):
    from yap.xtts_streaming.utils.formatter import format_audio_list
    print("IN preprocess_dataset")
    clear_gpu_cache()
    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)

    if audio_path is None:
        return "You should provide one or multiple audio files!", "", ""

    train_meta, eval_meta, audio_total_size = format_audio_list(audio_files=audio_path, whisper_model=whisper_model, target_language=language, out_path=out_path)

    if audio_total_size < 120:
        return "The sum of the duration of the audios should be at least 2 minutes!", "", ""

    return "Dataset Processed!", train_meta, eval_meta

def train_model(config_path, vocab_file, train_csv, eval_csv, output_path, num_epochs=16, batch_size=2, grad_accum=1, max_audio_length=11):
    from yap.xtts_streaming.utils.gpt_train import train_gpt

    clear_gpu_cache()
    max_audio_length = int(max_audio_length * 22050)
    speaker_xtts_path, _, original_xtts_checkpoint, _, exp_path, speaker_wav = train_gpt("", "v2.0.3", "en", num_epochs, batch_size, grad_accum, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)

    ready_dir = Path(output_path) / "ready"
    os.makedirs(ready_dir, exist_ok=True)
    shutil.copy(os.path.join(exp_path, "best_model.pth"), ready_dir / "unoptimize_model.pth")

    speaker_reference_new_path = shutil.copy(speaker_wav, ready_dir / "reference.wav")
    return "Model training done!", str(ready_dir / "unoptimize_model.pth"), config_path, vocab_file, speaker_xtts_path, speaker_reference_new_path

def optimize_model(out_path, clear_train_data="none"):
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    if clear_train_data in {"run", "all"} and run_dir.exists():
        shutil.rmtree(run_dir)
    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    model_path = ready_dir / "unoptimize_model.pth"
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]
    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    os.remove(model_path)
    optimized_model_path = ready_dir / "model.pth"
    torch.save(checkpoint, optimized_model_path)
    clear_gpu_cache()
    return f"Model optimized and saved at {optimized_model_path}!", optimized_model_path
