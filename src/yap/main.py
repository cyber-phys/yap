import curses
import sys
import threading
from yap.xtts_streaming import model_manager
from yap.xtts_streaming.model_manager import ModelManager
from yap.xtts_streaming.finetune import train_model, preprocess_dataset, optimize_model
from yap.xtts_streaming.inference import InferenceEngine
import time
import os

def clone_voice_cli(sample_audio_path, model_manager):
    cloned_voice_name = input("Enter the name for the cloned voice: ")

    print("Cloning voice, please wait...")

    def loading_animation():
        while not animation_done:
            for frame in r"-\|/":
                sys.stdout.write(f"\r[{frame}]")
                sys.stdout.flush()
                time.sleep(0.1)

    animation_done = False
    animation_thread = threading.Thread(target=loading_animation)
    animation_thread.start()

    # Fine-tune the model
    train_result = finetune_model(cloned_voice_name, sample_audio_path, model_manager)

    animation_done = True
    animation_thread.join()

    print(f"\nVoice cloned successfully and saved as '{cloned_voice_name}'!")
    input("Press any key to exit.")

def clone_voice_tui(stdscr, sample_audio_path, model_manager):
    curses.curs_set(1)
    stdscr.clear()
    stdscr.addstr("Enter the name for the cloned voice: ")
    stdscr.refresh()

    curses.echo()
    cloned_voice_name = stdscr.getstr().decode("utf-8")
    curses.noecho()

    stdscr.clear()
    stdscr.addstr("Cloning voice, please wait...\n[")
    stdscr.refresh()

    def loading_animation():
        while not animation_done:
            for frame in r"-\|/":
                stdscr.addstr(2, 1, frame)  # Update this line: (y, x, string)
                stdscr.refresh()
                time.sleep(0.1)

    animation_done = False
    animation_thread = threading.Thread(target=loading_animation)
    animation_thread.start()

    # Fine-tune the model
    train_result = finetune_model(cloned_voice_name, sample_audio_path, model_manager)

    animation_done = True
    animation_thread.join()

    stdscr.clear()
    stdscr.addstr(f"Voice cloned successfully and saved as '{cloned_voice_name}'!\n")
    stdscr.addstr("Press any key to exit.")
    stdscr.getch()

def finetune_model(cloned_voice_name, sample_audio_path, model_manager):
    output_path = model_manager.get_path(cloned_voice_name)
    print(output_path)
    preprocess_result = preprocess_dataset(audio_path=sample_audio_path, whisper_model="large-v3", language="en", out_path=output_path, train_csv="", eval_csv="")

    ready_model_path = output_path / "ready"
    vocab_path = ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path = ready_model_path / "speakers_xtts.pth"
    reference_path = ready_model_path / "reference.wav"

    train_result = train_model(config_path=config_path, vocab_file=vocab_path, train_csv=preprocess_result[1], eval_csv=preprocess_result[2], output_path=output_path)
    optimize_result = optimize_model(output_path)
    model_id="test"
    # model_manager.upload_model_to_hf(f"xtts-finetune-{cloned_voice_name}", output_path)
    return train_result

def generate_voice_tui(stdscr, voice_name, model_manager):
    curses.curs_set(1)
    stdscr.clear()
    stdscr.addstr(f"You are using the voice model: {voice_name}\n")
    stdscr.addstr("Please wait while the model is being loaded...\n")
    stdscr.refresh()

    finetuned_path = model_manager.get_path(voice_name)
    config_path = f"{finetuned_path}/ready/config.json"
    checkpoint_dir = f"{finetuned_path}/ready"
    sample_audio_path = f"{finetuned_path}/ready/reference.wav"
    print(config_path)
    # model_manager.load_model_from_hf(f"xtts-finetune-{voice_name}")
    model_manager.load_model(config_path, checkpoint_dir)
    model_manager.set_conditioning(sample_audio_path)
    inference_engine = InferenceEngine(model_manager)

    stdscr.addstr("Enter text to synthesize: \n")
    stdscr.refresh()

    curses.echo()
    tts_text = stdscr.getstr().decode("utf-8")
    curses.noecho()

    wav = inference_engine.infer(tts_text, play=True)
    inference_engine.save_wav(wav, f"{voice_name}_output.wav")

    stdscr.clear()
    stdscr.addstr(f"Synthesized voice saved as '{voice_name}_output.wav'\n")
    stdscr.addstr("Press any key to exit.")
    stdscr.getch()

def main():
    hf_token = os.getenv("HF_TOKEN")
    model_manager = ModelManager(hf_token=hf_token)

    if len(sys.argv) < 2:
        print("Usage: yap <command> [options]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "clone":
        if len(sys.argv) < 3:
            print("Usage: yap clone <sample_audio.wav>")
            sys.exit(1)

        sample_audio_path = sys.argv[2]
        # curses.wrapper(clone_voice_tui, sample_audio_path, model_manager)
        clone_voice_cli(sample_audio_path, model_manager)
    elif command == "-n":
        if len(sys.argv) < 3:
            print("Usage: yap -n <voice_name>")
            sys.exit(1)

        voice_name = sys.argv[2]
        curses.wrapper(generate_voice_tui, voice_name, model_manager)
        print("Unknown command.")
        sys.exit(1)

if __name__ == "__main__":
    main()
main()
