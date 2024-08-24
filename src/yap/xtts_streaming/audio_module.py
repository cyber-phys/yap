import threading
import queue
import sounddevice as sd
import numpy as np
import sys

class AudioPlayer:
    def __init__(self, sample_rate=24000, blocksize=2048, buffersize=20):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.buffersize = buffersize
        self.queue = queue.Queue(maxsize=self.buffersize)
        self.event = threading.Event()

    def play_audio(self):
        def callback(outdata, frames, time, status):
            if status.output_underflow:
                print('Output underflow: increase blocksize?', file=sys.stderr)
                raise sd.CallbackAbort
            if status:
                print(status)
            try:
                data = self.queue.get()
                print(data)
            except queue.Empty as e:
                print('Buffer is empty: increase buffersize?', file=sys.stderr)
                raise sd.CallbackAbort from e
            outdata[:len(data)] = data
            if len(data) < len(outdata):
                outdata[len(data):] = 0

        stream = sd.OutputStream(
            samplerate=self.sample_rate, blocksize=self.blocksize,
            channels=1, dtype='int16', callback=callback,
            finished_callback=self.event.set
        )
        with stream:
            self.event.wait()  # Wait until playback is finished

    def add_audio_chunk(self, chunk):
        self.queue.put(chunk)

    def stop_audio(self):
        self.event.set()
        self.queue.put(np.zeros((self.blocksize, 1), dtype='int16'))  # Send a dummy chunk to unblock the queue
