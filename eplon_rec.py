#-*-coding:utf-8-*-
#!/usr/bin/python
import pyaudio
import wave
import sys
import time

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 2**12
RECORD_SECONDS = 1
DEVICE_INDEX = 1
frames = []


WAVE_FILE = "./predict_441kHz/out.wav"

def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return(None, pyaudio.paContinue)


def rec():


    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index = DEVICE_INDEX,
        start=False,
        stream_callback=callback
    )
    stream.start_stream()
    time.sleep(RECORD_SECONDS)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf = wave.open(WAVE_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()