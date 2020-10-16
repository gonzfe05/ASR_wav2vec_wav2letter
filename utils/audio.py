from __future__ import absolute_import, division, print_function
# Model imports
from .silenceRemove import frame_generator, vad_collector, read_wave
# Audio manipulation imports
import webrtcvad
from pydub import AudioSegment
# System imports
import os
# Misc imports
import numpy as np

def preprocessing(args: tuple) -> None:
    """Prepares audio for transcription

    Read wav, converts from 8k to 16k, chunks to length of 12.
    To be used by pool parallel process, that is why it acepts just one argument (tuple)
    Args:
        args (tuple): (file_path, output_path)
    """
    file_path = args[0]
    output_path = args[1]
    audio = trim_audio(file_path, output_path, max_len = 3)
    audio = audio.set_frame_rate(16000)
    filename = file_path.split('/')[-1]
    audio.export(os.path.join(output_path, filename), format="wav")

def silenceRemoveWrapper(file_path: str, aggresiveness: int = 0) -> AudioSegment:
    """Wrap over tools for removing silence from audios

    Args:
        file_path (str): path of the audio file
        aggresiveness (int, optional): Parameter of the VAD. Defaults to 0.

    Returns:
        AudioSegment: audio without silences.
    """
    audio, sample_rate, num_channels, sample_width = read_wave(file_path)
    # Aggressiveness mode
    # An integer between 0 and 3.
    vad = webrtcvad.Vad(aggresiveness)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    # Segmenting the Voice audio and save it in list as bytes
    concataudio = [segment for segment in segments]
    joinedaudio = b"".join(concataudio)
    audio = AudioSegment(joinedaudio, sample_width=sample_width, frame_rate=sample_rate, channels=num_channels)
    return audio


def trim_audio(file_path: str, output_path: str, max_len: int = 12) -> AudioSegment:
    """Truncate audio duration

    Removes silences from audio and makes them max_len seconds long.
    Args:
        file_path (str): Input file path
        output_path (str): output file path
        max_len (int, optional): maximum audio duration in seconds. Defaults to 12.
    """
    audio = silenceRemoveWrapper(file_path, aggresiveness = 0)
    # Milliseconds to seconds
    if len(audio) / 1000  > max_len:
        audio = audio[:max_len*1000]
    else:
        pad_ms = max_len*1000 - len(audio)  # milliseconds of silence needed
        silence = AudioSegment.silent(duration=pad_ms)
        audio = audio + silence
    return audio
