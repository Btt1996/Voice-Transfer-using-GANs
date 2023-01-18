import librosa
import numpy as np

def preprocess_audio(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path)

    # Convert audio to a format that can be used for training
    audio = librosa.effects.harmonic(audio)

    # Extracting features from the audio such as Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(audio, sr=sr)

    # Pitch shifting the audio recordings
    shifted_audio = librosa.effects.pitch_shift(audio, sr, n_steps=2)

    return audio, mfccs, shifted_audio
