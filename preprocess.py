# -*- coding: utf-8 -*-
# #/usr/bin/python3

'''
fangshuming519@gmail.com. 
https://www.github.com/FonzieTree/deepvoice3
'''
import numpy as np
import librosa
from hyperparams import Hyperparams as hp
import os


def get_spectrograms(sound_file):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # Sequence length
    done = np.ones_like(mel[0, :]).astype(np.int32)

    # to decibel
    mel = librosa.amplitude_to_db(mel)
    mag = librosa.amplitude_to_db(mag)

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 0, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    # Padding
    if mel.shape[0]<= hp.Ty:
        mel = np.concatenate((mel, np.zeros((hp.Ty - mel.shape[0], mel.shape[1]))), axis = 0)
        done = np.concatenate((done, np.zeros((hp.Ty - done.shape[0]))), axis = 0)
        mag = np.concatenate((mag, np.zeros((hp.Ty - mag.shape[0], mag.shape[1]))), axis = 0)
    else:
        mel = mel[:hp.Ty, :]
        done = done[:hp.Ty, :]
        mag = mag[:hp.Ty, :]
    return mel, done, mag

if __name__ == "__main__":
    wav_folder = os.path.join(hp.data, 'wavs')
    mel_folder = os.path.join(hp.data, 'mels')
    done_folder = os.path.join(hp.data, 'dones')
    mag_folder = os.path.join(hp.data, 'mags')

    for folder in (mel_folder, done_folder, mag_folder):
        if not os.path.exists(folder): os.mkdir(folder)

    files = os.listdir(wav_folder)
    for f in files:
        mel, done, mag = get_spectrograms(os.path.join(wav_folder, f))  # (n_mels, T), (1+n_fft/2, T) float32
        #print(mel.shape, done.shape, mag.shape)
        np.save(os.path.join(mel_folder, f.replace(".wav", ".npy")), mel)
        np.save(os.path.join(done_folder, f.replace(".wav", ".npy")), done)
        np.save(os.path.join(mag_folder, f.replace(".wav", ".npy")), mag)
