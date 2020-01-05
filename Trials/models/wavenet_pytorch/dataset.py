from torch.utils.data import Dataset
from config import *
import numpy as np
import librosa
import random
import torch
import os
from utils import *


class SpeechDataset(Dataset):
    def __init__(self,
                 n_class,
                 slice_length,
                 frame_length,
                 frame_stride,
                 test_size,
                 device,
                 dataset_path='./dataset'):
        self.n_class = n_class
        self.slice_length = slice_length
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.test_size = test_size
        self.device = device
        self.dataset_path = dataset_path

    def create_dataset(self, max_files, prefix=None):
        lj_path = './LJSpeech-1.1'
        wav_names = []
        with open(lj_path + '/metadata.csv', encoding='utf-8') as f:
            for line in f:
                name = line.split('|')[0]
                if prefix and name.split('-')[0] != prefix:
                    continue
                wav_names.append(line.split('|')[0])

        random.seed(42)
        random.shuffle(wav_names)
        wav_names = wav_names[:max_files]

        count = 0
        x, cond = None, None
        for wav_name in wav_names:
            wav_path = lj_path + '/wavs/' + wav_name + '.wav'

            # calculate log mel spectrum
            y, sr = librosa.core.load(wav_path)
            input_nfft = int(round(sr * self.frame_length))
            input_stride = int(round(sr * self.frame_stride))
            s = librosa.feature.melspectrogram(y=y, n_mels=N_MELS, n_fft=input_nfft, hop_length=input_stride)
            s = librosa.core.power_to_db(s, ref=np.max)

            # scale to [0, 1]
            s /= 80.0

            new_x = quantize_signal(y, self.n_class)
            new_cond = self.time_resolution(s, y.shape[0])
            print(wav_name, 'processed.')

            if x is None:
                x, cond = new_x, new_cond
            else:
                x, cond = np.concatenate((x, new_x)), np.column_stack((cond, new_cond))

            for i in range(0, x.shape[0]-self.slice_length+1, self.slice_length):
                np.save(self.dataset_path + '/x_' + str(count) + '.npy', x[i:i+self.slice_length])
                np.save(self.dataset_path + '/cond_' + str(count) + '.npy', cond[:, i:i+self.slice_length])
                count += 1

            if x.shape[0] % self.slice_length == 0:
                x, cond = None, None
            else:
                x, cond = x[-x.shape[0]%self.slice_length:], cond[:, -x.shape[0] % self.slice_length:]

    def init_dataset(self, test_mode):
        self.test_mode = test_mode
        file_list = os.listdir(self.dataset_path)
        tot = len(file_list)//2
        self.test_length = int(tot*self.test_size)
        self.train_length = tot-self.test_length

    def __len__(self):
        return self.test_length if self.test_mode else self.train_length

    def __getitem__(self, idx):
        if self.test_mode:
            idx += self.train_length
        x = np.load(self.dataset_path+'/x_'+str(idx)+'.npy')
        cond = np.load(self.dataset_path+'/cond_'+str(idx)+'.npy')

        # one hot encoding
        embedded_x = np.zeros((self.n_class, x.shape[0]))
        embedded_x[x, np.arange(x.shape[0])] = 1

        return torch.tensor(embedded_x[:, :-1], dtype=torch.float, device=self.device),\
               torch.tensor(x[1:], dtype=torch.long, device=self.device),\
               torch.tensor(cond[:, :-1], dtype=torch.float, device=self.device)

    def time_resolution(self, cond, target_length):
        z = np.zeros((cond.shape[0], target_length))
        repeated_cond = np.repeat(cond, target_length//cond.shape[1], axis=1)
        z[:, :repeated_cond.shape[1]] = repeated_cond
        return z

