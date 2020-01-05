### test ###

import tensorflow as tf
import numpy as np
import os

import librosa
from librosa import load
from librosa.output import write_wav
from librosa.util import find_files

from cycle_began import CycleBeGAN
from cyclegan import CycleGAN

from Utils.utils import *
from config import *


def test(direction=direction, model_dir=model_dir, test_dir=test_dir, sr=sr, n_features=n_features, frame_period=frame_period) :
    
    outputs_dir = "./sample"
    
    if began == True:      
        model = CycleBeGAN(num_features=n_features,mode="test")
    elif began == False:
        model = CycleGAN(num_features=n_features,mode="test")
    
    model.load(os.path.join("./model"))

    mcep = np.load(os.path.join("./data/", 'mcep.npz'))
    mcep_mean_A = mcep['A_mean']
    mcep_std_A = mcep['A_std']
    mcep_mean_B = mcep['B_mean']
    mcep_std_B = mcep['B_std']

    logf0s = np.load(os.path.join("./data/", 'logf0s.npz'))
    logf0s_mean_A = logf0s['A_mean']
    logf0s_std_A = logf0s['A_std']
    logf0s_mean_B = logf0s['B_mean']
    logf0s_std_B = logf0s['B_std']
    
    if not os.path.exists(outputs_dir) :
        os.mkdir(outputs_dir)
    
    file_list = librosa.util.find_files(test_dir,ext="wav")
    
    for file in file_list :
        wav,_ = load(file, sr=sr)
        wav = wav_padding(wav = wav, sr = sr, frame_period = frame_period, multiple = 4)
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = sr, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = sr, dim = n_features)
        coded_sp_transposed = coded_sp.T
        
        if direction == "A2B" :
            f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_A, std_log_src = logf0s_std_A, mean_log_target = logf0s_mean_B, std_log_target = logf0s_std_B)
            #f0_converted = f0
            coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A
            coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = direction)[0]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
        else : # B2A
            f0_converted = pitch_conversion(f0 = f0, mean_log_src = logf0s_mean_B, std_log_src = logf0s_std_B, mean_log_target = logf0s_mean_A, std_log_target = logf0s_std_A)
            #f0_converted = f0
            coded_sp_norm = (coded_sp_transposed - mcep_mean_B) / mcep_std_B
            coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = direction)[0]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_A + mcep_mean_A
            
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, fs = sr)
        wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, fs = sr, frame_period = frame_period)
        write_wav(os.path.join(outputs_dir, os.path.basename(file)), wav_transformed, sr)


if __name__ == "__main__" :
    test(direction = direction)
    print("Test.py Sample Create Completed!")