import numpy as np
import librosa
from Utils.utils import *
from config import *
import pickle
import gc

def preprocess(dataset_A, dataset_B, sr = sr, n_features=n_features, frame_period = frame_period ) :
    
    print("Constructing MCEPs....")
    dataset_A = load_wavs(dataset_A, sr = sr)
    dataset_B = load_wavs(dataset_B, sr = sr)

    # f0s_A, _, _, _, coded_sps_A = world_encode_data(wavs = dataset_A, fs = sr, frame_period = frame_period, coded_dim = n_features)
    # f0s_B, _, _, _, coded_sps_B = world_encode_data(wavs = dataset_B, fs = sr, frame_period = frame_period, coded_dim = n_features)
    f0s_A, coded_sps_A = world_encode_data(wavs = dataset_A, fs = sr, frame_period = frame_period, coded_dim = n_features)
    f0s_B, coded_sps_B = world_encode_data(wavs = dataset_B, fs = sr, frame_period = frame_period, coded_dim = n_features)

    # 메모리 절약
    del dataset_A
    del dataset_B
    gc.collect()

    # transpose한거 덮어씌우기 메모리 절약하기 위함
    coded_sps_A = transpose_in_list(coded_sps_A)
    coded_sps_B = transpose_in_list(coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(coded_sps_A)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(coded_sps_B)
    
    # 리턴 값 Pickle로 저장하기
    print('Constructing Norm.npy....')
    with open("./data/A_norm.pickle", "wb") as fp:   
        pickle.dump(coded_sps_A_norm, fp)
    with open("./data/B_norm.pickle", "wb") as fp:   
        pickle.dump(coded_sps_B_norm, fp)

    if not os.path.exists(os.path.join("./data","mcep.npz")) :
        np.savez(os.path.join("./data",'mcep.npz'),
                 A_mean = coded_sps_A_mean, A_std = coded_sps_A_std,
                 B_mean = coded_sps_B_mean, B_std = coded_sps_B_std)

    print("Constructing Log_f0s....")
    if not os.path.exists(os.path.join("./data","logf0s.npz")) :
        logf0s_A_mean, logf0s_A_std = logf0_statistics(f0s_A)
        logf0s_B_mean, logf0s_B_std = logf0_statistics(f0s_B)
        np.savez(os.path.join("./data","logf0s.npz"),
                 A_mean = logf0s_A_mean, A_std = logf0s_A_std, 
                 B_mean = logf0s_B_mean, B_std = logf0s_B_std)
    print("Preprocessing Done!!!")    

    return coded_sps_A_norm, coded_sps_B_norm
        
# function ClickConnect() { 
#     // 백엔드를 할당하지 못했습니다. 
#     // GPU이(가) 있는 백엔드를 사용할 수 없습니다.가속기가 없는 런타임을 사용하시겠습니까?
#     // 취소 버튼을 찾아서 클릭
#     var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel");
#     buttons.forEach(function(btn) { btn.click(); });
#     console.log("1분마다 자동 재연결");
#     document.querySelector("colab-toolbar-button#connect").click();
# }
# setInterval(ClickConnect,1000*60);