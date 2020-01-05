import tensorflow as tf
import numpy as np
import os, time
import pickle

from cyclegan import CycleGAN
from cycle_began import CycleBeGAN
from preprocess import *

from config import *

def train(input_A, input_B):

    # Config 파일에서 불러오는거라 할당할 때 에러가 떠서 전역변수로 설정 
    global k_t_A, k_t_B, lambda_k_A, lambda_k_B, balance_A, balance_B, checkpoint_every
    global identity_lambda, generator_lr, discriminator_lr

    # Make Directory
    if os.path.exists(log_dir) is False :
        os.mkdir(log_dir)
    if os.path.exists(model_dir) is False :
        os.mkdir(model_dir)

    # Preprocessing datasets
    # 있으면 기존에 있는걸로 실행
    if os.path.exists(os.path.join("./data", "A_norm.pickle")):
        print('Preprocess한 pickle파일 load!!!!!! \n새로운 파일로 train 할거면 data에 있는 pickle파일들 지우고 하세요~')
        with open(os.path.join("./data", "A_norm.pickle"), "rb") as fp:   # Unpickling
            A_norm = pickle.load(fp)
        with open(os.path.join("./data", "B_norm.pickle"), "rb") as fp:   # Unpickling
            B_norm = pickle.load(fp)
    else:
        A_norm, B_norm = preprocess(input_A,input_B)

    # 모델 불러오기 CycleBeGAN
    if began == True:
        model = CycleBeGAN(num_features = n_features, log_dir = log_dir)
    elif began == False:
        model = CycleGAN(num_features=n_features, g_type=g_type, log_dir=log_dir)

    # 저장한 모델 체크포인트 불러오기
    try:
        saved_global_step = model.load(model_dir)
        if saved_global_step is None:
            # check_point없으면 0부터 시작함
            saved_global_step = -1

    # 잘못 저장된 것 불러오면 에러
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise
        
    print("Start Training...")
    try:
        # 훈련 시작
        for epoch in range(saved_global_step + 1, n_epochs) :
            print("Epoch : %d " % epoch ) 
            start_time = time.time()
            train_A, train_B = sample_train_data(dataset_A=A_norm, dataset_B=B_norm,n_frames=n_frames) # random data
        
            n_samples = train_A.shape[0]

            # Cycle beGAN
            if began == True:
                # 일단 여기 뭔지 모르겠음
                for i in range(n_samples) : # mini_ batch_size = 1
                    n_iter = n_samples * epoch + i
                    if n_iter % 50 == 0:
                        
                        k_t_A = k_t_A + (lambda_k_A *balance_A)
                        if k_t_A > 1:
                            k_t_A = 1
                        if k_t_A < 0 :
                            k_t_A = 0
                        
                        k_t_B = k_t_B + (lambda_k_B *balance_B)
                        if k_t_B > 1.0:
                            k_t_B = 1.0
                        if k_t_B < 0. :
                            k_t_B = 0.
                
                    if n_iter > 10000 :
                        identity_lambda = 0
                    if n_iter > 200000 :
                        generator_lr = max(0, generator_lr - generator_lr_decay)
                        discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)
                
                    start = i
                    end = start + 1
                    # Loss 구하기
                    generator_loss, discriminator_loss, measure_A, measure_B, k_t_A, k_t_B, balance_A, balance_B = model.train(
                                    input_A=train_A[start:end], input_B=train_B[start:end], 
                                    lambda_cycle=lambda_cycle,
                                    lambda_identity=lambda_identity,
                                    gamma_A=gamma_A, gamma_B=gamma_B, lambda_k_A=lambda_k_A, lambda_k_B=lambda_k_B,
                                    generator_learning_rate=generator_learning_rate,
                                    discriminator_learning_rate=discriminator_learning_rate, 
                                    k_t_A = k_t_A, k_t_B = k_t_B)
            # CycleGAN
            elif began == False:
                for i in range(n_samples) : # mini_ batch_size = 1
                    n_iter = n_samples * epoch + i
                    
                    # 비갠은 이부분에서 k_t_A추가 
                    
                    if n_iter > 10000 :
                        identity_lambda = 0
                    if n_iter > 200000 :
                        generator_lr = max(0, generator_lr - generator_lr_decay)
                        discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)
                    
                    start = i
                    end = start + 1
                    
                    # 로스형식도 다름
                    generator_loss, discriminator_loss = model.train(input_A = train_A[start:end], 
                                                                    input_B = train_B[start:end], 
                                                                    cycle_lambda = cycle_lambda,
                                                                    identity_lambda = identity_lambda,
                                                                    generator_lr = generator_lr,
                                                                    discriminator_lr = discriminator_lr)
            
            
            end_time = time.time()
            epoch_time = end_time-start_time
            print("Generator Loss : %f, Discriminator Loss : %f, Time : %02d:%02d" % (generator_loss, discriminator_loss,(epoch_time % 3600 // 60),(epoch_time % 60 // 1)))

            
            # every 에폭마다 저장
            if epoch % checkpoint_every == 0:
                model.save(directory = model_dir, filename = "model", epoch=epoch)
                print(epoch, 'model save 완료')
    
    finally:
        print('잘못된 종료 또는 학습이 끝남 모델 저장')
        model.save(directory = model_dir, filename = "model", epoch=epoch)


if __name__ == "__main__" :
    train(input_A = dataset_A, input_B = dataset_B)
    print("Training Done!")


# 코랩 런타임 세션 종료 방지
# f12 눌러 console창에 주석풀고 입력해주세요~!
# function ClickConnect() {
#     // 백엔드를 할당하지 못했습니다.
#     // GPU이(가) 있는 백엔드를 사용할 수 없습니다. 가속기가 없는 런타임을 사용하시겠습니까?
#     // 취소 버튼을 찾아서 클릭 
#     var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel");
#     buttons.forEach(function(btn) { btn.click(); });
#     console.log("1분마다 자동 재연결");
#     document.querySelector("colab-toolbar-button#connect").click();
# }
# setInterval(ClickConnect,1000*60);

# 출처: https://bryan7.tistory.com/1077 [민서네집]
