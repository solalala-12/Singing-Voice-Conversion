n_epochs = 300
g_type = "gated_cnn"  # or "u_net"
# began = True  # True : Cycle-BeGan, False : CycleGan

sr = 16000  # sampling rate
n_features = 24 # Mceps coefficient 
# n_frames = 128   # fixed-length segment randomly 

mini_batch_size = 1  # mini_batch_size = 1 is better
generator_learning_rate = 0.0002
generator_learning_rate_decay = generator_learning_rate / 200000
discriminator_learning_rate = 0.0001
discriminator_learning_rate_decay = discriminator_learning_rate / 200000
sampling_rate = 16000
num_mcep = 24
frame_period = 5.0
## n_frames 128 is about 0.5 sec
n_frames = 512
lambda_cycle = 10
lambda_identity = 5
# 추가
gamma_A = 0.5
gamma_B = 0.5
lambda_k_A = 0.001
lambda_k_B = 0.001
balance_A = 0
balance_B = 0
# kta 초기값
k_t_A = 0
k_t_B = 0


dataset_A = "./data/train/A"
dataset_B = "./data/train/B"
test_dir = "./data/test"
direction = "B2A"   # or "B2A"

log_dir = "./log"
model_dir = "./model"
