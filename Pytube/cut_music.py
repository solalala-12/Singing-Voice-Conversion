#%% 
#### http://melonicedlatte.com/machinelearning/2018/07/02/215933.html 에서 데이터 얻었음
import os 

current_dir = os.getcwd() + '/' # 현재 디렉토리 얻기
audio_file_dir = f'{current_dir}/multi-speaker-tacotron-tensorflow/datasets/son/audio'
file_list = os.listdir(audio_file_dir)

#%%
import subprocess
# file list 얻고 11초부터 ㅓ짜르기
for one_file in file_list:
    subprocess.call(['ffmpeg','-ss',str(11),'-i',
        os.path.join(audio_file_dir,one_file),
        os.path.join(f'{audio_file_dir}/cut/',one_file),
    ])    

 

# %%
