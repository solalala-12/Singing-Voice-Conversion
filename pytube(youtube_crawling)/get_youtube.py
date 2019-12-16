#%% 
### 데이터 위에 주석 두줄 삭제해야 잘 됨 
import pandas as pd
current_dir = 'sound/Tobigs_music_project/pytube(youtube_crawling)'
# https://research.google.com/audioset/ 에서 얻음
class_df = pd.read_csv(f'{current_dir}/class_labels_indices.csv')
# 데이터에 ,로 구분되는게 있어서 ", "로 구분되기 해야한다.
balanced_df = pd.read_csv(f'{current_dir}/balanced_train_segments.csv',sep=", ")

#%%
flute_mid = class_df[class_df['display_name']=='Flute']['mid']
cello_mid = class_df[class_df['display_name']=='Cello']['mid']
# to_string 하면 index까지 같이 나옴
# balanced_df[balanced_df["positive_labels"].str.find(flute_mid.to_string())!=-1]
flute_df = balanced_df[balanced_df["positive_labels"].str.find(flute_mid.values[0])!=-1]
cello_df = balanced_df[balanced_df["positive_labels"].str.find(cello_mid.values[0])!=-1]

#%%
### flute
from pytube import YouTube
import subprocess
import os
## iter_rows 해줘야 함
from_check = False
error_list = []
for index, sr in flute_df.iterrows():    
    ## 해당번호까지 크롤 했으면 다음거 부터 크롤 해야 하니까 일단 임시로 만들기
    # if(from_check==False):
    #     if (index==6945):
    #         from_check = True
    #     continue
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
        vids = yt.streams.filter(file_extension='mp4').all() 
        # print(vids)
        # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
        vids[0].download(f'{current_dir}/data/flute/')
        print(vids[0].default_filename)
        # ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file)

        k = os.path.join(f'{current_dir}/data/flute/',"a.mp3")
        ### Overwrite 안됨.
        subprocess.call(['ffmpeg','-ss',str(int(sr["start_seconds"])),'-i',
            os.path.join(f'{current_dir}/data/flute/' ,vids[0].default_filename),
            '-t',str(int(sr["end_seconds"]-sr["start_seconds"])),
            os.path.join(f'{current_dir}/data/flute/audio/',f'{index}.mp3')
        ])

        subprocess.call(['mv',
            os.path.join(f'{current_dir}/data/flute/' ,vids[0].default_filename),
            os.path.join(f'{current_dir}/data/flute/',f'{index}.mp4')
        ])
        
    except Exception as ex:
        print('에러가 발생 했습니다', ex)
        error_list.append(index)
    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3
flute_err_pd = pd.DataFrame(error_list)
flute_err_pd.to_csv('flute_err.csv')

#%%
from pytube import YouTube
import subprocess
import os
## iter_rows 해줘야 함
from_check = False
error_list = []
for index, sr in cello_df.iterrows():    
    ## 해당번호까지 크롤 했으면 다음거 부터 크롤 해야 하니까 일단 임시로 만들기
    if(from_check==False):
        if (index==6945):
            from_check = True
        continue
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
        vids = yt.streams.filter(file_extension='mp4').all() 
        # print(vids)
        # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
        vids[0].download(f'{current_dir}/data/cello/')
        print(vids[0].default_filename)
        # ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file)

        k = os.path.join(f'{current_dir}/data/cello/',"a.mp3")
        ### Overwrite 안됨.
        subprocess.call(['ffmpeg','-ss',str(int(sr["start_seconds"])),'-i',
            os.path.join(f'{current_dir}/data/cello/' ,vids[0].default_filename),
            '-t',str(int(sr["end_seconds"]-sr["start_seconds"])),
            os.path.join(f'{current_dir}/data/cello/audio/',f'{index}.mp3')
        ])

        subprocess.call(['mv',
            os.path.join(f'{current_dir}/data/cello/' ,vids[0].default_filename),
            os.path.join(f'{current_dir}/data/cello/',f'{index}.mp4')
        ])
    except Exception as ex:
        print('에러가 발생 했습니다', ex)
        error_list.append(index)
    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3
#%%
cello_err_pd = pd.DataFrame(error_list)
cello_err_pd.to_csv('cellor_err.csv')

########################################################################
######################## FILE NAME CHANGE###############################
########################################################################
#%%
### flute
from pytube import YouTube
import subprocess
import os
## iter_rows 해줘야 함
from_check = False
error_list = []
for index, sr in flute_df.iterrows():    
    ## 해당번호까지 크롤 했으면 다음거 부터 크롤 해야 하니까 일단 임시로 만들기
    # if(from_check==False):
    #     if (index==6945):
    #         from_check = True
    #     continue
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
        vids = yt.streams.filter(file_extension='mp4').all() 
        # print(vids)
        # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
        print(vids[0].default_filename)

        # 파일 이름 바꾼다
        subprocess.call(['mv',
            os.path.join(f'{current_dir}/data/flute/' ,vids[0].default_filename),
            os.path.join(f'{current_dir}/data/flute/',f'{index}.mp4')
        ])
                
    except Exception as ex:
        print('에러가 발생 했습니다', ex)
        error_list.append(index)
    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3

#%%
#### CEllo
from pytube import YouTube
import subprocess
import os
## iter_rows 해줘야 함
from_check = False
error_list = []
for index, sr in cello_df.iterrows():    
    ## 해당번호까지 크롤 했으면 다음거 부터 크롤 해야 하니까 일단 임시로 만들기
    if(from_check==False):
        if (index==6945):
            from_check = True
        continue
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
        vids = yt.streams.filter(file_extension='mp4').all() 
        # print(vids)
        # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
        print(vids[0].default_filename)

        subprocess.call(['mv',
            os.path.join(f'{current_dir}/data/cello/' ,vids[0].default_filename),
            os.path.join(f'{current_dir}/data/cello/',f'{index}.mp4')
        ])

    except Exception as ex:
        print('에러가 발생 했습니다', ex)
        error_list.append(index)
    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3


# %%
