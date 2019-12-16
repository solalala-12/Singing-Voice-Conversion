#%% 
### 데이터 위에 주석 두줄 삭제해야 잘 됨 
### YTID에도 #없앧기
import pandas as pd
current_dir = 'sound/Tobigs_music_project/pytube(youtube_crawling)'
class_df = pd.read_csv(f'{current_dir}/class_labels_indices.csv')
# 데이터에 ,로 구분되는게 있어서 ", "로 구분되기 해야한다.
# # https://research.google.com/audioset/ 에서 얻음
unbalanced_df = pd.read_csv(f'{current_dir}/unbalanced_train_segments.csv',sep=", ")

#%%
flute_mid = class_df[class_df['display_name']=='Flute']['mid']
cello_mid = class_df[class_df['display_name']=='Cello']['mid']
# to_string 하면 index까지 같이 나옴
# unbalanced_df[unbalanced_df["positive_labels"].str.find(flute_mid.to_string())!=-1]
flute_df = unbalanced_df[unbalanced_df["positive_labels"].str.find(flute_mid.values[0])!=-1]
cello_df = unbalanced_df[unbalanced_df["positive_labels"].str.find(cello_mid.values[0])!=-1]

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
    if(from_check==False):
        if (index==84845):
            from_check = True
        continue
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
        vids = yt.streams.filter(file_extension='mp4').all() 
        # print(vids)
        # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
        vids[0].download(f'{current_dir}/data/flute/unbalanced/')
        print(vids[0].default_filename)
        # ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file)

        ### Overwrite 안됨.
        subprocess.call(['ffmpeg','-ss',str(int(sr["start_seconds"])),'-i',
            os.path.join(f'{current_dir}/data/flute/unbalanced/' ,vids[0].default_filename),
            '-t',str(int(sr["end_seconds"]-sr["start_seconds"])),
            os.path.join(f'{current_dir}/data/flute/unbalanced/audio/',f'{index}.mp3')
        ])

        subprocess.call(['mv',
            os.path.join(f'{current_dir}/data/flute/unbalanced/' ,vids[0].default_filename),
            os.path.join(f'{current_dir}/data/flute/unbalanced/',f'{index}.mp4')
        ])
    except Exception as ex:
        print('에러가 발생 했습니다', ex)
        error_list.append(index)
    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3

#%%
flute_err_pd = pd.DataFrame(error_list)
flute_err_pd.to_csv('flute_err_unba.csv')

#%%
##### Cello
from pytube import YouTube
import subprocess
import os
## iter_rows 해줘야 함
from_check = False
error_list = []
for index, sr in cello_df.iterrows():    
    ## 해당번호까지 크롤 했으면 다음거 부터 크롤 해야 하니까 일단 임시로 만들기
    if(from_check==False):
        if (index==87526):
            from_check = True
        continue
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
        vids = yt.streams.filter(file_extension='mp4').all() 
        # print(vids)
        # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
        vids[0].download(f'{current_dir}/data/cello/unbalanced/')
        print(vids[0].default_filename)
        # ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file)

        ### Overwrite 안됨.

        subprocess.call(['ffmpeg','-ss',str(int(sr["start_seconds"])),'-i',
            os.path.join(f'{current_dir}/data/cello/unbalanced/' ,vids[0].default_filename),
            '-t',str(int(sr["end_seconds"]-sr["start_seconds"])),
            os.path.join(f'{current_dir}/data/cello/unbalanced/audio/',f'{index}.mp3')
        ])

        subprocess.call(['mv',
            os.path.join(f'{current_dir}/data/cello/unbalanced/' ,vids[0].default_filename),
            os.path.join(f'{current_dir}/data/cello/unbalanced/',f'{index}.mp4')
        ])
    except Exception as ex:
        print('에러가 발생 했습니다', ex)
        error_list.append(index)
    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3
#%%
cello_err_pd = pd.DataFrame(error_list)
cello_err_pd.to_csv('cellor_err_unba.csv')

