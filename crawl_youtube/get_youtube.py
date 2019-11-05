#%% 
import pandas as pd
current_dir = 'sound/Tobigs_music_project/crawl_youtube'
class_df = pd.read_csv(f'{current_dir}/class_labels_indices.csv')
# 데이터에 ,로 구분되는게 있어서 ", "로 구분되기 해야한다.
balanced_df = pd.read_csv(f'{current_dir}/balanced_train_segments.csv',sep=", ")

#%%
baby_mid = class_df[class_df['display_name']=='Baby cry, infant cry']['mid']
snore_mid = class_df[class_df['display_name']=='Snoring']['mid']
# to_string 하면 index까지 같이 나옴
# balanced_df[balanced_df["positive_labels"].str.find(baby_mid.to_string())!=-1]
baby_df = balanced_df[balanced_df["positive_labels"].str.find(baby_mid.values[0])!=-1]
snore_df = balanced_df[balanced_df["positive_labels"].str.find(snore_mid.values[0])!=-1]

#%%
from pytube import YouTube
import subprocess
import os
## iter_rows 해줘야 함
for index, sr in snore_df.iterrows():
    print(sr)
    print(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    yt = YouTube(f'https://www.youtube.com/watch?v={sr["YTID"]}')
    vids = yt.streams.filter(file_extension='mp4').all() 
    # print(vids)
    # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
    vids[0].download(f'{current_dir}/data/snore/')
    print(vids[0].default_filename)
    # ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file)

    k = os.path.join(f'{current_dir}/data/snore/',"a.mp3")
    ### Overwrite 안됨.
    subprocess.call(['ffmpeg','-ss',str(int(sr["start_seconds"])),'-i',
        os.path.join(f'{current_dir}/data/snore/' ,vids[0].default_filename),
        '-t',str(int(sr["end_seconds"]-sr["start_seconds"])),
        os.path.join(f'{current_dir}/data/snore/',"a.mp3")
    ])

    # ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3
#%%
