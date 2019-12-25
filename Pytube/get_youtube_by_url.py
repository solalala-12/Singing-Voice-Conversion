#%% 
### 데이터 위에 주석 두줄 삭제해야 잘 됨 
import pandas as pd
import os 

# current_dir = os.getcwd() + '/Tobigs_music_project/pytube'
current_dir = 'Tobigs_music_project/pytube'
# Yumctu46ET0
# YTID = 'Yumctu46ET0' # guitar
# YTID = 'exH-dZr7--M' # guitar
YTID = '5c0go4TXs44' # piano
# NAME = 'guitar'
NAME = 'piano'
#%%
### get_data
from pytube import YouTube
import subprocess
import os
error_list = []

print(f'https://www.youtube.com/watch?v={YTID}')
try:
    yt = YouTube(f'https://www.youtube.com/watch?v={YTID}')
    vids = yt.streams.filter(file_extension='mp4').all() 
    # print(vids)
    # [<Stream: itag="140" mime_type="audio/mp4" abr="128kbps" acodec="mp4a.40.2">, <Stream: itag="249" mime_type="audio/webm" abr="50kbps" acodec="opus">, <Stream: itag="250" mime_type="audio/webm" abr="70kbps" acodec="opus">, <Stream: itag="251" mime_type="audio/webm" abr="160kbps" acodec="opus">]
    vids[0].download(f'{current_dir}/data')
    print(vids[0].default_filename)
    # ffmpeg -ss (start time) -i (direct video link) -t (duration needed) -c:v copy -c:a copy (destination file)

    ### Overwrite 안됨.
    subprocess.call(['ffmpeg','-ss','0','-i',
        os.path.join(f'{current_dir}/data/' ,vids[0].default_filename),
        os.path.join(f'{current_dir}/data/audio/',f'{vids[0].default_filename}.mp3')
    ])
except Exception as ex:
    print('에러가 발생 했습니다', ex)
    error_list.append(NAME)
# ffmpeg -ss 10 -i kk.mp4 -t 30 a.mp3
# ffmpeg -i guitar1.mp4 guitar1.mp3



# %%
