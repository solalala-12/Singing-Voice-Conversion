#%%
#### tensorflow 1.14에서 돌아감
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.contrib import ffmpeg
from scipy.fftpack import rfft, irfft
from glob import iglob
from pydub import AudioSegment
import os 

#%%
# current_dir = os.getcwd()
DATA_FILES_MP3 = f'/content/drive/My Drive/tobigs/dnn_auto_encoder/music_makes_one'
DATA_FILES_WAV = f'/content/drive/My Drive/tobigs/dnn_auto_encoder/music_makes_one'
# DATA_FILES_MP3 = f'{current_dir}/audio'
# DATA_FILES_WAV = f'{current_dir}/audio_wav'

file_arr = []
curr_batch = 0
file_data_ch1 = []
file_data_ch2 = []

def convert_mp3_to_wav():
	index = 0
	# iglob는 재귀적으로도 파일 찾아줄 수 있게 한다.

	for file in iglob(DATA_FILES_MP3 + '/*.mp3'):
		mp3_to_wav = AudioSegment.from_mp3(file)
		mp3_to_wav.export(DATA_FILES_WAV+'/'+str(index)+'.wav', format='wav')
		index += 1

def process_wav():
	file_range = 0
	for file in iglob(DATA_FILES_WAV +'/*.wav'):
		file_range += 1
      
		file_arr.append(file)

def file_to_data(sess):
	### 계속 파일 로드하는거 보다는 미리 저장해두는게 더 빠를수도 있겠따.
	for idx in range(len(file_arr)):
		audio_binary = tf.read_file(file_arr[idx])
		wav_decoder = audio_ops.decode_wav(
			audio_binary,
			desired_channels=2)
		# 변수값 얻어오는 방법(sess.run([a,b]))
		sample_rate, audio = sess.run([wav_decoder.sample_rate, wav_decoder.audio])
		audio = np.array(audio)

		if len(audio[:, 0]) != 5292000: 
			continue

		file_data_ch1.append(rfft(audio[:,0]))
		file_data_ch2.append(rfft(audio[:,1]))
		print("Returning File: " + file_arr[idx])
	return sample_rate

def get_next_batch(curr_batch, songs_per_batch, sample_rate,sess):
	wav_arr_ch1 = []
	wav_arr_ch2 = []
	if (curr_batch) >= (len(file_arr)):
		curr_batch = 0

	start_position = curr_batch * songs_per_batch
	end_position = start_position + songs_per_batch
	for idx in range(start_position, end_position):
		wav_arr_ch1.append(file_data_ch1[idx])
		wav_arr_ch2.append(file_data_ch2[idx])
		# print("Returning File: " + file_arr[idx])
  # print("Returning File")
	return wav_arr_ch1, wav_arr_ch2, sample_rate

def get_next_batch_original(curr_batch, songs_per_batch, sess):
	wav_arr_ch1 = []
	wav_arr_ch2 = []
	if (curr_batch) >= (len(file_arr)):
		curr_batch = 0

	start_position = curr_batch * songs_per_batch
	end_position = start_position + songs_per_batch
	for idx in range(start_position, end_position):
		audio_binary = tf.read_file(file_arr[idx])
		wav_decoder = audio_ops.decode_wav(
			audio_binary,
			desired_channels=2)
		sample_rate, audio = sess.run([wav_decoder.sample_rate, wav_decoder.audio])
		audio = np.array(audio)

		if len(audio[:, 0]) != 5292000: 
			continue

		wav_arr_ch1.append(rfft(audio[:,0]))
		wav_arr_ch2.append(rfft(audio[:,1]))
		print("Returning File: " + file_arr[idx])
		

	return wav_arr_ch1, wav_arr_ch2, sample_rate

def save_to_wav(audio_arr_ch1, audio_arr_ch2, sample_rate, original_song_ch1, original_song_ch2, idty, folder, sess):
	audio_arr_ch1 = irfft(np.hstack(np.hstack(audio_arr_ch1)))
	audio_arr_ch2 = irfft(np.hstack(np.hstack(audio_arr_ch2)))

	original_song_ch1 = irfft(np.hstack(np.hstack(original_song_ch1)))
	original_song_ch2 = irfft(np.hstack(np.hstack(original_song_ch2)))
	
	original_song = np.hstack(np.array((original_song_ch1, original_song_ch2)).T)
	audio_arr = np.hstack(np.array((audio_arr_ch1, audio_arr_ch2)).T)

	print(original_song)
	w = np.linspace(0, sample_rate, len(audio_arr))
	w = w[0:len(audio_arr)]
	plt.figure(1)

	plt.plot(w, original_song)
	plt.savefig(str(folder) + '/original.png')
	plt.plot(w, audio_arr)
	plt.xlabel('sample')
	plt.ylabel('amplitude')
	plt.savefig(str(folder) + '/compressed' + str(idty) + '.png')
	plt.clf()	

	cols = 2
	rows = math.floor(len(audio_arr)/2)
	audio_arr = audio_arr.reshape(rows, cols)
	original_song = original_song.reshape(rows, cols)

	wav_encoder = ffmpeg.encode_audio(
		audio_arr, file_format='wav', samples_per_second=sample_rate)
	wav_encoder_orig = ffmpeg.encode_audio(
		original_song, file_format='wav', samples_per_second=sample_rate)

	wav_file = sess.run(wav_encoder)
	wav_orig = sess.run(wav_encoder_orig)
	open(str(folder)+'/out.wav', 'wb').write(wav_file)
	open(str(folder)+'/wav_orig.wav', 'wb').write(wav_orig)



