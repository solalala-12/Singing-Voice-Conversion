#%%
import process_data ### 폴더 안에 있는거 접근하려면 아래와 같이 해야함
# from TFMusicAudioEncoder import process_data
import math
import tensorflow as tf
import numpy as np
from functools import partial
import importlib

#%%
### mp3로 변환하기
# process_data.convert_mp3_to_wav()

#%%
LOSS_OUT_FILE = 'Epoch_Loss.txt'
# 파일 리스트 얻기
process_data.process_wav()

#%%
# Learning rate
lr = 0.0001

# L2 regularization
l2 = 0.0001

inputs = 12348
hidden_1_size = 8400
hidden_2_size = 3440
hidden_3_size = 2800

# Change the epochs variable to define the 
# number of times we iterate through all our batches
epochs = 1000

# Change the batch_size variable to define how many songs to load per batch
batch_size = 3

# Change the batches variable to change the number of batches you want per epoch
batches = 1

# Define our placeholder with shape [?, 12348]
X = tf.placeholder(tf.float32, shape=[None, inputs])
l2_regularizer = tf.contrib.layers.l2_regularizer(l2)

autoencoder_dnn = partial(tf.layers.dense, 
						activation = tf.nn.elu,
						kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
						kernel_regularizer=  tf.contrib.layers.l2_regularizer(l2))

hidden_1 = autoencoder_dnn(X, hidden_1_size)
hidden_2 = autoencoder_dnn(hidden_1, hidden_2_size)
hidden_4 = autoencoder_dnn(hidden_2, hidden_3_size)
hidden_5 = autoencoder_dnn(hidden_4, hidden_2_size)
outputs =  autoencoder_dnn(hidden_5, inputs, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs-X))
reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_loss)

optimizer = tf.train.AdamOptimizer(lr)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

#%%
def next_batch(c_batch, batch_size, sess):
	ch1_arr = []
	ch2_arr = []
	print(c_batch, batch_size)
	wav_arr_ch1, wav_arr_ch2, sample_rate = process_data.get_next_batch_original(c_batch, batch_size, sess)

	for sub_arr in wav_arr_ch1:
		batch_size_ch1 = math.floor(len(sub_arr)/inputs)
		sub_arr = sub_arr[:(batch_size_ch1*inputs)]
		ch1_arr.append(np.array(sub_arr).reshape(batch_size_ch1, inputs))

	for sub_arr in  wav_arr_ch2:
		batch_size_ch2 = math.floor(len(sub_arr)/inputs)
		sub_arr = sub_arr[:(batch_size_ch2*inputs)]
		ch2_arr.append(np.array(sub_arr).reshape(batch_size_ch2, inputs))

	return np.array(ch1_arr), np.array(ch2_arr), sample_rate



#%%
##### Run training
with tf.Session() as sess:
	init.run()
	### saver 만들기
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	print('processing...')
	sample_rate = process_data.file_to_data(sess)
	# ckpt_path = saver.restore(sess, "saved/train1")
	print('processing Done!')

	for epoch in range(epochs):
		epoch_loss = []
		print("Epoch: " + str(epoch))
		for i in range(batches):
			ch1_song, ch2_song, sample_rate = next_batch(i, batch_size, sess)
			total_songs = np.hstack([ch1_song, ch2_song])
			batch_loss = []

			for j in range(len(total_songs)):
				x_batch = total_songs[j]
				_, l = sess.run([training_op, loss], feed_dict={X:x_batch})
				batch_loss.append(l)
				print("Song loss: " + str(l))

			print("Curr Epoch: " + str(epoch) + " Curr Batch: " + str(i) + "/"+ str(batches))
			print("Batch Loss: " + str(np.mean(batch_loss)))
			epoch_loss.append(np.mean(batch_loss))
			
		print("Epoch Avg Loss: " + str(np.mean(epoch_loss)))

		if epoch % 100 == 0:
			ckpt_path = saver.save(sess, f'saved/train{epoch}')

	ckpt_path = saver.save(sess, "saved/train")
	ch1_song_new, ch2_song_new, sample_rate_new = next_batch(2, 1, sess)
	x_batch = np.hstack([ch1_song_new, ch2_song_new])[0]
	print("Sample rate: " + str(sample_rate_new))

	orig_song = []
	full_song = []
	evaluation = outputs.eval(feed_dict={X: x_batch})
	print("Output: " + str(evaluation))
	full_song.append(evaluation)
	orig_song.append(x_batch)

	# Merge the nested arrays
	orig_song = np.hstack(orig_song) 
	full_song = np.hstack(full_song) 

	# Compute and split the channels
	orig_song_ch1 = orig_song[:math.floor(len(orig_song)/2)] 
	orig_song_ch2 = orig_song[math.floor(len(orig_song)/2):] 
	full_song_ch1 = full_song[:math.floor(len(full_song)/2)] 
	full_song_ch2 = full_song[math.floor(len(full_song)/2):] 

	# Save both the untouched song and reconstructed song to the 'output' folder
	process_data.save_to_wav(full_song_ch1, full_song_ch2, sample_rate, orig_song_ch1, orig_song_ch2, epoch, 'output', sess)


# %%
##### test
sample_rate = 44100
epoch = 200
from scipy.fftpack import rfft, irfft
from tensorflow.contrib import ffmpeg
import json 

with tf.Session() as sess:
	init.run()
	### saver 만들기
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	
	# sample_rate = process_data.file_to_data(sess)
	ckpt_path = saver.restore(sess, "saved/train200")

	# 두번째 곡  받기
	ch1_song_new, ch2_song_new, sample_rate_new = next_batch(2, 1, sess)
	x_batch = np.hstack([ch1_song_new, ch2_song_new])[0]
	print("Sample rate: " + str(sample_rate_new)) 

	orig_song = []
	full_song = []
	evaluation = outputs.eval(feed_dict={X: x_batch})
	print("Output: " + str(evaluation)) 
	full_song.append(evaluation)
	orig_song.append(x_batch)
	print(len(full_song), len(full_song[0]))
	# Merge the nested arrays
	# 1*856 짜리인걸 한 array로 바꾼다!
	orig_song = np.hstack(orig_song)
	full_song = np.hstack(full_song)
	
	print('shape')
	print(orig_song.shape, full_song.shape)

	# Compute and split the channels
	orig_song_ch1 = orig_song[:math.floor(len(orig_song)/2)]
	orig_song_ch2 = orig_song[math.floor(len(orig_song)/2):]
	full_song_ch1 = full_song[:math.floor(len(full_song)/2)]
	full_song_ch2 = full_song[math.floor(len(full_song)/2):]
	print('next')
	print(orig_song_ch1.shape, full_song_ch1.shape)
	
	# Save both the untouched song and reconstructed song to the 'output' folder
	# process_data.save_to_wav(full_song_ch1, full_song_ch2, sample_rate, orig_song_ch1, orig_song_ch2, epoch, 'output', sess)


	audio_arr_ch1 = irfft(np.hstack(np.hstack(full_song_ch1)))
	audio_arr_ch2 = irfft(np.hstack(np.hstack(full_song_ch2)))

	print('reconstruced done')
	original_song_ch1 = irfft(np.hstack(np.hstack(orig_song_ch1)))
	original_song_ch2 = irfft(np.hstack(np.hstack(orig_song_ch2)))

	print('irfft done')
	original_song = np.hstack(np.array((original_song_ch1, original_song_ch2)).T)
	audio_arr = np.hstack(np.array((audio_arr_ch1, audio_arr_ch2)).T)

	print(original_song)
	w = np.linspace(0, sample_rate, len(audio_arr))
	w = w[0:len(audio_arr)]


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
	open('output/out.wav', 'wb').write(wav_file)
	open('output/wav_orig.wav', 'wb').write(wav_orig)





# %%
