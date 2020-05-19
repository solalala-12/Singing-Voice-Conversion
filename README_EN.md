# 🎵 Singing Voice Conversion Using CycleGAN 🎶

## Tobig's Audio Project

### Preprocessing
- [Splitiing](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/time_cutting.ipynb)
  <br> If you need to train data in a short period of time, split the audio files  in seconds you want.
- [Voice Seperation](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/Vocal_Separation_JH.ipynb)
  <br> Only voice data will be extracted  from music files and inst files.
   <br> example : https://bit.ly/2SZQJdX
- [Data Autmentation](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/data_augmentation.ipynb)
   <br> 3 Ways to augmentation audio files
   ![preprocessing pic](https://github.com/sora-12/Tobigs_music_project/blob/master/aug_pic.png)
  

---

> ### Model
- Feature Extraction
<br>  we use MCEP in our code.  
The reason is that MCEP contains more information, so you can get more details like  Vocal's tone, intonation, etc.  
	After the Preprocessing process, A.pickle, B.pickle, logf0.npz, and mcep.npz files are created, and this is used for the next train with the same dataset.

- modeling
<br> By Using CycleGan' and 'Cycle Began' models, we changed the Vocal style. The code related to the model used the reference below. 'CycleBegan' showed cleaner sound quality results, but CycleGan had a more robust vocal change.


### How to Run it!

Please refer to the [jupyter notebook](https://github.com/sora-12/Tobigs_music_project/blob/master/Vocal_Style_Transfer.ipynb) for detailed instructions on how to do it or how to set it up. <br>
If you want to modify the code or download it yourself, go to [code folder](https://github.com/sora-12/Singing-Voice-Conversion/tree/master/Vocal_Style_Transfer).

---

### Result(Pickin good example)<br>
[Gummy to IU](https://drive.google.com/file/d/1K91OiGdTp8S6-mM0UgnZ0SXqEZBMhDDz/view?usp=sharing) <br><br>
[Kwill to 10cm ](https://drive.google.com/file/d/1djsn1H-AdOCq0EYc3q9w0Zn8kvAaFGxS/view?usp=sharing)

---

### Members of this Project

- [소라](https://github.com/sora-12)
- [유민](https://github.com/rhawl97)
- [승호](https://github.com/smothly)
- [진혁](https://github.com/ParkJinHyeock)
- [혜인](https://github.com/hyennneee)
- [민호](https://github.com/dizwe)

---

### Reference
- https://github.com/eliceio/vocal-style-transfer/tree/master/Singing-Style-transfer
- https://github.com/NamSahng/SingingStyleTransfer
- https://github.com/serereuk/Voice_Converter_CycleGAN
- Takuhiro Kaneko, Hirokazu Kameoka. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. 2017. (Voice Conversion CycleGAN)


