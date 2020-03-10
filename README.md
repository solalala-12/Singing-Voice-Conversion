# 🎵 Singing Voice Conversion Using CycleGAN 🎶

## 투빅스 11기 & 12기 음성 프로젝트 


### Preprocessing
- [Splitiing](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/time_cutting.ipynb)
  <br>짧은 시간 단위로 학습이 필요할 때에는 노래를 원하는 초 단위로 잘라줍니다.
- [Voice Seperation](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/Vocal_Separation_JH.ipynb)
  <br>Vocal이 있는 음원 파일과 Vocal이 없는 Inst 파일을 input으로 넣으면 Vocal만 extract 되도록 합니다.
   <br> Vocal만 extract된 음원 예시: https://bit.ly/2SZQJdX
- [Data Autmentation](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/data_augmentation.ipynb)
   <br> dataset을 다양하게 확보하기 위하여 뒤집거나 반대로 재생되는 파일을 얻습니다. data augmentation을 한 파일을 input 데이터로 함께 사용하니 noise가 덜해지고 conversion 효과가 더 좋아졌습니다.
![preprocessing 그림](https://github.com/sora-12/Tobigs_music_project/blob/master/aug_pic.png)

### Model
- Feature Extraction
<br> MFCC 보다 MCEP이 단어의 억양을 잘 표현한다고 합니다. 때문에 MCEP을 이용하여 wav 파일을 train할 수 있는 수치 파일로 바꾸었습니다. 한번 processing 되면 A.pickle, B.pickle, logf0.npz, mcep.npz 파일이 만들어지고 다음에 같은 dataset으로 train 할 때에는 이를 활용합니다.
- modeling
<br> CycleGan, Cycle Began 모델을 활용하여 Vocal style을 바꾸었습니다. 모델과 관련한 코드는 아래 reference를 활용했습니다. CycleBegan이 더 깔끔한 음질의 결과를 보였으나, CycleGan이 조금 더 robust한 보컬 변화가 있었습니다.

### How to Run it!
[jupyter notebook](https://github.com/sora-12/Tobigs_music_project/blob/master/Vocal_Style_Transfer.ipynb)을 코랩에서 열어서 실행하면 됩니다. Feature Extracition 과정이 train을 시작할 때 꽤 오래걸립니다! 중간에 colab에서 cell을 중지시키면 현재 epoch의 model을 저장하도록 했습니다.

### Example(Pickin good example)
[거미 노래를 아이유 목소리로 바꾼 파일](https://drive.google.com/file/d/1K91OiGdTp8S6-mM0UgnZ0SXqEZBMhDDz/view?usp=sharing) <br>
[케이윌 노래를 10cm 권정열 목소리로 바꾼 파일](https://drive.google.com/file/d/1djsn1H-AdOCq0EYc3q9w0Zn8kvAaFGxS/view?usp=sharing)

---
### Members of this Project

- [소라](https://github.com/sora-12)
- [유민](https://github.com/rhawl97)
- [승호](https://github.com/smothly)
- [진혁](https://github.com/ParkJinHyeock)
- [혜인](https://github.com/hyennneee)
- [민호](https://github.com/dizwe)

### Reference
- https://github.com/eliceio/vocal-style-transfer/tree/master/Singing-Style-transfer
- https://github.com/NamSahng/SingingStyleTransfer
