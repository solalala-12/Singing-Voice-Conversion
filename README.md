# 🎵 Singing Voice Conversion Using CycleGAN 🎶

## 투빅스 11기 & 12기 음성 프로젝트 


### Preprocessing
- [Splitiing](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/time_cutting.ipynb)
  <br>짧은 시간 단위로 학습이 필요할 때에는 노래를 원하는 초 단위로 잘라줍니다.
- [Voice Seperation](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/Vocal_Separation_JH.ipynb)
  <br>Vocal이 있는 음원 파일과 Vocal이 없는 Inst 파일을 input으로 넣으면 Vocal만 extract 되도록 합니다.
   <br> Vocal만 extract된 음원 예시: https://bit.ly/2SZQJdX
- [Data Autmentation](https://github.com/sora-12/Tobigs_music_project/blob/master/Preprocessing/data_augmentation.ipynb)
   <br> Dataset을 증가시키기 위해
   ![preprocessing 그림](https://github.com/sora-12/Tobigs_music_project/blob/master/aug_pic.png)
   3가지 방법을 사용하였습니다.<br>
   white noise나 stretch 방법은 오히려 잡음이 껴서 성능을 저하시켰고,<br> 위에 그림에 나온 방법들은 Conversion 성능향상에 영향을 끼쳤습니다.

---

> ### Model
- Feature Extraction
<br> 보통 음성분야에서는 `MFCC`를 쓰는데, 우리 코드에선 `MCEP`을 사용합니다.<br>
그 이유는 `MCEP`이 보다 많은 정보를 포함하고 있어, Vocal의 음색, 억양 등을 담을 수 있습니다.<br>
Preprocessing과정 후에 A.pickle, B.pickle, logf0.npz, mcep.npz 파일이 만들어지고 다음에 같은 dataset으로 train 할 때에는 이를 활용합니다.

- modeling
<br> `CycleGan`, `Cycle Began` 모델을 활용하여 Vocal style을 바꾸었습니다. 모델과 관련한 코드는 아래 reference를 활용했습니다. `CycleBegan`이 더 깔끔한 음질의 결과를 보였으나, CycleGan이 조금 더 robust한 보컬 변화가 있었습니다.

---

### How to Run it!
[jupyter notebook](https://github.com/sora-12/Tobigs_music_project/blob/master/Vocal_Style_Transfer.ipynb)을 코랩에서 열어서 실행하면 됩니다.<br>
실행방법이나 설정방법은 노트북 파일에 자세하게 쓰여져 있으니, 참고하셔서 실행해주시기 바랍니다.<br>
코드를 수정하시거나 직접 다운받아 사용하실 분은 [코드 폴더](https://github.com/sora-12/Singing-Voice-Conversion/tree/master/Vocal_Style_Transfer)를 보시면 됩니다. 

---

### Result(Pickin good example)<br>
[거미 노래를 아이유 목소리로 바꾼 파일](https://drive.google.com/file/d/1K91OiGdTp8S6-mM0UgnZ0SXqEZBMhDDz/view?usp=sharing) <br><br>
[케이윌 노래를 10cm 권정열 목소리로 바꾼 파일](https://drive.google.com/file/d/1djsn1H-AdOCq0EYc3q9w0Zn8kvAaFGxS/view?usp=sharing)

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
