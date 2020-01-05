from dataset import SpeechDataset
from model import Wavenet
from config import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import librosa
from utils import *
import logging
import os


class Trainer(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Wavenet(N_CLASS, HIDDEN_CHANNELS, COND_CHANNELS, N_REPEAT, N_LAYER, self.device)

        # training state
        self.sample_count = 0
        self.tot_steps = 0

        logger = logging.getLogger("my")
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s; %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        stream_hander = logging.StreamHandler()
        file_handler = logging.FileHandler('trainer.log')
        stream_hander.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream_hander)
        self.logger.addHandler(file_handler)


    def save_model(self):
        dic = {
            'state': self.model.state_dict(),
            'sample_count': self.sample_count,
            'tot_steps': self.tot_steps
        }
        torch.save(dic, 'save/model_{0}.tar'.format(self.tot_steps))
        torch.save(dic, 'save/latest_model.tar')
        self.logger.info('model_{0} saved'.format(self.tot_steps))

    def load_model(self, path='save/latest_model.tar'):
        if not os.path.isfile(path):
            return
        dic = torch.load(path)
        self.model.load_state_dict(dic['state'])
        self.sample_count = dic['sample_count']
        self.tot_steps = dic['tot_steps']

    def create_dataset(self):
        dataset = SpeechDataset(N_CLASS, SLICE_LENGTH, FRAME_LENGTH, FRAME_STRIDE, TEST_SIZE, self.device)
        dataset.create_dataset(MAX_FILES, FILE_PREFIX)

    def train(self):
        self.model.train()
        dataset = SpeechDataset(N_CLASS, SLICE_LENGTH, FRAME_LENGTH, FRAME_STRIDE, TEST_SIZE, self.device)
        dataset.init_dataset(test_mode=False)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        for epoch in range(MAX_EPOCHS):
            for i, data in enumerate(data_loader):
                x, y, cond = data
                pred_y = self.model(x, cond)
                loss = F.cross_entropy(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), MAX_NORM)
                optimizer.step()

                if i % PRINT_FREQ == 0:
                    self.logger.info('epoch: %d, step:%d, tot_step:%d, loss: %f'%(epoch, i, self.tot_steps, loss.item()))
                if i % VALID_FREQ == 0:
                    self.validate()
                    self.model.eval()

                self.tot_steps += 1
                if self.tot_steps % 100 == 0:
                    self.save_model()

    def validate(self):
        self.model.eval()
        dataset = SpeechDataset(N_CLASS, SLICE_LENGTH, FRAME_LENGTH, FRAME_STRIDE, TEST_SIZE, self.device)
        dataset.init_dataset(test_mode=True)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        res = []
        for i, data in enumerate(data_loader):
            if i == MAX_VALID:
                break
            x, y, cond = data
            pred_y = self.model(x, cond)
            loss = F.cross_entropy(pred_y.squeeze(), y.squeeze())
            res.append(loss.item())
        self.logger.info('valid loss: '+str(sum(res)/len(res)))

    def generate(self):
        self.model.eval()
        dataset = SpeechDataset(N_CLASS, SLICE_LENGTH, FRAME_LENGTH, FRAME_STRIDE, TEST_SIZE, self.device)
        dataset.init_dataset(test_mode=True)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for i, data in enumerate(data_loader):
            if i == MAX_GENERATE:
                break
            _, _, cond = data
            res = self.model.generate(cond, MAX_GENERATE_LENGTH)
            res = dequantize_signal(res, N_CLASS)
            for j in range(res.shape[0]):
                librosa.output.write_wav('./samples/sample%d.wav'%(self.sample_count), res[j], SAMPLE_RATE)
                self.sample_count += 1

if __name__ == '__main__':
    tr = Trainer()
    #tr.create_dataset()
    tr.load_model()
    tr.generate()
