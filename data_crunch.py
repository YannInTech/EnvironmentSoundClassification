import torch
from torch.utils import data
import transforms
import torchvision
import torchaudio
import os
import config


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class MyDataset(data.Dataset):
    
	def __init__(self, train=True):
		self.root = './data/'
		self.train = train
        
		self.file_paths = []
        
		if train:
			file_names = os.listdir(self.root + 'train/')
                
			for name in file_names:
				if name.endswith('.wav'):
					self.file_paths.append(os.path.join('train/',name))

		else:
			file_names = os.listdir(self.root + 'test/')
                
			for name in file_names:
				if name.endswith('.wav'):
					self.file_paths.append(os.path.join('test/',name))
        

		if self.train:
			self.wave_transforms = torchvision.transforms.Compose([transforms.RandomScale(max_scale = 1.25), 
                                                                   transforms.RandomPadding(out_len = 22050*5),
                                                                   transforms.RandomCrop(out_len = 22050*5)])
             
			self.spec_transforms = torchvision.transforms.Compose([transforms.FrequencyMask(max_width = config.freq_masks_width, numbers = config.freq_masks),
                                                                   transforms.TimeMask(max_width = config.time_masks_width, numbers = config.time_masks)])
            
            
		else: # for test
			self.wave_transforms = torchvision.transforms.Compose([transforms.RandomPadding(out_len = 22050*5),
                                                                   transforms.RandomCrop(out_len = 22050*5)])

    
	def __len__(self):
		
		return len(self.file_paths) 
    

	def __getitem__(self, index):
		
		file_path = self.file_paths[index]  
		path = self.root + file_path

		class_id = int(path.split('_')[1])

		wave, rate = torchaudio.load(path)

		if wave.shape[0] > 1:
			wave = (wave[0, : ] + wave[1, : ]) / 2 
		
		if abs(wave.max()) > 1.0:
			wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
		wave = wave* 32768.
        
		# remove silent sections
		start = wave.nonzero().min()
		end = wave.nonzero().max()
		wave = wave[:, start:end+1]
        
		wave = self.wave_transforms(wave)
	
		transform_melspec=torchaudio.transforms.MelSpectrogram(sample_rate=rate, n_mels=128, n_fft=1024, hop_length=512)
		s=transform_melspec(wave)

		transform_log=torchaudio.transforms.AmplitudeToDB()
		log_s=transform_log(s)
        
		if self.train:
			log_s = self.spec_transforms(log_s)
        
		# creating 3 channels by copying log_s1 3 times 
		spec = torch.cat((log_s, log_s, log_s), dim=0)
        	
		return spec, class_id 
    

def create_generators():
	
	train_dataset = MyDataset(train=True)
	test_dataset = MyDataset(train=False)
	train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=4, drop_last=False)
	test_loader = data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=4, drop_last=False)
	return train_loader, test_loader

def rename():

	[os.rename(os.path.join('./data/dataset/accelerating/1_New/',f),os.path.join('./data/train/','accelerating_0_New_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/accelerating/1_New/')) if f.endswith('.wav')]
	[os.rename(os.path.join('./data/dataset/accelerating/2_CKD_Long/',f),os.path.join('./data/train/','accelerating_1_CKD_Long_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/accelerating/2_CKD_Long/')) if f.endswith('.wav')]
	[os.rename(os.path.join('./data/dataset/accelerating/3_CKD_Short/',f),os.path.join('./data/train/','accelerating_2_CKD_Short_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/accelerating/3_CKD_Short/')) if f.endswith('.wav')]
	[os.rename(os.path.join('./data/dataset/accelerating/4_Old/',f),os.path.join('./data/train/','accelerating_3_Old_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/accelerating/4_Old/')) if f.endswith('.wav')]		
	[os.rename(os.path.join('./data/dataset/braking/1_New/',f),os.path.join('./data/train/','braking_4_New_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/braking/1_New/')) if f.endswith('.wav')]
	[os.rename(os.path.join('./data/dataset/braking/2_CKD_Long/',f),os.path.join('./data/train/','braking_5_CKD_Long_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/braking/2_CKD_Long/')) if f.endswith('.wav')]
	[os.rename(os.path.join('./data/dataset/braking/3_CKD_Short/',f),os.path.join('./data/train/','braking_6_CKD_Short_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/braking/3_CKD_Short/')) if f.endswith('.wav')]
	[os.rename(os.path.join('./data/dataset/braking/4_Old/',f),os.path.join('./data/train/','braking_7_Old_'+str(i)+'.wav')) for i,f in enumerate(os.listdir('./data/dataset/braking/4_Old/')) if f.endswith('.wav')]

