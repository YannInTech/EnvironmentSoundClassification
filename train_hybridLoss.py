import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import datetime

import model_classifier
import model_projection
from utils import EarlyStopping, WarmUpExponentialLR
import config
import hybrid_loss

import data_crunch as dataset
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model =torchvision.models.resnet50(weights='DEFAULT').to(device)
model.fc = nn.Sequential(nn.Identity())

model = nn.DataParallel(model, device_ids=[0, 1]) 
model = model.to(device)

projection_layer = model_projection.ProjectionModel().to(device)
classifier = model_classifier.Classifier().to(device)

train_loader, val_loader = dataset.create_generators()

loss_fn = hybrid_loss.HybridLoss(alpha = config.alpha, temperature = config.temperature).to(device)

optimizer = torch.optim.AdamW(list(model.parameters())+list(projection_layer.parameters())+list(classifier.parameters()), lr=config.lr, weight_decay=1e-3)

scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= config.warm_epochs, gamma=config.gamma)

# creating a folder to save the reports and models
root = './results/'
main_path = root + str(datetime.datetime.now().strftime('%Y-%m-%d'))
if not os.path.exists(main_path):
	os.mkdir(main_path)

projector_path = main_path + '/' + 'projector'
if not os.path.exists(projector_path):
    os.mkdir(projector_path)

classifier_path = main_path + '/' + 'classifier'
if not os.path.exists(classifier_path):
    os.mkdir(classifier_path)

trainloss={}
valloss={}
trainacc={}
valacc={}

def hotEncoder(v):
	ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
	for s in range(v.shape[0]):
		ret_vec[s][v[s]] = 1
	return ret_vec


def train_hybrid():
	num_epochs = config.epochs
    
	with open(main_path + '/results.txt','w', 1) as output_file:
		mainModel_stopping = EarlyStopping(patience=20, verbose=True, log_path=main_path, output_file=output_file)
		proj_stopping = EarlyStopping(patience=20, verbose=False, log_path=projector_path, output_file=output_file)
		classifier_stopping = EarlyStopping(patience=20, verbose=False, log_path=classifier_path, output_file=output_file)


		print('*****', file=output_file)
		print('HYBRID', file=output_file)
		print('alpha is {}'.format(config.alpha), file=output_file)
		print('temperature of contrastive loss is {}'.format(config.temperature), file=output_file)
		print('Freq mask number {} and length {}, and time mask number {} and length is {}'.format(config.freq_masks, config.freq_masks_width, config.time_masks, config.time_masks_width), file=output_file)
		print('*****', file=output_file)
        
		for epoch in range(num_epochs):
			print('epoch', epoch+1)
			print('\nlearning rate: ' + str(optimizer.param_groups[0]["lr"]), file=output_file)

			model.train()
			projection_layer.train()
			classifier.train()

			train_loss = []
			train_loss1 = []
			train_loss2 = []
			train_corrects = 0
			train_samples_count = 0

			for x, label in train_loader:
				loss = 0
				optimizer.zero_grad()
				
				x = x.float().to(device)
				label = label.to(device).unsqueeze(1)
				label_vec = hotEncoder(label)
            
				y_rep = model(x)
				y_rep = F.normalize(y_rep, dim=0)
                
				y_proj = projection_layer(y_rep)
				y_proj = F.normalize(y_proj, dim=0)
                
				y_pred = classifier(y_rep)
            
				loss1, loss2 = loss_fn(y_proj, y_pred, label, label_vec)
				
				loss = loss1 + loss2
				torch.autograd.backward([loss1, loss2])
                
				train_loss.append(loss.item())
				train_loss1.append(loss1.item())
				train_loss2.append(loss2.item())

				trainloss[epoch]=loss.item()
                
				optimizer.step()
                
				train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
				train_samples_count += x.shape[0]
                
                
			val_loss = []
			val_loss1 = []
			val_loss2 = []
			val_corrects = 0
			val_samples_count = 0
                
			model.eval()
			projection_layer.eval()
			classifier.eval()
                
			with torch.no_grad():
				for val_x, val_label in val_loader:
					val_x = val_x.float().to(device)
					label = val_label.to(device).unsqueeze(1)
					label_vec = hotEncoder(label)
					y_rep = model(val_x)
					y_rep = F.normalize(y_rep, dim=0)
                        
					y_proj = projection_layer(y_rep)
					y_proj = F.normalize(y_proj, dim=0)
                        
					y_pred = classifier(y_rep)
                            
					loss1, loss2 = loss_fn(y_proj, y_pred, label, label_vec)
					
					loss = loss1 + loss2
                        
					val_loss.append(loss.item())
					val_loss1.append(loss1.item())
					val_loss2.append(loss2.item())

					valloss[epoch]=loss.item()
                        
					val_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
					val_samples_count += val_x.shape[0]
                        
			train_acc = train_corrects / train_samples_count
			val_acc = val_corrects / val_samples_count
                
			scheduler.step()

			trainacc[epoch]=train_acc
			valacc[epoch]=val_acc

                   
			print("\nEpoch: {}/{}...".format(epoch+1, num_epochs), "Train Loss: {:.4f}...".format(np.mean(train_loss)), "Val Loss: {:.4f}".format(np.mean(val_loss)), file=output_file)
			print('train_loss1 is {:.4f} and train_loss2 is {:.4f}'.format(np.mean(train_loss1), np.mean(train_loss2)),file=output_file)
			print('val_loss1 is {:.4f} and val_loss2 is {:.4f}'.format(np.mean(val_loss1), np.mean(val_loss2)), file=output_file)
			print('train_acc is {:.4f} and val_acc is {:.4f}'.format(train_acc, val_acc), file=output_file)

        
			# add validation checkpoint for early stopping here
			mainModel_stopping(-val_acc, model, epoch+1)
			proj_stopping(-val_acc, projection_layer, epoch+1)
			classifier_stopping(-val_acc, classifier, epoch+1)
			if mainModel_stopping.early_stop:
				print("Early stopping", file=output_file)
				return

def plotthis():
	fig, axs = plt.subplots(1, 2,figsize=(14,7))
	axs[0].plot(trainloss.keys(), trainloss.values(), label='train loss')
	axs[0].plot(valloss.keys(), valloss.values(), label='val loss')
	axs[0].set_title('Loss')
	axs[0].set(xlabel='epochs')
	axs[0].legend()
	axs[1].plot(trainacc.keys(), trainacc.values(), label='train accuracy')
	axs[1].plot(valacc.keys(), valacc.values(), label='val accuracy')
	axs[1].set_title('Accuracy')
	axs[1].legend()
	plt.tight_layout()
	plt.savefig(os.path.join(main_path,'tramways learning curves.png'))
    

if __name__ == "__main__":
	train_hybrid()
	plotthis()
