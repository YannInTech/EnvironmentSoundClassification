import numpy as np
import torch
import os

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=5, verbose=False, delta=0, log_path='', output_file = './results.txt'):
		"""
		Args:
		patience (int): How long to wait after last time validation loss improved.
                            Default: 7
		verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
		delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.log_path = log_path
		self.output_file = output_file
        

	def __call__(self, val_loss, model, epoch):

		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, epoch)
		elif score < self.best_score - self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}', file=self.output_file)
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, epoch)
			self.counter = 0

	def save_checkpoint(self, val_loss, model, epoch):
		'''Saves model when validation loss decreases'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...', file=self.output_file)
        
		torch.save(model.state_dict(), os.path.join(self.log_path, 'checkpoint.pt'))
		self.val_loss_min = val_loss
        
        

        
class WarmUpStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int, step_size: int, 
			gamma: float = 0.1, last_epoch: int = -1):
		
		super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
		self.cold_epochs = cold_epochs
		self.warm_epochs = warm_epochs
		self.step_size = step_size
		self.gamma = gamma

		

	def get_lr(self):
		if self.last_epoch < self.cold_epochs:
			return [base_lr * 0.1 for base_lr in self.base_lrs]
		elif self.last_epoch < self.cold_epochs + self.warm_epochs:
			return [
				base_lr * 0.1 + (1 + self.last_epoch - self.cold_epochs) * 0.9 * base_lr / self.warm_epochs
				for base_lr in self.base_lrs
				]
		else:
			return [
				base_lr * self.gamma ** ((self.last_epoch - self.cold_epochs - self.warm_epochs) // self.step_size)
				for base_lr in self.base_lrs
				]


class WarmUpExponentialLR(WarmUpStepLR):

	def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int,
                 	gamma: float = 0.1, last_epoch: int = -1):

		self.cold_epochs = cold_epochs
		self.warm_epochs = warm_epochs
		self.step_size = 1
		self.gamma = gamma

		super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)