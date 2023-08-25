### Environment Sound Classification
## The Prague tramways application
### Yann Monneyron

	Subject

Classify four Prague tramway types using sound recordings of those tramways accelerating or braking in a urban environment. There are eight classes in total.

	Methodology

The key in the performance achieved by the present classifier are the loss function and the strong data augmentation pipeline.  
For the purpose of hyperparameters tuning while defining the loss function and data augmentation configuration, the dataset provided covering 8 classes for accelerating and braking tramways of 4 different types is used for supervised learning and accuracy calculation.  
The provided dataset is split over a training and test subsets with a 80/20% split applied to each of the eight classes. This is to avoid random splits which could lead to instances where the test set is void of samples from the classes that count very few samples, which are more difficult to fit. The separation is made permanent by splitting the data set in two separate subfolders. Also, the test samples are never seen during training and the training is executed in batches over all the training subset.  
The approach on a high level is to employ a convolutional neural network applied to the log transform of the mel spectrograms representation of the audio files. This effectively treats the problem as a computer vision problem with the mel spectrograms consisting of a discrete representation of the frequency and time length of the .wav sound files. The log transform scales those in order to set a clear emphasis on the lower end frequencies distinguishable by the human ear. Although the classification can be done using frequencies which are inaudible, this is scaling down the high end of the frequency range of the sound signal.

	Loss function

The main idea guiding the present approach is the use of Supervised Contrastive Learning, introduced in 2020 in this key paper: https://arxiv.org/abs/2004.11362 The implementation of the contrastive loss function by the team that developed it is used in my project code with the concerned module citing this source.  
I believe the use of the appropriate loss function governing gradient backpropagation and the entire training algorithm is just as powerful as the architecture choices for the neural network itself.  
The contrastive loss focuses on positives and negatives separation including the augmented versions of the samples in order to determine the most discriminative features between samples of different classes.  
In comparison, the usual cross entropy loss function for classifiers which is based on mapping samples to labels by minimizing the loss given the samples distribution, does not focus on learning discriminative features between samples of different classes resulting in lower performance when classes are highly similar in patterns and features describing them.  
This work employs both loss functions in an hybrid total loss, which is balanced after some hyperparameter tuning experiments to 50/50, determined by the alpha=0.5 parameter entered in the config.py module.  
Loss=α.contrastive_loss+(1-α)cross-entropy_loss    
The hybrid loss relies on a projection stage using a fully connected layer. This additional layer added to the base classifier CNN model serves the contrastive loss operations in discriminating classes in that new embedding. Please note the present work leverages transfer learning by initializing the CNN ResNet-50 model on a set of weights trained on the ImageNet dataset.  

	Data augmentation
	
As a data set of limited size and strongly imbalanced, data augmentation is key in avoiding classification biases due to the classes cardinal differences. That imbalanced distribution in the training set affects the model performance by perpetrating the same biases when querying the model at inference time. This is due to the model overfitting the features found in samples when those are much fewer in one class compared to another. Along with using the contrastive loss in a final hybrid function, proper data augmentation is key in achieving excellent generalization for the classifier model.  
The samples of all classes are augmented in both the raw wave signals and in the spectrogram representations.  
The raw wave signals are treated to remove possible silent parts first, then randomly scaled in order to achieve a combination of time-stretching and pitch-shifting, then randomly padded and cropped to generate first stage augmented new samples in the same classes.  
Second, all the first stage augmented wave signals are converted to log mel spectrograms and a new stage of augments is executed. The frequency bands are randomly masked to hide portions of them over an interval chosen by hyperparameter tuning. The time length of each mel spectrogram is also randomly masked over a length hyperparameter which is also tuned. Two frequency masks and one time mask are executed for every sample to generate new augmented ones. This can also be changed and adapted in the config.py module.

	Operations

All samples are named to contain the class id (int) in the file name, ranging from 0 to 7.  
0: accelerating Skoda T15 https://en.wikipedia.org/wiki/%C5%A0koda_15_T  
1: accelerating CKD_Long - Tatra KT8D5R.N2P https://cs.wikipedia.org/wiki/Tatra_KT8D5R.N2P  
2: accelerating CKD_Short - Tatra T6A5 one or two carriages https://cs.wikipedia.org/wiki/Tatra_T6A5  
3: accelerating Old - Tatra T3 https://cs.wikipedia.org/wiki/Tatra_T3  
4: braking Skoda T15  
5: braking CKD_Long - Tatra KT8D5R.N2P  
6: braking CKD_Short - Tatra T6A5  
7: braking Old - Tatra T3  
  
The current setup achieved 82% accuracy on the training subset and 93% on the test subset after 200 epochs (configurable in config.py)  
The results are automatically saved at each epoch in the results folder, followed by a subfolder automatically generated with the date of training. The model weights are saved as checkpoint.pt files. The results.txt file contains the loss and accuracy for both loss functions and for both training, validation subsets for every epoch.  
The training procedure is programmed to stop if the loss did not decrease over 20 consecutive epochs.  
After 200 epochs, the learning curve has not converged yet and there is still significant margin for improvement. I recommend executing the training module with at least 500 epochs before the training is complete and the learning curve has converged.  
Please execute from the terminal the train_hybridLoss.py module as the start main module.  
