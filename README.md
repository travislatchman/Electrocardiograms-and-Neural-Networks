# Electrocardiograms-and-Neural-Networks 

## Abnormal Heartbeat Detection with LSTMs and CNNs

**Description:** This project involved training a deep learning model to perform binary classification on heartbeats (normal vs abnormal) using electrocardiogram (ECG) data. The primary tools used were Long Short-Term Memory (LSTM) models and Convolutional Neural Networks (CNNs). 
We will use MIT-BIH Arrythmia dataset (https://www.physionet.org/content/mitdb/1.0.0/).
It consists of ECG recordings of several patients with sample rate 360 Hz. Experts annotated/classified specific points in the signals as normal, abnormal, or non beat.

### Implementation (see notebook for each task):
* Data was first visualized and processed, including loading ECG files.
* Feature extraction was performed using Short-Time Fourier Transform (STFT).
* A Bidirectional LSTM was used for classification, and the model was evaluated by calculating accuracy.
* Deep CNNs were also explored for classification tasks.  

### **`Task 1 Data visualization`** 

Plot any random 10-second long portion of this ECG file (patient 100). Then plot any 1-second portion of this ECG file (patient 100) which has an abnormality approximately in the middle of the signal.  

![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/637e819f-1fa8-4a19-97a5-961a14be18e4)
![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/6657da5d-542d-4782-a23c-55bab75d17e3)

### **`Task 2 Data preparation`** 
Training data or test data is usually represented by a matrix $X \in \mathbb{R}^{N\times D}$. N represents the number of training points, and D represents the data dimension. We will consider one data point as +/- 2 seconds sequence of samples centered around a Q wave (annotation). Therefore, $D = 4f$ , where $f$ is the sample rate. Your goal is to construct such data matrix $X$. Your function should also output the corresponding label vector $y \in \mathbb{R}^{N\times 1}$. Labels should be 0 for Normal and 1 for abnormal. You should get close to a total of 100k data points.


will use Signal Processing lib [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html) to extract features for training and testing data matrices. We will do short-time Fourier transform to extract the spectrogram of ECG signal. A common format of spectrogram is a graph with two geometric dimensions: one axis represents time, and the other axis represents frequency; a third dimension indicating the amplitude of a particular frequency at a particular time is represented by the intensity or color of each point in the image.

Examples of ECG spectrogram are shown below:

![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/e7ed5fa9-30a5-43ac-ba3a-f42da51d4d6d)

M. Salem, S. Taheri and J. Yuan, "ECG Arrhythmia Classification Using Transfer Learning from 2- Dimensional Deep CNN Features," 2018 IEEE Biomedical Circuits and Systems Conference (BioCAS), Cleveland, OH, USA, 2018, pp. 1-4, doi: 10.1109/BIOCAS.2018.8584808.

### **`Task 4 Distribution and compensation for imbalance`** 
Plot the distribution of percentage of oberservations with normal and abnormal labels

![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/1effa0ee-1a96-467c-b4f3-3f86eef31fe4)

To compensate for the imbalanced dataset during training, we can use a weighted loss function: Modify the loss function during training to give more importance to the minority class. This can be achieved by assigning class weights inversely proportional to their frequency in the dataset.

we can use techniques such as oversampling, undersampling, or a combination of both. Oversampling involves increasing the number of samples in the minority class (abnormal class in this case), while undersampling involves decreasing the number of samples in the majority class (normal class in this case). A combination of both involves both oversampling and undersampling.

### **`Task 5 LSTM`** 
Now, we are going to train a classifier to detect abnormal ECG cycles. We will train a simplified version of the LSTM-based network described in one of the [previously cited papers](https://www.sciencedirect.com/science/article/pii/S0010482518300738?casa_token=qrJ6hAf9tkYAAAAA:7uXqrKY5WqUM6Mjc_qg7wJ4R6QA02BGFXP0o_pOKN09yB8JIXb7067JZWY88rZc8M1G6gkkA).

Using Pytorch, create a single layer Bidirectional LSTM model. Followed by LSTM layer, you should have linear layer with sigmoid activation and a single output (we are predicting Normal/Abnormal).Train and test your model, report your accuracy and F1-score on test set and train set. (You can find the definition and formula of accuracy, precision, recall-rate, and f1-score from this [link](https://towardsdatascience.com/the-f1-score-bec2bbc38aa6)). Print the loss function and accuracy while training to make sure your model works. Add a flattening layer after LSTM layer (and before linear layer).

### **`Task 6 1-D CNNs`** 
Different to LSTM model, we will have [1D CNN](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) layer with ReLU activation. You need to add a flattening layer just after this (and before linear layer).

Using Pytorch, create a deep CNN model (more than 1 layer). Followed by CNN layer, you should have one or several linear layers with different kinds of activation and number of final output units equals to 1. Train and test your model, report your accuracy and F1-score on test set and train set. Print the loss function and accuracy while training to make sure your model works. Add a flattening layer after CNN layer (and before linear layer). Sigmoid is recommended as the activation of your last linear layer.

### **`Task 7 Alexnet`** 
AlexNet is a deep convolutional neural network (CNN) designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, which achieved a significant breakthrough in the field of computer vision by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. It was one of the first deep neural networks to use multiple layers and dropout regularization to prevent overfitting. You can find the introduction of Alexnet in this paper ["ImageNet Classification with Deep Convolutional
Neural Networks"](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

Establish and train a AlexNet which is similiar to the AlexNet of the paper: ["Classification of ECG signal using FFT based improved Alexnet classifier"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9514660/) The structure of Alex Net is shown below. Train and test AlexNet, report accuracy and F1-score on test set and train set. If you think the features are too less for so many layers, you can use one conv layer to replace conv 3 - conv 5 and one linear layer to replace FC 6 - FC 8. You can add dropout layer if needed.

![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/fdcc4d0b-bda1-47b2-8e76-dba82d56b4ef)

Kumar M A, Chakrapani A. Classification of ECG signal using FFT based improved Alexnet classifier. PLoS One. 2022;17(9):e0274225. Published 2022 Sep 27. doi:10.1371/journal.pone.0274225

### **`Task 8 GRU`** 
Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks, [introduced in 2014 by Kyunghyun Cho et al](https://arxiv.org/abs/1409.1259). You need to create your own GRUs, train and test the model, and report accuracy and F1-score on train set and test set.
