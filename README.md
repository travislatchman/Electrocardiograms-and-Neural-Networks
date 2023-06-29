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

### **`TASK 5 LSTM`** 
Now, we are going to train a classifier to detect abnormal ECG cycles. We will train a simplified version of the LSTM-based network described in one of the [previously cited papers](https://www.sciencedirect.com/science/article/pii/S0010482518300738?casa_token=qrJ6hAf9tkYAAAAA:7uXqrKY5WqUM6Mjc_qg7wJ4R6QA02BGFXP0o_pOKN09yB8JIXb7067JZWY88rZc8M1G6gkkA).

**Task 5.1:** Using Pytorch, create a single layer Bidirectional LSTM model. Followed by LSTM layer, you should have linear layer with sigmoid activation and a single output (we are predicting Normal/Abnormal).

**Task 5.2:** While creating your LSTM model, how could you validate your model in a real scientific experiment? (Describe and comment it in a text cell)
5.2: In a real scientific experiment, we could validate the LSTM model by using a holdout validation set to evaluate the performance of the model on data that was not used during training. Also, cross-validation can be used to estimate the generalization performance of the model on a wider range of data. We can also compare the performance of the LSTM model to that of other models or methods on the same dataset and visualize the learned representations in the LSTM layer to gain insights into the features that are important for classification.

**Task 5.3:** Train and test your model, report your accuracy and F1-score on test set and train set. 

**Task 5.4:** What do you think is a way to increase the accuracy and F1-score? Try it and show if it is helpful.
5.4: To increase the accuracy and F1-score, we can try several approaches:

Increase the number of layers or hidden units in the LSTM model.
Apply regularization techniques such as dropout or weight decay.
Optimize hyperparameters using techniques like grid search, random search, or Bayesian optimization. Train the model for a higher number of epochs or with different learning rates.

**Note 1:** Print the loss function and accuracy while training to make sure your model works.

**Note 2:** You need to add a flattening layer after LSTM layer (and before linear layer).

**Note 3:** The output of LSTM in pytorch lib have a tuple outout, add the following GetLSTMOutput after your layer. If your model doesn't have this problem, you can ignore this.
