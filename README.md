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

**`Task 1 Data visualization`** - Plot any random 10-second long portion of this ECG file (patient 100). Then plot any 1-second portion of this ECG file (patient 100) which has an abnormality approximately in the middle of the signal.  

![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/637e819f-1fa8-4a19-97a5-961a14be18e4)
![image](https://github.com/travislatchman/Electrocardiograms-and-Neural-Networks/assets/32372013/6657da5d-542d-4782-a23c-55bab75d17e3)

