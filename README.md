# BrainTumorClassification-CNN
A deep learning project that leverages Convolutional Neural Networks (CNNs) built with PyTorch to classify MRI brain images as tumor or non-tumor. The model is trained to assist early diagnosis by analyzing medical images automatically.

<img width="190" height="225" alt="braintumorclassification" src="https://github.com/user-attachments/assets/3f374dac-0ec6-4edc-8b3d-d1ef89de5a81" />

## Project Overview
Brain tumors are life-threatening and early diagnosis plays a crucial role in treatment. This project presents an end-to-end image classification pipeline using PyTorch to identify whether an MRI scan indicates the presence of a brain tumor.

## Dataset
<img width="522" height="295" alt="aboutthedataset" src="https://github.com/user-attachments/assets/966f57c9-a7a5-46f5-9bc4-56c3e2ae2d68" />

The dataset consists of MRI images labeled as:
- Tumor
- No Tumor<br>
Each image is preprocessed (resized, normalized, etc.) and split into training, validation, and test sets.<br>
**Source:**<br>
Brain MRI Images for Brain Tumor Detection from Kaggle<br>
URL: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection<br>
Credit: Navoneel Chakrabarty

**Technologies Used**
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Seaborn
- torchmetrics (for evaluation)

## Model Architecture
The model is a custom CNN with the following components:
- 2 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer to convert feature maps to a vector
- Fully connected (Dense) layers
- Final output: single raw logit (for binary classification)
- Trained with BCEWithLogitsLoss (sigmoid + binary cross-entropy)

## Performance Metrics & Results
The model is evaluated using: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.<br>
<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/da5e7823-afb2-4aea-a93b-e80975927a95" />

**Metric Value on test set:**<br>
Accuracy	77%<br>
Precision	93%<br>
Recall	72%<br>
F1-Score	81%

## Future Work
- Incorporate transfer learning (e.g., ResNet, EfficientNet)
- Deploy model using Flask or FastAPI
- Improve class balance using data augmentation or synthetic data
