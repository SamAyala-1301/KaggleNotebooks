# Digit Recognition Model

## Overview
This project is a Convolutional Neural Network (CNN)-based digit recognition model trained on the MNIST dataset. The model reads digit images from a CSV file, processes them, and predicts the corresponding digit (0-9). It is designed for submission to Kaggle's "Digit Recognizer" competition.

## Features
- Custom dataset loader for CSV-formatted MNIST data
- Data augmentation and normalization using PyTorch's `torchvision.transforms`
- CNN architecture with three convolutional layers
- Training and inference on GPU (if available)
- Batch processing using `torch.utils.data.DataLoader`
- Generates predictions for Kaggle submission

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib

### Install Dependencies
```sh
pip install torch torchvision pandas numpy matplotlib
```

## Dataset
The dataset is provided in CSV format where:
- Each row represents a 28x28 grayscale image.
- The first column (in training data) contains the digit label.
- The remaining 784 columns contain pixel values.

### Data Files
- **train.csv**: Training data with labels
- **test.csv**: Test data without labels

## Model Architecture
The CNN consists of:
1. **Conv Layer 1**: 32 filters, 3x3 kernel, ReLU activation, MaxPooling (2x2)
2. **Conv Layer 2**: 64 filters, 3x3 kernel, ReLU activation, MaxPooling (2x2)
3. **Conv Layer 3**: 128 filters, 3x3 kernel, ReLU activation, MaxPooling (2x2)
4. **Fully Connected Layer 1**: 128 neurons with ReLU activation
5. **Fully Connected Layer 2**: 20 neurons
6. **Output Layer**: 10 neurons (softmax activation for digit classification)

## Training
The model is trained using:
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum=0.9
- **Learning Rate**: 0.001

### Training Script
```python
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## Inference and Prediction
- The trained model is used to predict labels for the test dataset.
- The results are stored in a CSV file for Kaggle submission.

### Generate Predictions
```python
model.eval()
predictions = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().tolist())
```

### Submission File
```python
submission = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})
submission.to_csv('predictions_final.csv', index=False)
```

## Improvements
- **More Advanced CNN Architecture**: Using deeper networks like ResNet, EfficientNet
- **Data Augmentation**: Applying random rotations, brightness adjustments, and affine transformations
- **Hyperparameter Tuning**: Optimizing learning rates, dropout rates, and batch sizes
- **Ensemble Models**: Combining predictions from multiple models for higher accuracy

## Acknowledgments
- PyTorch documentation and tutorials
- Kaggle's Digit Recognizer competition dataset

## License
This project is open-source and available for learning and research purposes.

