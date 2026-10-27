# Cat and Dog Image Classification Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify images as either a cat or a dog. The model is built using TensorFlow and Keras and trained on the Microsoft Cats and Dogs dataset.

## Technologies Used
- Python
- TensorFlow and Keras
- CNN (Convolutional Neural Network)
- PIL (Python Imaging Library)
- OS and Glob modules

## Dataset
The dataset is downloaded automatically from Microsoft.

It contains two categories:
- Cat
- Dog

The dataset is divided into:
- 80% Training Data
- 20% Validation Data

Images are resized to 128 × 128 pixels and converted to RGB format.

## Model Architecture
The CNN model consists of:

1. Rescaling layer for pixel normalization.
2. Data augmentation using random flipping, rotation, and zoom.
3. Three convolutional layers with 32, 64, and 128 filters.
4. Batch normalization layers to improve training stability.
5. Max pooling layers to reduce feature dimensions.
6. Flatten layer to convert feature maps into a one-dimensional vector.
7. Dense layer with 128 neurons.
8. Dropout layer to reduce overfitting.
9. Output layer with sigmoid activation to classify images into two categories.

## Project Structure

```text
CNN_PROJECT/
│
├── cnn.py
├── README.md
└── data/
    └── PetImages/
        ├── Cat/
        └── Dog/
```

The dataset is downloaded and extracted automatically when the program runs.

## Installation

Install the required libraries:

```bash
pip install tensorflow pillow
```

## How to Run

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install the required libraries.
4. Run the Python program:

```bash
python cnn.py
```

The program downloads the dataset, removes invalid images, preprocesses the images, trains the CNN model for two epochs, and evaluates its performance.

## Model Training

The model uses:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Batch Size: 32
- Image Size: 128 × 128
- Training Epochs: 2

## Output

After training, the model displays the evaluation loss and classification accuracy on the validation dataset.

Example:

```text
Test Accuracy: XX.XX%
```

The actual accuracy depends on the training process and dataset.

## Applications
- Pet image classification
- Animal image recognition
- Computer vision learning
- Deep learning applications

## Future Improvements
- Increase the number of training epochs.
- Improve accuracy through hyperparameter tuning.
- Use transfer learning with pretrained CNN models.
- Develop a web application for uploading and classifying images.
- Save the trained model for future predictions.
