# Image Classification Model Comparison
The classification task has been achieved using a CNN for feature extraction. Then performd using Stochastic Gradient Descent and an Adam Optimizer. Lastly tested with Transfer Learning.

Every step's evaluation mertics has been measured and plotted for the sake of comparing and testing the effectiveness of each model.

## Environement
The models have been run on google colab and the dataset has been mounted via drive.

## Dataset Overview
The project utilizes an image dataset, for a classification task. It involves steps for data loading and preprocessing tailored to handle images. This includes resizing the images to 128x128 pixels, transforming them into tensors, and normalizing with predefined mean and standard deviation values.

## Code Functionality
The code in this project is designed to implement a deep learning model using the PyTorch framework. The primary functions of the code include:

- Model Loading: Loading the model from a saved state for evaluation.
- Data Preprocessing: Applying transformations to images to prepare them for model input.
- Model Evaluation: Assessing the model's performance on the test set using metrics such as accuracy, precision, recall, and F1 score.
- Visualization: Creating a confusion matrix to visually assess model performance across different classes.

## Libraries Used
PyTorch (for deep learning tasks)
PIL (for image operations)
NumPy (for numerical operations)
Matplotlib and Seaborn (for plotting and visualization)

