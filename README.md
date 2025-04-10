# Hotdog Deep learning model
It is a deep learning model trained with Keras and the TensorFlow framework on the Saturn Cloud. This model is put together to recognize hotdogs in any restaurant's serving logic. This is a binary classification model that recognizes hotdogs or not.

# Project goals 
* To differentiate between "hot dog" and "not hot dog" images, train a Deep learning model.
* Use various datasets to ensure the model works effectively on unknown data.
* To minimize computation time without sacrificing accuracy.
* Use the pre-trained model as Transfer learning.

# Project Objectives
## Data Collection & Preprocessing
* Gather dataset of hot dog and non-hot dog images.
* Resize and normalize images for consistent model input.
* Apply data augmentation (flipping left, right, horizontally, and vertically) for performance improvement and robustness.

## Model Training & Evaluation
* Use python libraries such as TensorFlow, Keras, numpy, and MatplotLib.
* Use a Convolutional Neural Network (CNNs) model, Xception trained on ImageNet.
* Evaluate accuracy, precision, recall, and F1-score.

## Iteration & Fine-tuning
* Experiment with hyperparameters (batch size, learning rate, dropout).
* Implement early stopping to prevent overfitting.
* Use Transfer Learning to leverage pre-trained models.
* 
# Techniques
This is a supervised machine learning model. Supervised learning is the best approach since I have labeled data (hot dog vs. not hot dog). I used Convolutional Neural networks (CNNs), ideal for image classification tasks and designed from scratch since it is a smaller dataset. I used pre-trained models called Xception and also introduced checkpointing which helped in saving the best model after each epoch of training.

The main language used in the development of this model is Python programming Language while incorporating a deep learning framework such as TensorFlow and Keras. Notable libraries, Numpy, was efficiently used to generate image array for the data handling representation and Matplotlib was constantly been used for that data visualization to pick the best hyperparameter during the most especially Regularization with a lot of Iteration.

The Final Technique is Data Augmentation, This is a very small dataset for the model to perform well on unknown data. I need loads of data, which definitely consumes more energy and time. The best approach is to generate data from the existing images by zooming in and out and flipping inward and outward for better representation. Data Augmentation improved the performance of the model. The image was loaded to test the model, generating an array in the form of a dictionary, and the model correctly predicted rightly not_hot_dog.

# Conclusion
In real-world image classification tasks, the Hot Dog Classifier showcases the effectiveness of machine learning and deep learning.  The model can reliably differentiate between "hot dog" and "not hot dog" images by utilizing Convolutional Neural Networks (CNNs) and Transfer Learning.

Techniques for Image processing, data augmentation, and model improvement were used to improve performance and make sure the classifier performs well when applied to unseen images. The finished models were made available in the directory of the code with an accuracy score of 91.9%, 93.1%, and 93.6%, respectively. This project demonstrates important methods in computer vision, deep learning, and model deployment while showcasing how AI may be used in entertaining yet useful contexts.  Future enhancements might involve: 
* Increasing the dataset's size for improved generalization 
* Improving inference speed for real-time applications and third party integration  
* Using TensorFlow Lite to deploy on web/mobile devices

# Installation
* Python (3.8 or later)
* Pip (Python package manager)
* Git (to clone the repository)
* FlaskAPI (Running Locally)
* Streamlit(Using Web Apps)
* Docker (if you want to run it in a container)

* Check out the article on this project at

https://medium.com/@sashefrro/meal-classification-model-using-deep-learning-c11336d0cfc9





