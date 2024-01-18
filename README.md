# Multiclass SVM and Softmax Classifier


   This repository contains implementations of Multiclass [*Support Vector Machine (SVM)*](https://en.wikipedia.org/wiki/Support_vector_machine) and [*Softmax Classifier*](https://en.wikipedia.org/wiki/Softmax_function) in Python.
   These classifiers are commonly used in machine learning for multiclass classification problems. 


   The repository consists of the following four files:


   1. [*Classifier.py*](Source%20Code/Classifier.py): This file contains the implementation of the Classifier class.
   The Classifier class provides methods for training and predicting using the multiclass SVM and softmax classifiers.


   2. [*MulticlassSVM.py*](Source%20Code/MulticlassSVM.py): This file contains the implementation of the MulticlassSVM class.
   The MulticlassSVM class provides the loss function and gradient computation for the multiclass SVM classifier.


   3. [*SoftmaxClassifier.py*](Source%20Code/SoftmaxClassifier.py): This file contains the implementation of the SoftmaxClassifier class.
   The SoftmaxClassifier class provides the loss function and gradient computation for the softmax classifier.


   4. [*main.py*](Source%20Code/main.py): This file demonstrates the usage of the implemented classifiers by generating random samples, 
   training the classifier, and making predictions.


## Classifiers Supported

   The repository supports two types of loss functions:

   2. [*Softmax Loss*](https://pyimagesearch.com/2016/09/12/softmax-classifiers-explained/): The softmax loss is another commonly used loss function for classification tasks.
   It computes the probabilities of each class and aims to minimize the cross-entropy loss.


   1. [*Support Vector Machine (SVM) Loss*](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer): The SVM loss is a popular loss function used for classification tasks.
   It aims to maximize the margin between different classes.


## Classifier Class

   The `Classifier` class provides the main functionality for training and predicting using the multiclass SVM and softmax classifiers
    It has the following methods:

   1. `__init__()`: Initializes the classifier with an empty weight matrix.
   2. `train(X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False, loss_type='svm')`: 
   Trains the classifier on the given data using stochastic gradient descent (SGD). 
   It takes input data `X`, corresponding labels `y`, and various hyperparameters such as learning rate, 
   regularization strength, number of iterations, batch size, verbose mode, and loss type. 
   It updates the weight matrix based on the chosen loss function and returns the history of loss values during training.
   3. `predict(X)`: Predicts the class labels for a given set of input data `X` using the learned weight matrix.


## MulticlassSVM Class

   The `MulticlassSVM` class provides the loss function and gradient computation for the multiclass SVM classifier.
   It has the following method:

   1. `loss(X, y, W, reg)`: Computes the loss and gradient of the loss with respect to the weights for a multiclass SVM classifier.
   It takes the input data `X`, labels `y`, weight matrix `W`, and regularization strength `reg` as arguments.
   It returns the loss value and gradient.


## SoftmaxClassifier Class

   The `SoftmaxClassifier` class provides the loss function and gradient computation for the softmax classifier.
   It has the following method:

   1. `loss(X, y, W, reg)`: Computes the loss and gradient of the loss with respect to the weights for a softmax classifier.
   It takes the input data `X`, labels `y`, weight matrix `W`, and regularization strength `reg` as arguments.
   It returns the loss value and gradient.


## Usage

   You can use the provided [*main.py*](Source%20Code/main.py) file as a starting point to understand how to use the implemented classifiers.
   It demonstrates the following steps:

   1. Generate random samples and corresponding labels.
   2. Create an instance of the `Classifier` class.
   3. Prompt the user to choose the loss type (SVM or softmax).
   4. Train the classifier using the selected loss type by providing the input data, labels, and hyperparameters.
   5. Make predictions on the input data.


## References

   [Support Vector Machine (SVM) Algorithm](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)
   [Introduction to Softmax for Neural Network](https://www.analyticsvidhya.com/blog/2021/04/introduction-to-softmax-for-neural-network/)




   The implementation in this repository is based on the concepts and algorithms of multiclass SVM and softmax classifier 
   widely used in machine learning literature.
   For more information and theoretical background, please refer to relevant research papers and textbooks on 
   machine learning and classification.