# This program generates random samples and trains a classifier using either Softmax or SVM loss. 
# It then predicts labels and prints the predicted labels and loss history.


import numpy as np
from Classifier import Classifier
from SoftmaxClassifier import SoftmaxClassifier
from MulticlassSVM import MulticlassSVM


def main():
    print("\n-----------------------------------------------------------------")
    num_samples = int(input("---> Enter the number of samples: "))
    dim = int(input("---> Enter the dimension of samples: "))
    num_classes = int(input("---> Enter the number of classes: "))
    X = np.random.randn(num_samples, dim)                                   # Generate random samples
    y = np.random.randint(0, num_classes, size=num_samples)                 # Generate random labels for the samples
    classifier = Classifier()                                               # Create an instance of the classifier

    #******************************************************************************************************************************************
    # Model training
    # Prompt the user to enter the loss type (svm/softmax)
    loss_type = input("---> Enter loss type (svm/softmax): ")
    # Train the classifier using the provided loss type
    loss_history = classifier.train(X, y, learning_rate=1e-3, reg=1e-5, num_iters=1000, batch_size=200, verbose=True, loss_type=loss_type)
    # Prediction
    # Predict labels for the input samples
    y_pred = classifier.predict(X)
    #******************************************************************************************************************************************

    print("\n-----------------------------------------------------------------")
    print("*** Predicted labels:", y_pred)
    print("*** Loss history:", loss_history)
    print("-----------------------------------------------------------------\n")


if __name__ == "__main__":
    main()