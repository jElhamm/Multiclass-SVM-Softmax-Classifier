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
 