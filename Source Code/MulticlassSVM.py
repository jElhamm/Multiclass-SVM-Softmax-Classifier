# The purpose of this code is to compute the loss and gradient of the loss with respect 
# to the weights for a multiclass Support Vector Machine (SVM) classifier.


import numpy as np

class MulticlassSVM:
    def __init__(self):
        pass

    def loss(self, X, y, W, reg):
        delta = 1.0
        num_train = X.shape[0]
        num_classes = W.shape[1]
        
        scores = X.dot(W)
        correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
        margins = np.maximum(0, scores - correct_class_scores + delta)
        margins[np.arange(num_train), y] = 0
        loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
 