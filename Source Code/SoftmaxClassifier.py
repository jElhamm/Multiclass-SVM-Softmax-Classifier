# This program implements a softmax classifier and computes the loss and gradient of the loss with respect to the weights.


import numpy as np

class SoftmaxClassifier:
    def __init__(self):
        pass

    def loss(self, X, y, W, reg):
        num_train = X.shape[0]
        num_classes = W.shape[1]
        scores = X.dot(W)
        scores -= np.max(scores, axis=1, keepdims=True)                                         # Substracting the maximum value to improve numerical stability
        probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)          # Computing the softmax probabilities
        correct_logprobs = -np.log(probabilities[np.arange(num_train), y])
        loss = np.sum(correct_logprobs) / num_train + 0.5 * reg * np.sum(W * W)

        dscores = probabilities.copy()
        dscores[np.arange(num_train), y] -= 1
        dscores /= num_train

        dW = X.T.dot(dscores) + reg * W                                                         # Computing the gradient of the loss with respect to the weights
        return loss, dW