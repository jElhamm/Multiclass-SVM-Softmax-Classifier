# It supports two types of loss functions: Support Vector Machine (SVM) loss and Softmax loss. 

# The class has the following methods:
# __init__(): Initializes the classifier with an empty weight matrix.
# train()   : Trains the classifier on the given data using stochastic gradient descent (SGD). 
#             It takes input data X, corresponding labels y, and various hyperparameters such as learning rate, regularization strength, 
#             number of iterations, batch size, verbose mode, and loss type. It updates the weight matrix W based on the chosen loss function 
#             and returns the history of loss values during training.
# predict() : Predicts the class labels for a given set of input data X using the learned weight matrix W.



import numpy as np

class Classifier:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False, loss_type='svm'):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)                   # Initializing the weight matrix with small random values

        loss_history = []
        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]

            if loss_type == 'svm':
                loss, dW = self.multiclass_svm_loss(X_batch, y_batch, reg)
            elif loss_type == 'softmax':
                loss, dW = self.softmax_loss(X_batch, y_batch, reg)
            else:
                raise ValueError('Invalid loss type. Please choose \'svm\' or \'softmax\'.')

            loss_history.append(loss)
            self.W -= learning_rate * dW                                        # Updating the weight matrix based on the gradients
            if verbose and it % 100 == 0:
                print(f'iteration {it}/{num_iters}: loss {loss}')               # Printing the loss periodically

        return loss_history

    def predict(self, X):
        y_pred = np.argmax(X.dot(self.W), axis=1)                               # Predicting the class labels based on the highest scores
        return y_pred