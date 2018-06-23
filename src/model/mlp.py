import numpy as np

from util.loss_functions import BinaryCrossEntropyError, MeanSquaredError, AbsoluteError, SumSquaredError, DifferentError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from sklearn.metrics import accuracy_score


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.08, epochs=50):
        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128,
                                         None, inputActivation, False))

        hiddenActivation = "sigmoid"
        self.layers.append(LogisticLayer(
            128, 128, None, hiddenActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10,
                                         None, outputActivation, True))

        self.num_layers = len(self.layers)

        self.inputWeights = inputWeights

        self.outp = np.ndarray((10, 1))

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                           axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                             axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        self.outp = inp
        for layer in self.layers:
            self.outp = layer.forward(self.outp)
            self.outp = np.insert(self.outp, 0, 1)

        self.outp = np.delete(self.outp, 0)

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output 
        layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        target_onehot = np.zeros(10)
        target_onehot[target - 1] = 1
        return self.loss.calculateError(target_onehot, self.outp)

    def _compute_error_derivative(self, target):
        target_onehot = np.zeros(10)
        target_onehot[target - 1] = 1
        return self.loss.calculateDerivative(target_onehot, self.outp)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print(
                    "Training epoch {0}/{1}..".format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                classes = self.evaluate(test=self.validationSet.input)
                accuracy = accuracy_score(self.validationSet.label, classes)
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%".format(
                    accuracy * 100))
                print("-----------------------------")

            print("DEBUG =============================")
            for layer in self.layers:
                i = self.layers.index(layer)
                print("Layer {0}: max weight: {1}".format(
                    i, np.max(layer.weights)))
                print("Layer {0}: avg weight: {1}".format(
                    i, np.average(layer.weights)))
                print("Layer {0}: min weight: {1}".format(
                    i, np.min(layer.weights)))
                print("Layer {0}: gradient magnitude: {1}".format(
                    i, np.linalg.norm(layer.deltas)))
            print("===================================")

    def _train_one_epoch(self):
        for inp, label in zip(self.trainingSet.input, self.trainingSet.label):
            self._feed_forward(inp)
            loss = self._compute_error_derivative(label)

            # backprop @ output layer: input delta = loss, weights = [1.0...]
            self._get_output_layer().computeDerivative(loss, 1.0)

            # backprop @ inner layers: input delta = delta of upper layer,
            #                          weights = weights of upper layer
            for i in range(self.num_layers - 2, -1, -1):
                curr_layer = self._get_layer(i)
                prev_layer = self._get_layer(i + 1)

                # ignored the bias weights here (first row)... doesn't work
                # the other way with the instructors
                # code @ logistic_layer.py
                weights = prev_layer.weights[1:, :].T
                curr_layer.computeDerivative(prev_layer.deltas, weights)

            for layer in self.layers:
                layer.updateWeights(self.learningRate)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        # neural net outputs probabilites, choose the label in {0...9}
        # with the highest probabiliy
        self._feed_forward(test_instance)
        return np.argmax(self.outp)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                             axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
