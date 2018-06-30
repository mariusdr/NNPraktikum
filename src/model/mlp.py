import numpy as np

from util.loss_functions import BinaryCrossEntropyError, MeanSquaredError, AbsoluteError, SumSquaredError, DifferentError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from sklearn.metrics import accuracy_score


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.005, epochs=50,
                 record_performance=True):
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
        self.record = record_performance
        
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
        self.error_string = loss
        # Build up the network from specific layers, input layer first, output layer last
        self.layers = [
            LogisticLayer(train.input.shape[1], 128, None, "sigmoid", False),
            LogisticLayer(128, 128, None, "sigmoid", False, use_weight_decay=False),
            LogisticLayer(128, 10, None, self.outputActivation, True)
        ]
        self.num_layers = len(self.layers)

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
        return self.loss.calculateError(target, self.outp)

    def _compute_error_derivative(self, target):
        return self.loss.calculateDerivative(target, self.outp)

    def train(self, verbose=False):
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

            if self.record:
                self._record_performance(epoch, verbose)

    def _train_one_epoch(self):
        for inp, label in zip(self.trainingSet.input, self.trainingSet.label):
            self._feed_forward(inp)

            target_onehot = np.zeros(10)
            target_onehot[label] = 1
            loss_derivative = self._compute_error_derivative(target_onehot)

            # backprop @ output layer: input delta = loss, weights = [1.0...]
            self._get_output_layer().computeDerivative(loss_derivative, 1.0)

            # backprop @ inner layers: input delta = delta of upper layer,
            #                          weights = weights of upper layer
            for i in range(self.num_layers - 2, -1, -1):
                curr_layer = self._get_layer(i)
                prev_layer = self._get_layer(i + 1)

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

    def _record_performance(self, epoch, verbose):
        train_classes = self.evaluate(test=self.trainingSet.input)
        train_accuracy = accuracy_score(self.trainingSet.label, train_classes)

        valid_classes = self.evaluate(test=self.validationSet.input)
        valid_accuracy = accuracy_score(
            self.validationSet.label, valid_classes)

        if verbose:
            print("Accuracy on validation set: {0:.2f}%".format(
                valid_accuracy * 100))
            print("Accuracy on training set: {0:.2f}%".format(
                train_accuracy * 100))

        perf = {"epoch": epoch,
                "validation accuracy": valid_accuracy,
                "training accuracy": train_accuracy}
        self.performances.append(perf)

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                             axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)

    def __str__(self):
        out = "MultiLayerPerceptron(\n"
        for layer in self.layers:
            out += str(layer) + ",\n"
        out += "loss: " + self.error_string + ")"
        return out
