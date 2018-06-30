#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from report.evaluator import Evaluator
from model.mlp import MultilayerPerceptron


def draw_plot(performances, num_epochs):
    validation_vals = []
    training_vals = []
    for perf in performances:
        validation_vals.append(perf["validation accuracy"])
        training_vals.append(perf["training accuracy"])

    epochs = np.arange(0, num_epochs)
    plt.plot(epochs, validation_vals, "b-", label="validation accuracy")
    plt.plot(epochs, training_vals, "g-", label="training accuracy")
    plt.legend()
    plt.show()


def run_experiment(data, learningRate, epochs):
    mlpClassifier = MultilayerPerceptron(data.trainingSet,
                                         data.validationSet,
                                         data.testSet,
                                         learningRate=learningRate,
                                         epochs=epochs)

    print("parameters: {")
    print("learning rate: " + str(mlpClassifier.learningRate))
    print("num. epochs: " + str(mlpClassifier.epochs))
    print("layers:")
    print(mlpClassifier)
    print("}")
    mlpClassifier.train(verbose=False)

    print("results on test set:")
    mlpPred = mlpClassifier.evaluate()
    evaluator = Evaluator()
    evaluator.printAccuracy(data.testSet, mlpPred)

    print("error graphs:")
    draw_plot(mlpClassifier.performances, mlpClassifier.epochs)


if __name__ == '__main__':
    data = MNISTSeven("../data/mnist_seven.csv",
                      3000, 1000, 1000, oneHot=False)

    run_experiment(data, 0.25, 3)

    
# def main():
#     data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
#     myStupidClassifier = StupidRecognizer(data.trainingSet,
#                                           data.validationSet,
#                                           data.testSet)
#     myPerceptronClassifier = Perceptron(data.trainingSet,
#                                         data.validationSet,
#                                         data.testSet,
#                                         learningRate=0.005,
#                                         epochs=30)

#     # Train the classifiers
#     print("=========================")
#     print("Training..")

#     print("\nStupid Classifier has been training..")
#     myStupidClassifier.train()
#     print("Done..")

#     print("\nPerceptron has been training..")
#     myPerceptronClassifier.train()
#     print("Done..")

#     # Do the recognizer
#     # Explicitly specify the test set to be evaluated
#     stupidPred = myStupidClassifier.evaluate()
#     perceptronPred = myPerceptronClassifier.evaluate()

#     # Report the result
#     print("=========================")
#     evaluator = Evaluator()

#     print("Result of the stupid recognizer:")
#     # evaluator.printComparison(data.testSet, stupidPred)
#     evaluator.printAccuracy(data.testSet, stupidPred)

#     print("\nResult of the Perceptron recognizer:")
#     # evaluator.printComparison(data.testSet, perceptronPred)
#     evaluator.printAccuracy(data.testSet, perceptronPred)