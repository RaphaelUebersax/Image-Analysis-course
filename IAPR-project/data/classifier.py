from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle


class ValueClassifier:
    classifier = None

    def __init__(self, filename='../data/project-data/value_classifier_997_982.mdl'):
        self.classifier = pickle.load(open(filename, 'rb'))
        return

    def predict(self, images):
        images_ve = self.__vector_encode(images)
        prediction = self.classifier.predict(images_ve)
        return prediction

    def score(self, images, label):
        images_ve = self.__vector_encode(images)
        accuracy = self.classifier.score(images_ve, label)
        return accuracy

    def __vector_encode(self, images_stack):
        return images_stack.reshape(images_stack.shape[0], images_stack.shape[1] * images_stack.shape[2])
