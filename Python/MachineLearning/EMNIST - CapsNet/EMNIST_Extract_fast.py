import time
import csv
import numpy as np
import matplotlib.pyplot as plt


class emnist_ext:

    def __init__(self):
        self.count = 0
        self.terminus = 0

        self.testCount = 0
        self.testTerminus = 0
        
        self.myFileTrain = []
        with open('emnist-letters-train.csv', 'r') as csvfile:
            self.myFileTrain = list(csv.reader(csvfile))
        self.myFileTest = []
        with open('emnist-letters-test.csv', 'r') as csvfile:
            self.myFileTest = list(csv.reader(csvfile))

    def reset_counts(self):
        self.count = 0
        self.terminus = 0
        self.testCount = 0
        self.testTerminus = 0

    def train_batch(self, batch_size):
        internal = 0
        
        x_train = np.zeros((batch_size,784))
        y_train = np.zeros((batch_size))
        
        self.terminus = self.count + batch_size
        for i in range(self.count,self.terminus):
            y_train[internal] = self.myFileTrain[self.count][0]
            image = self.myFileTrain[self.count][1:785]
            image = np.reshape(image, (28,28))
            image = np.rot90(image,3)
            image = np.flip(image,1)
            image = np.reshape(image, (784))
            x_train[internal] = image
            self.count += 1
            internal += 1
                
        return x_train, y_train

    def test_batch(self, batch_size):
        internal = 0
        
        x_test = np.zeros((batch_size,784))
        y_test = np.zeros((batch_size))
        
        self.testTerminus = self.testCount + batch_size
        for i in range(self.testCount,self.testTerminus):
            y_test[internal] = self.myFileTest[self.testCount][0]
            image = self.myFileTest[self.testCount][1:785]
            image = np.reshape(image, (28,28))
            image = np.rot90(image,3)
            image = np.flip(image,1)
            image = np.reshape(image, (784))
            x_test[internal] = image
            self.testCount += 1
            internal += 1
                
        return x_test, y_test

