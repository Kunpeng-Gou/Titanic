import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import processing as pr



class KNN(object):
    def __init__(self, train_data, k=5):
        self.train_data = train_data
        self.k = k
        self.ratio_array = None

    def distance(self, test):
        train = np.array([self.train_data[i][0] for i in range(len(self.train_data))])
        difference = abs(train - test)
        return np.sum(difference ** 2, axis=1)

    def label_list(self, test):
        dis = self.distance(test)
        index = np.argsort(dis)
        ret = [self.train_data[i][1] for i in index]
        return ret

    def predict(self, test, k=None):
        if not k:
            k = self.k
        labels = self.label_list(test)
        count = np.zeros(len(set(labels)))
        for i in range(k):
            count[labels[i]] += 1
        return np.argsort(-count)[0]

    def accuracy(self, test_set, k=None):
        if not self.ratio_array:
            self.set_ratio_array(test_set, k)
        cnt = 0
        for i in range(len(test_set)):
            # print(test_set[i][1], self.ratio_array[i])
            label = 0
            if self.ratio_array[i][k] <= 0.5:
                label = 1
            if label == test_set[i][1]:
                cnt += 1
            # if label != test_set[i][1]:
                # print(test_set[i][1], self.ratio_array[i])
        # print(cnt / len(test_set))
        return cnt / len(test_set)


    def set_ratio_array(self, test_set, k):
        self.ratio_array = []
        length = len(self.train_data)
        # print(length)
        for i in range(len(test_set)):
            labels = self.label_list(test_set[i][0])
            zero_count = np.zeros(length)
            # print(zero_count)
            for j in range(length):
                if j == 0 and labels[j] == 1:
                    continue
                elif j == 0 and labels[j] == 0:
                    zero_count[j] = 1
                    continue
                zero_count[j] = zero_count[j - 1] + (1 - labels[j])

            ratio = np.array([zero_count[i] / (i + 1) for i in range(length)])
            self.ratio_array.append(ratio)
        # print(self.ratio_array)
        print('finished')


class Analysis(object):
    def __init__(self, classifier, test_data):
        self.classifier = classifier
        self.test_data = test_data


if __name__ == '__main__':
    da = pd.read_csv('train.csv')
    da = pr.Processing(da)
    da.data_processing()
    di = pr.Divide(da.get_data())
    '''
    c = 10
    times = 5
    for k in range(10):
        acc = 0
        for i in range(times):
            cv = di.cross_validation(c)
            for data in cv:
                train = data[0]
                test = data[1]
                knn = KNN(train, k)
                acc += knn.accuracy(test) / c / times
        print(k, acc)
    '''
    c = 10
    k = 100
    times = 1
    acc = np.zeros(k)
    for i in range(times):
        cv = di.cross_validation(c)
        for data in cv:
            train = data[0]
            test = data[1]
            knn = KNN(train)
            for j in range(k):
                acc[j] += knn.accuracy(test, j) / times / c
    print(acc)
