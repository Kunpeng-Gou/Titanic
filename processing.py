import random as rd
import pandas as pd


'''Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')'''


class Processing(object):
    def __init__(self, data):
        self.data = data
        self.labels = None

    def fill_use_average(self, feature):
        mean = self.data[feature].mean()
        self.data[feature].fillna(mean, inplace=True)

    def fill_use_mode(self, feature):
        count = self.data[feature].value_counts()
        fill = count.idxmax()
        self.data[feature].fillna(fill, inplace=True)

    def numeralization(self):
        feature_list = ['Sex', 'Embarked']
        for feature in feature_list:
            values = set(self.data[feature].values)
            val_to_num = dict(zip(values, list(range(len(values)))))
            for i in range(len(self.data)):
                # print(self.data.loc[i, feature])
                self.data.loc[i, feature] = val_to_num[self.data.loc[i, feature]]

    def data_drop(self, feature_list):
        for feature in feature_list:
            self.data.drop(feature, axis=1, inplace=True)

    def normalization(self):
        for featrue in self.data.keys():
            max_num = self.data[featrue].max()
            min_num = self.data[featrue].min()
            n = max_num - min_num
            for i in range(len(self.data)):
                self.data.loc[i, featrue] = (self.data.loc[i, featrue] - min_num) / n

    def data_processing(self):
        self.labels = self.data['Survived']
        self.data_drop(['Survived', 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'])
        self.fill_use_average('Age')
        self.fill_use_mode('Embarked')
        self.numeralization()
        self.normalization()
        return self.data

    def get_data(self):
        return [(self.data.values[i], self.labels.loc[i]) for i in range(len(self.data))]


class Divide(object):
    def __init__(self, data):
        self.data = data

    def cross_validation(self, k):
        id_list = list(range(len(self.data)))
        rd.shuffle(id_list)
        data_list = []
        length = int(len(id_list) / k)
        for i in range(k - 1):
            data_list.append(self.data[length * i:length * (i + 1)])
        data_list.append(self.data[length * (k - 1):])

        for i in range(k):
            train = []
            test = data_list[i]
            for j in (set(range(k)) - {i}):
                train.extend(data_list[j])
            yield train, test


if __name__ == '__main__':
    tr = pd.read_csv('train.csv')
    pr = Processing(tr)

    tr = pr.data_processing()
    d = Divide(pr.get_data())
    for data in d.cross_validation(5):
        print(len(data[0]), len(data[1]))

