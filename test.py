import pandas as pd


def test(n):
    while n > 0:
        yield n
        n -= 1


if __name__ == '__main__':
    tr = pd.read_csv('train.csv')
    # for fea in tr.
    print(tr['Sex'].value_counts())
    print(tr['Age'].mean())
    print(tr.keys())

    a = pd.Series([1, 1, 1, 1, 0, 0, 0])
    print(a.value_counts())
