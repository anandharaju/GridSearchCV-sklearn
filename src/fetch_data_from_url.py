import pandas as pd

def fetch_data_from_url():
    # dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    data = pd.read_csv(url, header=None)
    print(data.describe())

    # split into input (X) and output (Y) variables
    Y = data[8]
    X = data.drop(columns=[8])
    print(X.head(), Y.head())
    print(X.shape, Y.shape)

    return X, Y
