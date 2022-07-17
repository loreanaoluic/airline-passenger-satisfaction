import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path):
    csv_data = pd.read_csv(file_path)
    return csv_data


def preprocess_data(dataset):
    dataset.drop(["Arrival Delay"], axis="columns", inplace=True)
    dataset.drop(["ID"], axis="columns", inplace=True)

    x_dataset = dataset.drop("Satisfaction", axis="columns")
    print("\t\tX DATASET\n\n", x_dataset, "\n\n\n")

    y_dataset = dataset["Satisfaction"]
    print("\t\tY DATASET\n\n", y_dataset, "\n\n\n")

    processed_x_dataset = label_encoding(x_dataset)

    x_train, x_test, y_train, y_test = train_test_split(processed_x_dataset, y_dataset, train_size=0.75,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def label_encoding(x_dataset):

    x_dataset['Gender'] = x_dataset['Gender'].replace(
        {'Male': 1, 'Female': 0})

    x_dataset['Customer Type'] = x_dataset['Customer Type'].replace(
        {'Returning': 0, 'First-time': 1})

    x_dataset['Type of Travel'] = x_dataset['Type of Travel'].replace(
        {'Business': 0, 'Personal': 1})

    x_dataset['Class'] = x_dataset['Class'].replace(
        {'Business': 0, 'Economy': 1, 'Economy Plus': 2})

    print("\t\tX DATASET AFTER LABEL ENCODING\n\n", x_dataset, "\n\n\n")

    return x_dataset


def model(x_train, x_test, y_train, y_test):

    # SVM
    SVM = LinearSVC(C=0.2)
    SVM.fit(x_train, y_train)
    predicted = SVM.predict(x_test)
    score = f1_score(predicted, y_test, average='micro')
    print('SVM accuracy: ', score)

    # RANDOM FOREST
    RF = RandomForestClassifier()
    RF.fit(x_train, y_train)
    pred = RF.predict(x_test)
    score = f1_score(pred, y_test, average='micro')
    print('Random forest accuracy: ', score)


if __name__ == "__main__":
    data = load_data(sys.argv[1])
    xtrain, xtest, ytrain, ytest = preprocess_data(data)
    model(xtrain, xtest, ytrain, ytest)
