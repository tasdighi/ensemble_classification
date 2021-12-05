from sklearn import svm
import pickle
import pandas as pd

class SVM_model:

    def __init__(self, model_infos, training=False):
        self.gamma = model_infos["gamma"]
        self.C = model_infos["C"]
        self.kernel = model_infos["kernel"]
        self.feature_names = model_infos["feature_names"]
        self.history_features = model_infos["history_features"]
        self.window = model_infos["window"]
        self.model_name = model_infos["model_name"]
        if not training:
            self.load_model(self.model_name)

    def construct_data(self, data):
        x = data.drop('label', 1)
        y = data['label']
        return x, y

    def model_predict(self, data, idx):
        x, y = self.constuct_data(data, idx)
        pred = self.model.predict(x)
        return pred[0], y

    def train(self, x_train, y_train):
        model = svm.SVC(kernel = self.kernel, C = self.C, gamma = self.gamma)
        model.fit(x_train, y_train)
        return model

    def save_model(self, picke_path, model):
        # save the model to disk
        filename = picke_path + '.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self, picke_path):
        # load the model from disk
        self.model = pickle.load(open("data/" + picke_path, 'rb'))


if __name__ == "__main__":
    # read and preprocess data

    # use saved data
    data = pd.read_csv("data/data.csv")

    model_infos = []
    model_features_dictionary = dict()
    model_features_dictionary = {
        "model_name": "svm_rbf_win35",
        "C": 10 ** -2.94383391282954,
        "gamma": 10 ** -3.9080360544530914,
        "window": 35,
        "kernel": "rbf",
        "feature_names": [],
        "history_features": []
    }
    model_infos.append(model_features_dictionary)
    model_features_dictionary = {
        "model_name": "svm_rbf_win29",
        "C": 10 ** 0.9436507147037276,
        "gamma": 10 ** -3.0494916334933864,
        "window": 29,
        "kernel": "rbf",
        "feature_names": [],
        "history_features": []
    }

    model_infos.append(model_features_dictionary)
    train_ind = round(len(data) * 0.7)
    for info in model_infos:
        svm = SVM_model(info, training=True)
        x_train, y_train = svm.construct_data(data.iloc[:train_ind])
        model = svm.train(x_train, y_train)
        svm.save_model("data/" + info["model_name"], model)
        