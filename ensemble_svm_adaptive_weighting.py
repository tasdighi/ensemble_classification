from svm import SVM_model
import numpy as np
import pandas as pd
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix

class Ensemble_SVM:

    def __init__(self, model_infos, beta):
        self.beta = beta
        self.models = []
        for info in model_infos:
            self.models.append(SVM_model(info))
        self.model_weights = np.ones(len(self.models))/len(self.models)

    def predict(self, data, idx):
        results = np.zeros((3,), dtype=int)
        model_acc = np.zeros(len(self.models))
        for model_idx, model in enumerate(self.models):
            pred, y = model.model_predict(data, idx)
            model_acc[model_idx] = model_acc[model_idx] * (1-self.beta) + (pred == y) * self.beta
            results[pred + 1] += self.model_weights[model_idx]

        self.model_weights = model_acc / sum(model_acc) if sum(model_acc) != 0 else np.ones(len(self.models))/len(self.models)
        return results


if __name__ == "__main__":
    # use saved data
    data = pd.read_csv("data/data.csv")
    test_ind = round(len(data) * 0.7)

    beta = 0.01
    model_infos = []
    model_features_dictionary = dict()
    model_features_dictionary = {
        "model_name": "svm_rbf_win35.sav",
        "C": 10 ** -2.94383391282954,
        "gamma": 10 ** -3.9080360544530914,
        "window": 35,
        "kernel": "'rbf",
        "feature_names": [],
        "history_features": []
    }
    model_infos.append(model_features_dictionary)
    model_features_dictionary = {
        "model_name": "svm_rbf_win29.sav",
        "C": 10 ** 0.9436507147037276,
        "gamma": 10 ** -3.0494916334933864,
        "window": 29,
        "kernel": "'rbf",
        "feature_names": [],
        "history_features": []
    }
    model_infos.append(model_features_dictionary)
    ensemble = Ensemble_SVM(model_infos, beta)
    y_pred = []

    for idx in range(test_ind, len(data)):
        results = ensemble.predict(data, idx)
        y_pred.append(np.argmax(results))

    with open('data/ensemble_svm.txt', 'w') as f:
        sys.stdout = f
        score = metrics.accuracy_score(data['label'].iloc[test_ind:], y_pred)
        print("---Optimization SVM---")
        print(" score is:", score)
        CM = confusion_matrix(data['label'].iloc[test_ind:], y_pred)
        print('Confusion matrix : \n', CM)