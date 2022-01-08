import numpy as np
import pandas as pd
import sys
from ensemble_svm_adaptive_weighting import Ensemble_SVM
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import yaml

if __name__ == "__main__":
    # use saved data
    data = pd.read_csv("data/data.csv")
    test_ind = round(len(data) * 0.7)

    beta = 0.01

    model_infos = []
    with open(r'config/config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
        for key, value in config_dict.items():
            model_infos.append(value)

    ensemble = Ensemble_SVM(model_infos, beta)
    y_pred = []

    for idx in range(test_ind, len(data)):
        results = ensemble.predict(data, idx)
        y_pred.append(np.argmax(results)-1)

    CM = confusion_matrix(data['label'].iloc[test_ind:], y_pred)
    print('Confusion matrix : \n',CM)

    with open('data/ensemble_svm.txt', 'w') as f:
        sys.stdout = f
        score = metrics.accuracy_score(data['label'].iloc[test_ind:], y_pred)
        print("---Optimization SVM---")
        print(" score is:", score)
        print('Confusion matrix : \n', CM)