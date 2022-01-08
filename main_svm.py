import pandas as pd
from svm import SVM_model
import yaml

if __name__ == "__main__":
    ##read and preprocess data

    ## use saved data
    data = pd.read_csv("data/data.csv")

    model_infos = []
    with open(r'config/config.yaml') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
        for key, value in config_dict.items():
            model_infos.append(value)

    train_ind = round(len(data) * 0.7)
    for info in model_infos:
        svm = SVM_model(info, training=True)
        x_train, y_train = svm.construct_data(data.iloc[:train_ind])
        model = svm.train(x_train, y_train)
        svm.save_model("data/" + info["model_name"], model)