import os

import pandas as pd

dir_path = f"{os.path.dirname(__file__)}/ctr"


def load_display_advertising_challenge_df():
    """
    https://www.kaggle.com/c/criteo-display-ad-challenge/data

    Data fields
    Label - Target variable that indicates if an ad was clicked (1) or not (0).
    I1-I13 - A total of 13 columns of integer features (mostly count features).
    C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.
    """
    return pd.read_csv(f"{dir_path}/display_advertising_challenge_train.csv")


def judgeGender(gender):
    if gender == '1':  # male
        return 1
    elif gender == '2':  # female
        return 2
    else:
        return 0


def loadLabelEncoder(address):
    with open(address, 'rb') as f:
        age = pkl.load(f)
        gender = pkl.load(f)
        province = pkl.load(f)
        label = pkl.load(f)
    return age, gender, province, label


def bias_score(true_label, predict_pro):
    ctr_gt = np.mean(true_label)
    ctr_predict = np.mean(predict_pro)
    ans = ctr_predict / ctr_gt - 1
    return ans
