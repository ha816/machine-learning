import os

import pandas as pd


class DisplayAdvertisingChallenge:
    __CTR_PATH = dir_path = f"{os.path.dirname(__file__)}/ctr"

    @classmethod
    def load_train_and_test_df(cls):
        """
        https://www.kaggle.com/c/criteo-display-ad-challenge/data

        Data fields
        Label - Target variable that indicates if an ad was clicked (1) or not (0).
        I1-I13 - A total of 13 columns of integer features (mostly count features).
        C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.
        """
        return (pd.read_csv(f"{cls.__CTR_PATH}/display_advertising_challenge_train.csv"),
                pd.read_csv(f"{cls.__CTR_PATH}/display_advertising_challenge_test.csv"))

    @classmethod
    def load_df(cls):
        train_df = pd.read_csv(f"{cls.__CTR_PATH}/display_advertising_challenge_train.csv")
        test_df = pd.read_csv(f"{cls.__CTR_PATH}/display_advertising_challenge_test.csv")
        return pd.concat([train_df, test_df])


def train_test_split(df, test_size: float, random_state=None, shuffle=True):
    from sklearn.model_selection import train_test_split
    return tuple(train_test_split(df, test_size=test_size, random_state=random_state, shuffle=shuffle))


def load_display_advertising_challenge_train_test_df():
    """
    https://www.kaggle.com/c/criteo-display-ad-challenge/data

    Data fields
    Label - Target variable that indicates if an ad was clicked (1) or not (0).
    I1-I13 - A total of 13 columns of integer features (mostly count features).
    C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.
    """
    return (pd.read_csv(f"{dir_path}/display_advertising_challenge_train.csv"),
            pd.read_csv(f"{dir_path}/display_advertising_challenge_test.csv"))


def load_display_advertising_challenge_train_test_df():
    """
    https://www.kaggle.com/c/criteo-display-ad-challenge/data

    Data fields
    Label - Target variable that indicates if an ad was clicked (1) or not (0).
    I1-I13 - A total of 13 columns of integer features (mostly count features).
    C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes.
    """
    return (pd.read_csv(f"{dir_path}/display_advertising_challenge_train.csv"),
            pd.read_csv(f"{dir_path}/display_advertising_challenge_test.csv"))
