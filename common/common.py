from pandas import DataFrame


def min_max_Scale(df: DataFrame) -> DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def z_score_Norm(df: DataFrame) -> DataFrame:
    return (df - df.mean()) / df.std()
