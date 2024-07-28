from pandas import DataFrame


def minMaxScale(df: DataFrame) -> DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def zScoreNorm(df: DataFrame) -> DataFrame:
    return (df - df.mean()) / df.std()
