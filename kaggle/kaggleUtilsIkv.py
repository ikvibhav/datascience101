import pandas as pd


def perform_explore(df: pd.DataFrame) -> None:
    print(f"------df.head()--------")
    print(df.head())
    print(f"------df.info()--------")
    print(df.info())
    print(f"------df.describe()--------")
    print(df.describe())
    print(f"------df.shape--------")
    print(df.shape)


def perform_correlation_matrix(
    df: pd.DataFrame, feature_list: list = None
) -> pd.DataFrame | None:
    try:
        return pd.get_dummies(df[feature_list]).corr()
    except Exception as e:
        print(f"Error: {e}")
        return None
