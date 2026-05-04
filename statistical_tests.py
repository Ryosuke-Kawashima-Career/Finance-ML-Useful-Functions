# ADF検定用パッケージ
from statsmodels.tsa.stattools import adfuller

def adf_test(df: pd.Series) -> bool:
    """
    Test the stationality of the input series

    Args:
        df (pd.DataFrame) : 系列データ

    Returns:
        bool : Is Stationary(定常か否かのbool値)
    """
    # adfullerを用いて, 入力dfに対するadf検定を実施します.
    df = df.dropna()
    sig_level = 0.05 # Significance
    adfuller_result = adfuller(df)
    return adfuller_result[1] < sig_level