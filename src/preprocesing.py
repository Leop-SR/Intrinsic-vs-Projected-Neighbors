import pandas as pd
def robust_scaling (df: pd.DataFrame) -> pd.DataFrame:
    scaled_df = df.apply(lambda x: (x-x.median())/(x.quantile(0.75)-x.quantile(0.25)))
    return scaled_df

def standarization (df: pd.DataFrame) -> pd.DataFrame:
    scaled_df = df.apply(lambda x: ((x-x.mean())/x.std()))
    return scaled_df