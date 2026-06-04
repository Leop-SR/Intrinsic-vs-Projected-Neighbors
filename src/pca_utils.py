from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
def fit_pca(df: pd.DataFrame, n_comp: int):
    pca = PCA(n_components = n_comp)
    df_pca = pca.fit_transform(df)
    return pca, df_pca

def pca_loadings(pca: PCA, feature_names):
    return pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

def explained_variance(pca: PCA):
    var_p = pca.explained_variance_ratio_
    cumulative_var = var_p.sum()
    return var_p, cumulative_var

