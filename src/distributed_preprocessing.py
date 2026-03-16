import dask.dataframe as dd


def to_dask_dataframe(df, n_partitions=10):
    """
    Convert a pandas DataFrame to a Dask DataFrame.
    """
    return dd.from_pandas(df, npartitions=n_partitions)


def split_features_target(ddf, target_col="target"):
    """
    Split a distributed DataFrame into features and target.
    """
    X = ddf.drop(columns=[target_col])
    y = ddf[target_col]
    return X, y
