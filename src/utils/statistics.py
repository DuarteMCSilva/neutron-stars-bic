from pandas import DataFrame
import numpy as np

def normalize(data: DataFrame) :
    mean = data.mean()
    std = data.std()
    return (data - mean)/std

def inv_normalize(data: DataFrame, mean , std ):
    return data*std + mean 

def normalizeWithReference(data: DataFrame, ref_data: DataFrame) :
    mean = ref_data.mean()
    std = ref_data.std()
    return (data - mean)/std

def build_stats_dataframe(df: DataFrame) -> DataFrame:
    return DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'max': df.max(),
        'min': df.min()
    })

def generate_normal(size: int, mean=0, std=1) -> np.array:
    return np.random.randn(size)*std+mean
