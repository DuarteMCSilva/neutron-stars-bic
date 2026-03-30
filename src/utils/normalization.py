from pandas import DataFrame

def normalize(data: DataFrame) :
    mean = data.mean()
    std = data.std()
    return (data - mean)/std

def normalizeWithReference(data: DataFrame, ref_data: DataFrame) :
    mean = ref_data.mean()
    std = ref_data.std()
    return (data - mean)/std
