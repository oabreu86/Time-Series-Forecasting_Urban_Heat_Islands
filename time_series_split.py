import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def create_test_df():
    ts_2015 = pd.date_range('2015-01-01', '2015-12-31', periods=4).to_series()
    ts_2016 = pd.date_range('2016-01-01', '2016-12-31', periods=12).to_series()
    ts_2017 = pd.date_range('2017-01-01', '2017-12-31', periods=6).to_series()
    ts_2018 = pd.date_range('2018-01-01', '2018-12-31', periods=8).to_series()
    ts_2019 = pd.date_range('2019-01-01', '2019-12-31', periods=24).to_series()
    ts_2020 = pd.date_range('2020-01-01', '2020-12-31', periods=30).to_series()
    ts_all = pd.concat([ts_2015, ts_2016, ts_2017, ts_2018, ts_2019, ts_2020])
    df = pd.DataFrame({'X': np.random.randint(0, 100, size=ts_all.shape), 
                   'Y': np.random.randint(100, 200, size=ts_all.shape)},
                 index=ts_all)
    #df['year'] = df.index.year
    #df['month'] = df.index.month
    #df = df.reset_index()
    return df

def time_train_test_split(df, n_splits):
    """
    Splits a dataframe into training, validation, and testing sets for time series
    machine learning.

	Inputs:
		df(Dataframe): dataframe structure with features for machine learning
            splits; must have a 'Year' column or have the index be a datetime object 
            and must have at least three years of data for the splits.

        n_splits (int): number of splits by years of data; typically = 
            number of years - 2

	Outputs:
        splits (dict): a dictionary of key:value pairs with each component
        representing a sequentially increasing (by years) set of dataframes 
        to be used as training/validation and testing sets.
    """

    if (n_splits < 3):
        raise ValueError("Number of splits must be (3) or more.")

    if "Year" not in df.columns:
        if is_datetime(df.index):
            df['Year'] = df.index.year
            df = df.reset_index()
        else:
            raise TypeError("Index is not of datetime type to create year column.")
    
    year_list = df['Year'].unique().tolist()
    splits = {'train': [], 'validation':[], 'test': []}

    for idx, yr in enumerate(year_list[:-2]):
        train_yr = year_list[:idx+1]
        valid_yr = year_list[idx+1:idx+2]
        test_yr = [year_list[idx+2]]
        print('TRAIN: {}, VALIDATION: {}, TEST: {}'.format(train_yr, valid_yr, test_yr))
        
        splits['train'].append(df.loc[df.Year.isin(train_yr), :])
        splits['validation'].append(df.loc[df.Year.isin(valid_yr), :])
        splits['test'].append(df.loc[df.Year.isin(test_yr), :])
    return splits