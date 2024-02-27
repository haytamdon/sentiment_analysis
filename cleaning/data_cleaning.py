import pandas as pd
import datetime
import ast

def fix_datetime_column(rating_df: pd.core.frame.DataFrame,
                        data_col_name: str) -> pd.core.frame.DataFrame:
    """
    converts the date column from string to datetime

    Args:
        rating_df (pd.core.frame.DataFrame): rating_df on which we will apply
                                    the transformations
        data_col_name (str): name of the date column

    Returns:
        pd.core.frame.DataFrame: resulting dataframe
    """
    rating_df[data_col_name] = rating_df[data_col_name].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z"))
    return rating_df

def fix_type_column(rating_df: pd.core.frame.DataFrame,
                        data_col_name: str) -> pd.core.frame.DataFrame:
    """
    Convert columns to their appropriate datatype

    Args:
        rating_df (pd.core.frame.DataFrame): rating_df on which we will apply
                                    the transformations
        data_col_name (str): name of the column to convert

    Returns:
        pd.core.frame.DataFrame: dataframe with fixed datatype
    """
    rating_df[data_col_name] = rating_df[data_col_name].apply(lambda x: ast.literal_eval(x))
    return rating_df

def remove_empty_rows(rating_df: pd.core.frame.DataFrame,
                        data_col_name: str) -> pd.core.frame.DataFrame:
    """
    remove the rows that have null values for a specific column

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe to be filtered
        data_col_name (str): name of the column on which we
                            are filtering

    Returns:
        pd.core.frame.DataFrame: dataframe without null values in
                                the specific column
    """
    # Getting the indices of the all rows on which
    # the column isn't null
    not_null_rows = rating_df[[data_col_name]].notnull().any(axis=1)
    # Taking only these indices
    filtered_rating_df = rating_df[not_null_rows]
    filtered_rating_df = filtered_rating_df.reset_index(drop=True) # Resetting the index col
    return filtered_rating_df