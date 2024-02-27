import pandas as pd
import datetime
import ast
from typing import List

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

def split_ratings(rating_df: pd.core.frame.DataFrame,
                    col_name: str) -> pd.core.frame.DataFrame:
    """
    splits the ratings columns into the 2 columns
    normalized ratings and raw ratings and removes
    the original column

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe to be transformed
        col_name (str): name of the ratings column

    Returns:
        pd.core.frame.DataFrame: transformed dataframe
    """
    rating_df["normalized_rating"] = rating_df[col_name].apply(lambda x: x['normalized'])
    rating_df["raw_rating"] = rating_df[col_name].apply(lambda x: x['raw'])
    rating_df = rating_df.drop(col_name, axis=1)
    return rating_df

def get_tag_and_sentiment(rating_df: pd.core.frame.DataFrame,
                            col_name: str) -> pd.core.frame.DataFrame:
    """
    Returns a dataframe with the exact tags and sentiments

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe to be transformed
        col_name (str): name of the tags column

    Returns:
        pd.core.frame.DataFrame: transformed dataframe
    """
    rating_df["sentiment"] = rating_df[col_name].apply(lambda x: [element['sentiment'] for element in x])
    rating_df[col_name] = rating_df[col_name].apply(lambda x: [element['value'] for element in x])
    return rating_df

def map_tags(rating_df: pd.core.frame.DataFrame,
                mapping_json: dict,
                col_name: str) -> pd.core.frame.DataFrame:
    """
    maps tags ids to their appropriate values

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe containing the reviews
        mapping_json (dict): a mapping dict/json containing the ids with their
                            associated values
        col_name (str): name of the tags column

    Returns:
        pd.core.frame.DataFrame: mapped dataframe
    """
    mappers = mapping_json['tags_mapping']
    rating_df['transformed_tags'] = rating_df[col_name].apply(lambda x: [mappers[element] for element in x])
    return rating_df

def get_city(rating_df: pd.core.frame.DataFrame,
                col_name: str) -> pd.core.frame.DataFrame:
    """
    get the city value from the mapped tags

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe to be transformed
        col_name (str): name of the tags column

    Returns:
        pd.core.frame.DataFrame: transformed dataframe
    """
    # the tags column is composed of nested lists
    # the city value is the second element of each
    # of these lists
    rating_df["city"] = rating_df[col_name].apply(lambda x: [li[1] for li in x])
    return rating_df

def get_location_type(rating_df: pd.core.frame.DataFrame,
                col_name: str) -> pd.core.frame.DataFrame:
    """
    get the location type from the mapped tags

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe to be transformed
        col_name (str): name of the tags column

    Returns:
        pd.core.frame.DataFrame: transformed dataframe
    """
    # the tags column is composed of nested lists
    # the location type is the first element of each
    # of these lists
    rating_df["type"] = rating_df[col_name].apply(lambda x: [li[0] for li in x])
    return rating_df

def reformat_city(name_list: List[str]):
    """
    reformating the city name lists

    Args:
        name_list (List[str]): list of the city names

    Returns:
        List[str]: the fomatted names
    """
    if len(name_list)==1:
        # if the list is composed of only 1 city we
        # return that city
        return name_list[0]
    elif name_list == [name_list[0]]*len(name_list):
        # if the list is repetitive of only one element
        # we return that city
        return name_list[0]
    else:
        return name_list

def reformat_city_column(rating_df: pd.core.frame.DataFrame,
                        col_name: str) -> pd.core.frame.DataFrame:
    """
    reformating the city column

    Args:
        rating_df (pd.core.frame.DataFrame): dataframe of the cities
        col_name (str): name of the cities column

    Returns:
        pd.core.frame.DataFrame: formatted dataframe
    """
    rating_df[col_name] = rating_df[col_name].apply(lambda x: reformat_city(x))
    return rating_df