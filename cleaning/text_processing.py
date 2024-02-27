import pandas as pd

def get_data_per_language(df: pd.core.frame.DataFrame,
                        language: str) -> pd.core.frame.DataFrame:
    """
    Returns a subset of the data with one sepecific language

    Args:
        df (pd.core.frame.DataFrame): original dataframe
        language (str): language to return

    Returns:
        pd.core.frame.DataFrame: data subset with the specified language
    """
    language_df = df.loc[df['language']==language]
    return language_df

def separate_text_by_language(df: pd.core.frame.DataFrame)-> pd.core.frame.DataFrame:
    """
    Some reviews are translated from arabic to english so we keep both reviews 
    to augment the data for the modelling but each language in a seperate row

    Args:
        df (pd.core.frame.DataFrame): original data

    Returns:
        pd.core.frame.DataFrame: transformed data
    """
    for i in range(len(df)):
        # the translated reviews start with the line More(Translated by Google)
        if df.iloc[[i]]['content'].values[0].startswith("More(Translated by Google)"):
            # We get the review's text
            txt = df.iloc[[i]]['content'].values[0]
            # the review that is translated takes the following form
            # More(Translated by Google) english text (Original) arabic text
            # We split the text by the key word '(Original)' in the middle
            arabic_txt = txt.split(" (Original) ")[1] # We get the arabic text 
            # We extract the english text and we remove the beginning keywords 
            # since they offer no extra informations
            english_txt = txt.split(" (Original) ")[0].split('More(Translated by Google) ')[1] 
            # We update the row with the arabic text
            df.at[i, 'content'] = arabic_txt
            df.at[i, 'language'] = 'ara'
            # We create a new row with the english text only
            row_dict = df.iloc[i].to_dict()
            row_dict["content"] = english_txt
            row_dict["language"] = "eng"
            # We append this new row to the dataframe
            df = df._append(row_dict, ignore_index = True)
        elif df.iloc[[i]]['content'].values[0].startswith("moretranslated by google"):
            # the translated reviews can also start with the line moretranslated by google
            # We apply the same transformations
            txt = df.iloc[[i]]['content'].values[0]
            arabic_txt = txt.split(" original ")[1]
            english_txt = txt.split(" original ")[0].split('moretranslated by google ')[1]
            df.at[i, 'content'] = arabic_txt
            df.at[i, 'language'] = 'ara'
            row_dict = df.iloc[i].to_dict()
            row_dict["content"] = english_txt
            row_dict["language"] = "eng"
            df = df._append(row_dict, ignore_index = True)
    # We reset the index for smoother manipulations
    df = df.reset_index(drop = True)
    return df