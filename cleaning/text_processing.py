import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

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

def deEmojify(text: str):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def text_preprocessing(text: str) -> str:
    """
    Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    text = text.lower() # We transform the text to lowercase
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Remove Punctuations
    text = re.sub('\n', '', text) # We remove the return to line
    text = re.sub('\w*\d\w*', '', text) # Remove digits and words containing digits
    return text

def reduce_repeated_letters(text):
    # Define the pattern to match any letter repeated 3 or more times
    pattern = re.compile(r'(\w)\1{2,}')

    # Use the sub() function to replace occurrences of the pattern with just one letter
    result = pattern.sub(r'\1', text)

    return result

def preprocess_all_text(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df["content"] = df["content"].apply(lambda x: deEmojify(x))
    df["content"] = df["content"].apply(lambda x: text_preprocessing(x))
    df["content"] = df["content"].apply(lambda x: reduce_repeated_letters(x))
    return df

def get_eng_stopwords():
    nltk.download('stopwords')
    eng_stop_words = stopwords.words('english')
    return eng_stop_words

def get_contractions(contractions_dict):
    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
    return contractions_re

def get_eng_letters():
    eng_letters = list(string.ascii_lowercase)
    return eng_letters