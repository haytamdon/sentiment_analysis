import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from typing import List

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

def get_contractions_dict():
    # Dictionary of English Contractions
    contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                        "can't": "cannot","can't've": "cannot have",
                        "'cause": "because","could've": "could have","couldn't": "could not",
                        "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                        "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                        "hasn't": "has not","haven't": "have not","he'd": "he would",
                        "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                        "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                        "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                        "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                        "it'd": "it would","it'd've": "it would have","it'll": "it will",
                        "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                        "mayn't": "may not","might've": "might have","mightn't": "might not",
                        "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not",
                        "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                        "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                        "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                        "she'll": "she will", "she'll've": "she will have","should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                        "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                        "there'd've": "there would have", "they'd": "they would",
                        "they'd've": "they would have","they'll": "they will",
                        "they'll've": "they will have", "they're": "they are","they've": "they have",
                        "to've": "to have","wasn't": "was not","we'd": "we would",
                        "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                        "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                        "what'll've": "what will have","what're": "what are", "what've": "what have",
                        "when've": "when have","where'd": "where did", "where've": "where have",
                        "who'll": "who will","who'll've": "who will have","who've": "who have",
                        "why've": "why have","will've": "will have","won't": "will not",
                        "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                        "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                        "y'all'd've": "you all would have","y'all're": "you all are",
                        "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                        "you'll": "you will","you'll've": "you will have", "you're": "you are",
                        "you've": "you have"}
    return contractions_dict

def replace(match):
    contractions_dict = get_contractions_dict()
    return contractions_dict[match.group(0)]

# Function for expanding contractions
def expand_contractions(text: str, contractions_re):
    return contractions_re.sub(replace, text)

def remove_eng_stop_words(text, eng_stop_words):
    text = ' '.join(word for word in text.split() if word not in eng_stop_words)
    return text

def preprocess_english_text(df: pd.core.frame.DataFrame,
                            eng_stop_words: List[str],
                            contractions_re):
    english_df = get_data_per_language(df, language="eng")
    english_df['content'] = english_df['content'].apply(lambda x: expand_contractions(x, contractions_re))
    english_df['content'] = english_df['content'].apply(lambda x: remove_eng_stop_words(x, eng_stop_words))
    english_df = english_df.reset_index(drop = True)
    return english_df

def get_punctuations():
    punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
    return punctuations

def get_ara_stopwords():
    arab_stop_words = stopwords.words('arabic')
    return arab_stop_words

def preprocess(text: str,
                punctuations: str,
                arab_stop_words: List[str]):
    """
    text is an arabic string input

    remove punctuations and arabic stop words

    Args:
        text (str): the input text
        punctuations (str): a string containing all
                            the standard punctuations
        arab_stop_words (List[str]): list of all the
                                    arabic stop words

    Returns:
        _type_: _description_
    """
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    text = ' '.join(word for word in text.split() if word not in arab_stop_words)

    return text

def arabic_cleaning(data):
    data['content'] = data['content'].str.replace('[^\u0621-\u064A\u0660-\u0669 ]', '', regex=True)
    # We remove the arabic digits
    data['content'] = data['content'].str.replace('/[\u0660-\u0669]/', '', regex=True)
    # Keeps only the reviews with arabic letters
    data = data[data['content'].str.contains('^[\u0621-\u064A\u0660-\u0669]')]
    return data

def preprocess_arabic_text(df,
                        punctuations,
                        arab_stop_words):
    arabic_df = get_data_per_language(df, language="ara")
    arabic_df["content"] = arabic_df["content"].apply(lambda x: preprocess(x,
                                                                            punctuations,
                                                                            arab_stop_words))
    arabic_df = arabic_cleaning(arabic_df)
    arabic_df = arabic_df.reset_index(drop=True)
    return arabic_df