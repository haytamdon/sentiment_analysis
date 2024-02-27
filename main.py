import pandas as pd
import json
from cleaning.data_cleaning import (fix_datetime_column,
                                    fix_type_column,
                                    remove_empty_rows,
                                    fix_incorrect_cities,
                                    get_city,
                                    get_incorrect_cities,
                                    get_location_type,
                                    get_tag_and_sentiment,
                                    map_tags,
                                    split_ratings,
                                    reformat_city_column,
                                    reformat_sentiment_col,
                                    filter_columns,
                                    compute_sentiment_col)
from cleaning.text_processing import (separate_text_by_language)

location_name_to_city = {
    # Mapper of some common locations with their cities
    "Souq Al Zel" : "Riyadh",
    "Ushaiqer Heritage Village": "Ushaiqer",
    "King Salman Park": "Riyadh",
    "King Abdulaziz Center for World Culture - Ithra": "Dhahran",
    "Rawdah Park": "Dammam",
    "Modon Lake Park": "Dammam",
    "Khairah Forest Park": "Al Baha",
    "Wadi Namar Waterfall": "Riyadh",
    "منتجع شاطئ الدانة | Dana Beach Resort": "Dhahran",
    "Holiday Inn Resort Half Moon Bay": "Dhahran",
    "Boudl Al Qasr": "Riyadh",
    "Boudl Al Munsiyah": "Riyadh",
    "Boudl Gardenia Resort": "Riyadh",
    "Boudl Gaber Hotel": "Riyadh"
}
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

if __name__ == "__main__":
    # Reading the data
    rating_data = pd.read_csv("dataset.csv_(DS_A-L2).csv")
    json_file = open("mappings.json_(DS_A-L2).json")
    mapping_data = json.load(json_file)
    
    # Data Cleaning
    # Reformatting columns
    rating_data = fix_datetime_column(rating_data, 
                                    "date")
    rating_data = fix_type_column(rating_data, 
                                "tags")
    filtered_rating_data = remove_empty_rows(rating_data, 
                                            "ratings")
    filtered_rating_data = fix_type_column(filtered_rating_data, 
                                            "ratings")
    
    # Splitting Mixed columns
    transformed_rating_data = split_ratings(filtered_rating_data, 
                                            "ratings")
    transformed_rating_data = get_tag_and_sentiment(transformed_rating_data, 
                                                    "tags")
    
    # Mapping tag ids to their values
    mapped_rating_data = map_tags(transformed_rating_data,
                                mapping_data,
                                "tags")
    
    # Getting Locations
    detailed_rating_data = get_city(mapped_rating_data, 
                                    "transformed_tags")
    detailed_rating_data = get_location_type(detailed_rating_data, 
                                            "transformed_tags")
    
    # Reformating Cities Column
    reformated_df = reformat_city_column(detailed_rating_data, 'city')
    # Fixing Wrong Cities
    incorrect_cities = get_incorrect_cities(reformated_df)
    fixed_cities_rating_data = fix_incorrect_cities(reformated_df, 
                                                    incorrect_cities, 
                                                    location_name_to_city)
    
    # Formatting and filtering the columns
    formatted_rating_data = reformat_sentiment_col(fixed_cities_rating_data)
    filter_rating_data = filter_columns(formatted_rating_data)
    
    # computing sentiments
    correct_sentiment_data = compute_sentiment_col(filter_rating_data)
    main_rating_data = correct_sentiment_data.drop("raw_rating", axis=1)
    
    # Text Processing
    # Fixing Languages
    correct_language_rating_data = separate_text_by_language(main_rating_data)