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
                                    compute_sentiment_col,
                                    sentiment_encoder,
                                    filter_data,
                                    col_rename)
from cleaning.text_processing import (get_contractions, 
                                        separate_text_by_language,
                                        preprocess_all_text,
                                        preprocess_arabic_text,
                                        preprocess_english_text,
                                        get_ara_stopwords,
                                        get_punctuations,
                                        get_eng_stopwords,
                                        get_contractions_dict)

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

# Label encoding dict
label_enc = {
    "negative": 2,
    "neutral": 1,
    "positive": 0
}

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
    
    # Process all text
    text_processed_data = preprocess_all_text(correct_language_rating_data)
    
    # Process arabic text
    punctuations = get_punctuations()
    arab_stop_words = get_ara_stopwords()
    arabic_processed_data = preprocess_arabic_text(text_processed_data,
                                                    punctuations,
                                                    arab_stop_words)
    
    # Process english text
    eng_stop_words = get_eng_stopwords()
    contractions_dict = get_contractions_dict()
    contractions_re = get_contractions(contractions_dict)
    english_processed_data = preprocess_english_text(text_processed_data,
                                                    eng_stop_words,
                                                    contractions_re)
    
    # Concat all data
    full_processed_data = pd.concat([arabic_processed_data, 
                                    english_processed_data]).reset_index(drop=True)
    
    # Create a class column
    full_processed_data = sentiment_encoder(full_processed_data, label_enc)
    # Isolate only the wanted columns
    final_data = filter_data(full_processed_data)
    # Rename columns to conventional names
    final_data = col_rename(final_data)
    
    