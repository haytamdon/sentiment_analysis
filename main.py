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

if __name__ == "__main__":
    # Reading the data
    rating_data = pd.read_csv("dataset.csv_(DS_A-L2).csv")
    json_file = open("mappings.json_(DS_A-L2).json")
    mapping_data = json.load(json_file)
    
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