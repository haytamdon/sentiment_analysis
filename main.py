import pandas as pd
import json
from cleaning.data_cleaning import (fix_datetime_column,
                                    fix_type_column,
                                    remove_empty_rows)

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
    rating_data = fix_datetime_column(rating_data, "date")
    rating_data = fix_type_column(rating_data, "tags")
    filtered_rating_data = remove_empty_rows(rating_data, "ratings")
    filtered_rating_data = fix_type_column(filtered_rating_data, "ratings")
    