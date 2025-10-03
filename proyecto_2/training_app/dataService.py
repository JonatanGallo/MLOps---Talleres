import requests

RAW_COLUMN_NAMES = [
    'Elevation', 
    'Aspect', 
    'Slope', 
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area',
    'Soil_Type',
    'Cover_Type']

URL = 'http://10.43.100.103:8080/data?group_number=4'

def get_raw_column_names():
    return RAW_COLUMN_NAMES

def fetch_data():
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()['data']
        batch_number = response.json()['batch_number']
        print(f"✅ Batch number {batch_number}")
        print(f"✅ Data {data}")
        return data, batch_number
    else:
        print(f"❌ Failed to fetch data: {response.status_code}")

# def store_raw_data():
#   rawData = pd.DataFrame(fetch_data(), columns=get_raw_column_names())
#   create_table("raw_data", rawData)
#   insert_data("raw_data", rawData)

# print("raw data")
# store_raw_data()