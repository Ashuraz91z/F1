## juste recup les circuits de cette année


import requests
import xml.etree.ElementTree as ET
import pandas as pd


url = "http://ergast.com/api/f1/2024/circuits"


response = requests.get(url)
xml_content = response.content


root = ET.fromstring(xml_content)

namespace = {'ns': 'http://ergast.com/mrd/1.5'}


circuits = root.findall('.//ns:Circuit', namespace)


all_latitudes = []
all_longitudes = []


for circuit in circuits:
    circuit_name = circuit.find('ns:CircuitName', namespace).text
    location = circuit.find('ns:Location', namespace)
    latitude = location.get('lat')
    longitude = location.get('long')

    all_latitudes.append(float(latitude))
    all_longitudes.append(float(longitude))


csv_path = "data/csv/circuits.csv"
df = pd.read_csv(csv_path)

df['lat'] = df['lat'].astype(float)
df['lng'] = df['lng'].astype(float)


filtered_df = df[(df['lat'].isin(all_latitudes)) & (df['lng'].isin(all_longitudes))]



filtered_df.to_csv("data/clean/circuits_filtered.csv", index=False)
