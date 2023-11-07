import requests

def get_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
    r = requests.get(url)

    data = r.json()
    print(data)
    elevation = data['results'][0]['elevation']
    print(elevation)

def get_soil_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}"
    response = requests.get(url)
    data= response.json()
    print(data)

# Example usage
lat = 35.689487
lon = 139.691711
print( get_soil_data(lat, lon) )
get_elevation(lat, lon)

