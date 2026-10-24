import requests

url = "https://api.open-meteo.com/v1/forecast?latitude=31.52&longitude=74.36&current_weather=true"
response = requests.get(url)
#print(response.status_code)

data = response.json()

#print(data)

print("Windspeed", data["current_weather"]["windspeed"])

print("Temperature", data["current_weather"]["temperature"])