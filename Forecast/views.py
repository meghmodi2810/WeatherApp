import os

from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from django.conf import settings

from django.http import HttpResponse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from datetime import datetime, timedelta

import pytz
from pytz import timezone
# Create your views here.

API_KEY = 'a6632f7c2c6f9fafb60831c544cf7dd4'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'
UTC = pytz.utc
INDIA = pytz.timezone('Asia/Kolkata')

def get_current_weather(city):
  """
  Fetches the current weather data for a given city using the OpenWeatherMap API.

  Args:
    city: The name of the city.

  Returns:
    A dictionary containing the current weather data, or None if an error occurs.
  """
  url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
  response = requests.get(url)
  data = response.json()

  if response.status_code != 200:
    print("Failed to get data:", data)
    return None

  print("Sunrise raw:", data['sys']['sunrise'])
  print("Sunrise IST:", datetime.fromtimestamp(data['sys']['sunrise'], UTC).astimezone(INDIA))
  return {
      'city': data['name'],
      'current_temp' : round(data['main']['temp']),
      'feels_like' : round(data['main']['feels_like']),
      'temp_min' : round(data['main']['temp_min']),
      'temp_max' : round(data['main']['temp_max']),
      'humidity' : round(data['main']['humidity']),
      'description' : data['weather'][0]['description'],
      'country' : data['sys']['country'],
      'wind_gust_dir' : data['wind']['deg'],
      'pressure' : data['main']['pressure'],
      'Wind_Gust_Speed' : data['wind']['speed'],
      'sunrise': datetime.fromtimestamp(data['sys']['sunrise'], UTC).astimezone(INDIA).strftime("%H:%M"),
'sunset': datetime.fromtimestamp(data['sys']['sunset'], UTC).astimezone(INDIA).strftime("%H:%M"),
      'wind_speed' : round(data['wind']['speed']),
      'clouds' : data['clouds']['all'],
      'visibility' : data['visibility'],
  }

def read_historical_data(filename):
  df = pd.read_csv(filename)
  df = df.dropna()
  df = df.drop_duplicates()
  return df

def prepare_data(data):
  le = LabelEncoder()
  data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]

  y = data['RainTomorrow']

  return X, y, le

def train_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  X_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, X_pred)
  print(f'Mean Squared Error: {mse}')

  return model

def prepare_regression_data(data, feature):
  ''' X to store the feature values, Y will store target values to predict'''
  X, y = [], []

  for i in range(len(data) - 1):
    X.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i+1])

  return np.array(X).reshape(-1, 1), np.array(y)



def train_regression_model(X, y):
  model = RandomForestRegressor(random_state=42, n_estimators=100)
  model.fit(X, y)

  return model

def predict_future(model, curr):
  predictions = [curr]

  for i in range(5):
    next = model.predict(np.array([[predictions[-1]]]))
    predictions.append(next[0])

  return predictions[1:]



def weather_view(request):
    city = "Surat"
    if request.method == "POST":
        city_input = request.POST.get('city')
        if city_input:
            city = city_input

    current_weather = get_current_weather(city)

    if current_weather is None:
        return render(request, 'weather.html', {
            'location': city,
            'error': f"Could not retrieve weather for '{city}'"
        })

    try:
        csv_path = os.path.join(settings.BASE_DIR, 'Forecast', 'static', 'data', 'weather.csv')
        historical_data = read_historical_data(csv_path)
        X, y, le = prepare_data(historical_data)
        model = train_model(X, y)

        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['wind_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        df = pd.DataFrame([current_data])
        rain_predict = model.predict(df)[0]

        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        temp_model = train_regression_model(X_temp, y_temp)
        X_humidity, y_humidity = prepare_regression_data(historical_data, 'Humidity')
        humidity_model = train_regression_model(X_humidity, y_humidity)

        future_temp = predict_future(temp_model, current_weather['current_temp'])
        future_humidity = predict_future(humidity_model, current_weather['humidity'])

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'city': current_weather['city'],
            'country': current_weather['country'],
            'time': now,
            'date': now.strftime("%B %d, %Y"),
            'wind': current_weather['Wind_Gust_Speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            'sunrise': current_weather['sunrise'],
            'sunset': current_weather['sunset'],
            'time1': future_times[0], 'time2': future_times[1], 'time3': future_times[2],
            'time4': future_times[3], 'time5': future_times[4],
            'temp1': round(future_temp[0], 1), 'temp2': round(future_temp[1], 1),
            'temp3': round(future_temp[2], 1), 'temp4': round(future_temp[3], 1), 'temp5': round(future_temp[4], 1),
            'humidity1': round(future_humidity[0], 1), 'humidity2': round(future_humidity[1], 1),
            'humidity3': round(future_humidity[2], 1), 'humidity4': round(future_humidity[3], 1), 'humidity5': round(future_humidity[4], 1),
        }

        return render(request, 'weather.html', context)

    except Exception as e:
        return render(request, 'weather.html', {
            'location': city,
            'error': f"Something went wrong: {str(e)}"
        })
