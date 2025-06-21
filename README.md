Perfect, Megh! Here's a complete and polished **GitHub `README.md`** for your smart Django-based weather forecasting app — written in markdown format and including all key sections (features, setup, usage, screenshots placeholder, and tech stack):

---

```markdown
# 🌦️ Smart Weather Forecasting Web App (Django + ML)

This is a full-stack weather forecasting web application that uses **machine learning models** to predict:

- ✅ Temperature & humidity forecasts (next 5 hours)
- 🌧️ Rain probability for tomorrow
- 📍 Real-time weather for any city (powered by OpenWeatherMap API)
- 🌅 Accurate sunrise and sunset times (IST-aware)

This project was built as my **first Django + ML integration project**, combining frontend interactivity with backend intelligence and data-driven prediction.

---

## 🛠️ Tech Stack

| Area        | Tools Used |
|-------------|------------|
| Backend     | Django, Python 3, REST API integration |
| Frontend    | HTML5, CSS3, Bootstrap Icons, Chart.js |
| ML Models   | RandomForestClassifier, RandomForestRegressor (`scikit-learn`) |
| Data Prep   | `pandas`, `numpy`, `LabelEncoder` |
| API         | [OpenWeatherMap](https://openweathermap.org/api) |
| Timezones   | `pytz`, `datetime` (for IST-conversion) |
| Charting    | `Chart.js` for dynamic forecast charts |

---

## ⚙️ Features

- 🌍 **Enter any city** and fetch its real-time weather
- 📈 Forecast next 5 hours of:
  - Temperature (°C)
  - Humidity (%)
- 🌧️ Predicts **chance of rain** (in %)
- ☀️ Displays **sunrise and sunset times**, localized to Indian Standard Time (IST)
- ⚠️ Invalid city inputs are gracefully handled via alert boxes
- 📊 Live chart of forecast using Chart.js
- 🧠 Machine learning models trained on historical weather data (CSV)
- 🌤️ UI updates dynamically with background and icons based on live weather
- 🧪 Model accuracy (MSE) printed during training (console)

---

## 📂 Project Structure

```

weather\_forecast\_app/
├── Forecast/
│   ├── views.py               # All main ML + weather logic
│   └── templates/
│       └── weather.html       # Jinja2 template
├── static/
│   ├── css/styles.css
│   ├── js/chartSetup.js
│   └── img/                   # Icons, background, logo
├── manage.py
├── WeatherApp/
│   └── settings.py            # Static setup, BASE\_DIR, etc.
├── static/data/weather.csv    # ML training dataset

````

---

## 📸 Screenshots

| Live Forecast | Rain Prediction |
|---------------|-----------------|
| *(Add screenshot here)* | *(Add screenshot here)* |

---

## 🧠 Machine Learning Details

- **RandomForestClassifier**
  - Input: MinTemp, MaxTemp, WindGustDir (encoded), WindGustSpeed, Humidity, Pressure, Current Temp
  - Output: RainTomorrow (Yes/No)
  - Enhanced: Displays `predict_proba()` as rain chance (e.g. 78%)

- **RandomForestRegressor**
  - Input: Previous temperature & humidity
  - Output: Next hour’s predicted value
  - Forecasts 5 future hours recursively

- Models trained on historical data stored as `weather.csv`

---

## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/weather-forecast-django.git
   cd weather-forecast-django
````

2. **Set up virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Place your weather dataset** at:

   ```
   static/data/weather.csv
   ```

4. **Add your OpenWeatherMap API key** in `views.py`:

   ```python
   API_KEY = 'your_api_key_here'
   ```

5. **Run the app**:

   ```bash
   python manage.py runserver
   ```

---

## 🌐 Live Demo (Optional)

> *If deployed, drop your link here (e.g. Render, PythonAnywhere, etc.)*

---

## 📌 TODOs & Future Work

* Add location auto-detection with Geolocation API
* Save & display city search history
* Support dark/light themes
* Add deployment (Docker/Vercel/Render)
* Improve UI responsiveness on mobile

---

## 🧠 What I Learned

* Django template engine, static/media routing
* Integration of real-time APIs with ML models
* Proper timezone handling with `pytz`
* Using Chart.js for live data visualization
* Structuring backend logic to serve predictions
* End-to-end project delivery

---

## 🙌 Acknowledgements

* [OpenWeatherMap API](https://openweathermap.org/api)
* `scikit-learn`, `pandas`, `Chart.js`

