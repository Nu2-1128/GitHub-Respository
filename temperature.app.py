import streamlit as st
import openmeteo_requests
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests_cache
import warnings
import streamlit.components.v1 as components
import time
import math
import altair as alt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from retry_requests import retry
from geopy.geocoders import Nominatim
#from geopy.geolocators import Nominatim
from geopy.exc import GeocoderUnavailable
from datetime import date, timedelta

# Suppress specific warnings
warnings.filterwarnings("ignore", module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning)

# Function to call for map lcation, with lang_selection for display language
def create_map (lantitute, longtitude, address, lang_selection):
  if lang_selection == 'Location base':
    my_map = folium.Map(location=[lantitute, longtitude], zoom_start=10)
  else:
    my_map = folium.Map(location=[lantitute, longtitude], zoom_start=10, tiles='CartoDB positron')
  folium.Marker([lantitute, longtitude], popup=address).add_to(my_map)
  return my_map

# Function to call API to retrieve temperature history
def get_historical_data(latitude, longitude, start_date, end_date, temp_unit):

    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "temperature_unit": temp_unit
    }

    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_max"] = daily_data["temperature_2m_max"].round(2)

    daily_dataframe = pd.DataFrame(data = daily_data)
    return daily_dataframe

# function to calculate min and max display range
def calculate_y_range(min_temp, max_temp, temp_unit):
    min_y = 0
    max_y = 0
    if temp_unit == 'celsius':
      min_y = min_temp - 10
      max_y = max_temp + 10
      # Round down min_y to the nearest 5
      min_y = math.floor(min_y / 5) * 5
      # Round up max_y to the nearest 5
      max_y = math.ceil(max_y / 5) * 5
    else:
      min_y = min_temp - 20
      max_y = max_temp + 20
      # Round down min_y to the nearest 5
      min_y = math.floor(min_y / 5) * 5
      # Round up max_y to the nearest 5
      max_y = math.ceil(max_y / 5) * 5
    return min_y, max_y


def detect_outliers_iqr(data):
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = data[(data < lower_bound) | (data > upper_bound)]
  return outliers

# initialize
geolocator = Nominatim(user_agent="my_app")

# Set layout style
st.set_page_config(layout="wide", page_title="ForecastLens", page_icon="🌦️")

# initialize status for process
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'forecast' not in st.session_state:
    st.session_state.forcast = False
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame(columns=['date','temperature_2m_max'])
if 'forecast_clean' not in st.session_state:
    st.session_state.forecast_clean = pd.DataFrame(columns=['date','temperature_2m_max'])


forecast_button = False
column_names = ['date','temperature_2m_max']
#forecast_df = pd.DataFrame(columns = column_names)
hw = pd.DataFrame(columns=column_names)
ari = pd.DataFrame(columns=column_names)
temp_unit = 'fahrenheit'
Search_Location = 'Los Angeles, US'

# application title
st.title("ForecastLens")
st.markdown("For any question or suggestions, email us: <a href='mailto:support@yourcompany.com'>support@yourcompany.com</a>", unsafe_allow_html=True)

#Create tabs
tab_titles = ['Location Search','Historical Data','Data Cleansing','EDA','Holt-Winters','ARIMA','Forecast','Help']

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_titles)

with tab1:
  st.header("Location Search")
  # Prompt for user entry
  # user entry for location
  with st.form("user_input_form"):
    user_input = st.text_input("Enter a location:",
                            placeholder="Example: Los Angeles, US / Glendale, AZ / 3300 W Camelback Rd, Phoenix, AZ 85017")
    col1, col2 = st.columns(2)

  # radio button for temperature unit
    with col1:
      temp_selection = st.radio(
          "Select temperature Unit:",
          ('Fahrenheit','Celsius'),
          index=0
      )
  # radio button for display language
    with col2:
      lang_selection = st.radio(
          "Display Language:",
          ('English','Location base'),
          index=0
      )

    submitted = st.form_submit_button("Submit")

  # Check for entry
  if submitted:
    if user_input:
      st.session_state.submitted = True
      st.session_state.user_input = user_input
      st.session_state.temp_selection = temp_selection
      st.session_state.lang_selection = lang_selection

      #st.experimental_rerun()
    else:
      st.warning("Please enter a valid location.")

  # Display entry
  if st.session_state.submitted:

    st.markdown("---")

    Search_Location = st.session_state.user_input
    temp_unit = st.session_state.temp_selection.lower()
    lang = st.session_state.lang_selection

    #location = None
    retry_attempts = 5
    for attempt in range(retry_attempts):
        try:
            location = geolocator.geocode(Search_Location)
            if location:
                st.write(f"Location found on attempt {attempt + 1}:")
                st.write((location.latitude, location.longitude))
                st.write(location.raw['display_name'])
                forecast_button = not st.session_state.submitted
                break  # Exit the loop if location is found
        except GeocoderUnavailable as e:
            st.write(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retry_attempts - 1:
                st.write("Retrying in 2 seconds...")
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                st.write(f"Failed to retrieve location after {retry_attempts} attempts.")

    if not location:
        st.write("Could not identify the location. Please check the location name or try again later.")

    if location:
      loc_map = create_map(location.latitude, location.longitude, Search_Location, lang)

    # Save the map to an HTML file
      map_html_file = "location.html"
      loc_map.save(map_html_file)

    # Display the map in Streamlit
      with open(map_html_file, "r") as f:
        map_html = f.read()

      st.components.v1.html(map_html, width=700, height=500)

      ten_days_ago = date.today() - timedelta(days=10)
      ten_days_ago_str = ten_days_ago.strftime('%Y-%m-%d')

      yesterday = date.today() - timedelta(days=1)
      yesterday_str = yesterday.strftime('%Y-%m-%d')

      df = get_historical_data(location.latitude, location.longitude, ten_days_ago_str, yesterday_str, temp_unit)

    # Format the dataframe
      df.columns = ['Date', f'Temp {temp_unit}']
      df.dropna(inplace=True)
      df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
      df[f'Temp {temp_unit}'] = df[f'Temp {temp_unit}'].round(2)
      df.set_index('Date', inplace=True)

    # Display the last 5 non-NA records horizontally
      st.write("Last 5 days temperature:")
      st.dataframe(df.dropna().tail(5).T)

    #  if loc_map is not None:
    #    st.write(get_historical_data(location.latitude, location.longitude, ten_days_ago_str, yesterday_str, temp_unit))
    #  else:
    #    st.write("Could not identify the location. Please check the location name or try again later.")

    if st.button('Forecast', disabled=forecast_button, key='forecast_btn'):
      st.warning("The complete process may take up to 5 minutes to complete.")
      with st.spinner('Loading...'):
        time.sleep(2)
        st.session_state.forecast = True
        forecast_start_date = '2023-01-01'
        end_date = date.today().strftime('%Y-%m-%d')
        st.session_state.forecast_df = get_historical_data(location.latitude, location.longitude, forecast_start_date, end_date, temp_unit)

        # Clean NA value
        st.session_state.forecast_df = st.session_state.forecast_df.dropna()

        if st.session_state.forecast_df.empty:
          st.warning("Please go to the 'Location Search' tab and click the 'Forecast' button to load the data.")
        else:
        # Convert date column to datetime objects unconditionally
          st.session_state.forecast_df['date'] = pd.to_datetime(st.session_state.forecast_df['date'])
          NA_value = 0

        outliers = detect_outliers_iqr(st.session_state.forecast_df['temperature_2m_max'])
        #if not outliers.empty:
        #st.write("Outliers detected")

        # Treat outliers, create NANs at outlier position for interpolate()
        st.session_state.forecast_clean = st.session_state.forecast_df.copy()
        outlier_indices = outliers.index
        st.session_state.forecast_clean.loc[outlier_indices, 'temperature_2m_max'] = np.nan

        # Forecast Clean with interpolate info
        st.session_state.forecast_clean['temperature_2m_max'] = st.session_state.forecast_clean['temperature_2m_max'].interpolate(method='linear')

        time_series = st.session_state.forecast_clean[['date','temperature_2m_max']].rename(columns={'temperature_2m_max':'data'}).set_index('date')

        hw = time_series.copy()
        ari = time_series.copy()

        #Holt Winters Calcluation
        param_grid = {
          'seasonal': ['add', 'mul'],
          'seasonal_periods': [7, 14, 30, 365], # Example seasonal periods (weekly, bi-weekly, monthly, yearly)
          'smoothing_level': [0.1, 0.3, 0.5, 0.7, 0.9],
          'smoothing_trend': [None, 0.1, 0.3, 0.5, 0.7, 0.9],
          'smoothing_seasonal': [0.1, 0.3, 0.5, 0.7, 0.9],
          'trend': [None, 'add', 'mul']
        }

        # Remove invalid combinations (e.g., multiplicative seasonal with zero or negative data)
        # Also remove combinations where trend smoothing is specified but trend is None
        valid_params = []
        for params in ParameterGrid(param_grid):
            # Skip multiplicative seasonal if data contains non-positive values (check after fetching data)
            # In this case, temperature data is always positive, so this check is not strictly needed
            # if params['seasonal'] == 'mul' and (hw['data'] <= 0).any():
            #    continue

            # Skip trend smoothing if trend is None
            if params['trend'] is None and params['smoothing_trend'] is not None:
                continue

            valid_params.append(params)

        best_rmse = float('inf')
        best_params = None
        best_model = None

        # Split data for training and validation (simple split: last seasonal_period days for validation)
        # We need enough data to fit the seasonal component
        # Choose a validation set size that is at least as long as the maximum seasonal period
        validation_set_size = max(param['seasonal_periods'] for param in valid_params if param['seasonal_periods'] is not None)
        if len(hw) < validation_set_size + 1:
            st.write("Warning: Not enough data to split for validation with the given seasonal periods.")
            st.write(f"Need at least {validation_set_size + 1} data points, but have {len(hw)}.")
            # Proceed with a smaller validation set or just use all data for training
            # For this example, let's use a fixed smaller validation set if needed
            validation_set_size = min(validation_set_size, int(len(hw) * 0.2)) # Use 20% for validation if not enough
            if validation_set_size == 0 and len(hw) > 0:
                validation_set_size = 1 # Ensure at least one point for validation if data exists
            elif validation_set_size == 0 and len(hw) == 0:
                st.write("Error: No data available for fitting.")


        if validation_set_size > 0 and len(hw) > validation_set_size:
            train_data = hw['data'][:-validation_set_size]
            val_data = hw['data'][-validation_set_size:]

            # Grid search for best parameters
            for params in valid_params:
                try:
                    # Ensure valid seasonal_periods if seasonal is None
                    if params['seasonal'] is None:
                        params_copy = params.copy()
                        params_copy['seasonal_periods'] = None # Seasonal periods is irrelevant if no seasonality
                    else:
                        params_copy = params.copy()

                    model = ExponentialSmoothing(train_data,
                                                seasonal=params_copy['seasonal'],
                                                seasonal_periods=params_copy['seasonal_periods'],
                                                trend=params_copy['trend'])

                    fit = model.fit(smoothing_level=params_copy['smoothing_level'],
                                      smoothing_trend=params_copy['smoothing_trend'],
                                      smoothing_seasonal=params_copy['smoothing_seasonal'],
                                      optimized=False) # Set optimized=False for manual grid search

                    # Forecast on the validation set
                    val_forecast = fit.forecast(validation_set_size)

                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(val_data, val_forecast))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = params_copy
                        best_model = fit # Store the fit from the best parameters

                except Exception as e:
                    #print(f"Could not fit model with parameters {params}: {e}")
                    continue # Skip invalid parameter combinations that cause errors

            if best_params:
                #print(f"Best parameters found: {best_params}")
                #print(f"Best validation RMSE: {best_rmse:.4f}")

                # Retrain the model on the full dataset with the best parameters
                final_model = ExponentialSmoothing(hw['data'],
                                                  seasonal=best_params['seasonal'],
                                                  seasonal_periods=best_params['seasonal_periods'],
                                                  trend=best_params['trend'])

                final_fit = final_model.fit(smoothing_level=best_params['smoothing_level'],
                                            smoothing_trend=best_params['smoothing_trend'],
                                            smoothing_seasonal=best_params['smoothing_seasonal'],
                                            optimized=False) # Use optimized=False if manually setting params

                # Make the forecast for the next 6 days
                forecast_horizon = 6
                # Use forecast method for Holt-WintersResults object
                final_forecast = final_fit.forecast(forecast_horizon)
                # Note: Holt-WintersResults does not have a built-in conf_int method like SARIMAXResults

                #print("\nForecast for the next {} days using best Holt-Winters model:".format(forecast_horizon))
                #print(final_forecast)
                # print("\n95% Confidence Intervals:")
                # print(forecast_conf_int_hw)


                # Plot the original data and the forecast
                #plt.figure(figsize=(14, 7))
                #plt.plot(hw.index, hw['data'], label='Observed')

                # Create dates for the forecast period
                last_date = hw.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

                #plt.plot(forecast_dates, final_forecast, label='Holt-Winters Forecast (Optimized)', linestyle='--')
                # Plot confidence intervals (skipped for Holt-Winters)
                # plt.fill_between(forecast_dates,
                #                  forecast_conf_int_hw.iloc[:, 0],
                #                  forecast_conf_int_hw.iloc[:, 1], color='k', alpha=.15, label='95% Confidence Interval')

                #plt.xlabel('Date')
                #if temp_unit == 'celsius':
                #    plt.ylabel('Maximum Temperature (°C)')
                #else:
                #    plt.ylabel('Maximum Temperature (°F)')
                #plt.ylim(min_y,max_y)
                #plt.title('Daily Maximum Temperature Forecast using Optimized Holt-Winters') # Adjusted title
                #plt.legend()
                #plt.grid(True)
                #plt.show()

            else:
                st.write("No valid model found with the given parameter grid.")
        else:
            st.write("Not enough data points in the dataframe to perform a train-validation split.")
            # Optionally, fit a model on all data if validation is not possible/desired
            if len(hw) > 0:
                st.write("Fitting a default Holt-Winters model on all available data without validation...")

                # Define a simple default model or iterate through a minimal set of parameters
                # For demonstration, let's pick some default parameters if no validation is possible
                default_params = {
                    'seasonal': 'add' if len(hw) >= 7 else None, # Use seasonal if enough data for a week
                    'seasonal_periods': 7 if len(hw) >= 7 else None,
                    'trend': 'add',
                    'smoothing_level': 0.3,
                    'smoothing_trend': 0.1 if 'add' in ['add','mul'] else None, # Only set if trend is add or mul
                    'smoothing_seasonal': 0.1 if 'add' in ['add','mul'] and len(hw) >= 7 else None # Only set if seasonal is add or mul and enough data
                }
                # Clean up default_params based on data availability
                if default_params['seasonal_periods'] is not None and len(hw) < default_params['seasonal_periods']:
                    default_params['seasonal'] = None
                    default_params['seasonal_periods'] = None
                    default_params['smoothing_seasonal'] = None


                try:
                    model = ExponentialSmoothing(hw['data'],
                                                  seasonal=default_params['seasonal'],
                                                  seasonal_periods=default_params['seasonal_periods'],
                                                  trend=default_params['trend'])

                    fit = model.fit(optimized=True) # Allow optimization for default model

                    # Make the forecast for the next 6 days
                    forecast_horizon = 6
                    # Use forecast method for Holt-WintersResults object
                    forecast = fit.forecast(forecast_horizon)
                    # Confidence intervals are not directly available for forecast method
                    # forecast_conf_int = fit.conf_int(alpha=0.05)


                    st.write("\nForecast for the next {} days using a default Holt-Winters model:".format(forecast_horizon))
                    st.write(forecast)
                    # print("\n95% Confidence Intervals:")
                    # print(forecast_conf_int)


                    # Plot the original data and the forecast
                    #plt.figure(figsize=(14, 7))
                    #plt.plot(hw.index, hw['data'], label='Observed')

                    # Create dates for the forecast period
                    last_date = hw.index[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

                    #plt.plot(forecast_dates, forecast, label='Default Holt-Winters Forecast', linestyle='--')
                    # Plot confidence intervals (skipped for Holt-Winters)
                    # plt.fill_between(forecast_dates,
                    #                  forecast_conf_int.iloc[:, 0],
                    #                  forecast_conf_int.iloc[:, 1], color='k', alpha=.15, label='95% Confidence Interval')


                    #plt.xlabel('Date')
                    #if temp_unit == 'celsius':
                    #    plt.ylabel('Maximum Temperature (°C)')
                    #else:
                    #    plt.ylabel('Maximum Temperature (°F)')
                    #plt.ylim(min_y,max_y)
                    #plt.title('Daily Maximum Temperature Forecast using Default Holt-Winters') # Adjusted title
                    #plt.legend()
                    #plt.grid(True)
                    #plt.show()

                except Exception as e:
                      st.write(f"Could not fit a default Holt-Winters model: {e}")

        st.write('Please view result in "Historical Data" tab')

  with tab2:
    st.header("Historical Data")

    #forecast_df = forecast_df.dropna()

    Min_temp = 0
    Max_temp = 0
    min_y = 0
    max_y = 0

    if not st.session_state.forecast_df.empty:
      Min_temp = st.session_state.forecast_df['temperature_2m_max'].min()
      Max_temp = st.session_state.forecast_df['temperature_2m_max'].max()
    else:
      st.write('No data available')

    #min_y, max_y = calculate_y_range(Min_temp, Max_temp, temp_unit)

    st.write('Historical temperature for ',Search_Location,', unit in ',temp_unit)

    st.line_chart(st.session_state.forecast_df, x='date', y='temperature_2m_max')
    col1, col2 = st.columns(2)
    with col1:
      st.write(f"Min temp: {Min_temp:.2f}")
    with col2:
      st.write(f"Max temp: {Max_temp:.2f}")

    st.markdown("---")
    st.dataframe(st.session_state.forecast_df)



  with tab3:
    st.header("Data Cleansing")


    if st.session_state.forecast_df.empty:
        st.warning("Please go to the 'Location Search' tab and click the 'Forecast' button to load the data.")
    else:

        NAN_container = st.container(border=True)
        Missing_container = st.container(border=True)
        Outlier_container = st.container(border=True)

        with NAN_container:
          st.write("Search for NA values...")
          NA_value = st.session_state.forecast_df.isnull().sum()
          st.write('Number of NAN value:',NA_value)

        st.markdown("---")

        with Missing_container:
          st.write("Search for missing date records...")
          st.session_state.forecast_df['data_diff'] = st.session_state.forecast_df['date'].diff().dt.days
          missing_days = st.session_state.forecast_df[st.session_state.forecast_df['data_diff'] > 1]

          if not missing_days.empty:
            st.write("Missing days:")
            st.write(missing_days)
          else:
            st.write("No missing days found.")

        st.markdown("---")

        with Outlier_container:
          st.write("Search for Outliers...")
          outliers = detect_outliers_iqr(st.session_state.forecast_df['temperature_2m_max'])
          if not outliers.empty:
            st.write("Outliers:")
            st.write(outliers)

            st.write('Outliers detected')

            fig=px.line(st.session_state.forecast_clean, x='date', y='temperature_2m_max', title=f'Cleaned temperature over time in {Search_Location}')
            st.plotly_chart(fig)

          else:
            st.write("No outliers found.")


        st.markdown("---")

  with tab4:
    st.header("EDA")

    Monthly_MinMax_container = st.container(border=True)
    Monthly_Histo_container = st.container(border=True)
    Annual_container = st.container(border=True)
    Annual_Rolling_Container = st.container(border=True)

    if(st.session_state.forecast_clean.empty):
      st.warning("Please go to the 'Location Search' tab and click the 'Forecast' button to load the data.")
    else:
      with Monthly_MinMax_container:
        st.write("Monthly Min/Max/Mean Temperature")
        # Makes a copy of the forecast_clean df
        monthly_stats_df = st.session_state.forecast_clean.copy()

        # Extract month and year
        monthly_stats_df['month_year'] = monthly_stats_df['date'].dt.to_period('M')

        # Group by month and year and calculate min, max, and mean
        monthly_stats = monthly_stats_df.groupby('month_year')['temperature_2m_max'].agg(['min', 'max', 'mean']).reset_index()

        # Rename columns for better readability
        monthly_stats.columns = ['Month-Year', 'Min Temp', 'Max Temp', 'Mean Temp']

        # Display the table
        st.dataframe(monthly_stats)

      with Monthly_Histo_container:
        st.write("Monthly Temperature Histogram")
        # Create a copy for plotting
        monthly_hist_df = st.session_state.forecast_clean.copy()

        # Extract month name for plotting
        monthly_hist_df['month_name'] = monthly_hist_df['date'].dt.strftime('%B')

        # Define the order of months for consistent plotting
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        monthly_hist_df['month_name'] = pd.Categorical(monthly_hist_df['month_name'], categories=month_order, ordered=True)

        # Create the histogram using Plotly Express
        fig_hist = px.histogram(monthly_hist_df,
                                x='temperature_2m_max',
                                color='month_name',
                                facet_col='month_name',
                                facet_col_wrap=4,  # Arrange plots in 4 columns
                                title=f'Distribution of Daily Maximum Temperature by Month in {Search_Location}',
                                nbins=30)  # Adjust the number of bins as needed

        # Update layout for better readability
        fig_hist.update_layout(
            xaxis_title=f'Maximum Temperature ({temp_unit})',
            yaxis_title='Frequency',
            showlegend=False,
            bargap=0.1 # Add space between bars
        )

        # Update subplot titles
        fig_hist.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # Display the plot in Streamlit
        st.plotly_chart(fig_hist, use_container_width=True)

      with Annual_container:
        st.write("Annual Temperature")
      with Annual_Rolling_Container:
        st.write("Annual Rolling Temperature")

  with tab5:
    st.header("Holt-Winters")
    st.write(hw)
    #st.write(forecast)

  with tab6:
    st.header("ARIMA")
    st.write(ari)

  with tab7:
    st.header("Forecast")

  with tab8:
    st.header("Help")
    st.write("""
    Welcome to the help section. You can find the instruction on how to utilize this application effectively.
    """)
    st.markdown("""
    ForecastLens is an application designed for genearl users to identify a location around the world, to learn the current and historical temperature, and the forecast of the next 6-days period.
    For users that interested in the complete progress, it also included step by step guide on data collection, data cleaing, EDA (Exploratory Data Analysis), the two algorithms used - Holt-Winters and ARIMA, and final forecast result.
    Below is what you can expect to find in each tab:
      * Location Search: This is where everything starts, by entering a location, the temperature unit, and display language.
        * The location can be accepted in different format (as shown in the example), but it must be in English.
        * Temperature unit in Fahreheit or Celsius.
        * Display language in English or location base. For example, the map will display Japanese if you search for Tokyo, or Chinese if you seach for Beijing.
        * User can click the :blue[**"Submit"**] button or click :blue[**"Enter"**] for searching.
        * Upon a location is find, a map of the location is shown, with the corresponding temperature reading of the location will be displayed.
          * This is to ensure the correct location is identified, as there can be multiple locations with the same name (e.g. Glendale, California and Glendale, Arizona).
        * With the click on the :blue[**"Forecast Button"**], the complete process will start, a warning message of the loading time is shown. When ready, please navigate to Historical Data tab.
      * Historical Data
        * This is the location of collected historical data.
        * The data is a retrieved using the Open-Metro API, which contains weather related data from 1940 to current date.
          * Genearl data are collected through combined observations from weather stations, aircraft, buoys, radar, and satellites to prepare a comprehensive record of historical weather conditions.
          * Any gaps in the datasets are filled using mathematical models to estimate the values of various weather variables.
          * Together, it provided a reanalyzed datasets that offer detailed historical weather informaiton for locations that may not have had any weather stations, like rural areas or open ocean.
        * Data is displayed in the form of a time-series line chart, along with the temperature with selected unit.
        * A table format of the data is also displayed, for detail review.
      * Data Cleaning
        * This area contains the data cleaning process, step by step.
        * Collected data is treated for the following:
          * Missing data 1 - toward new date, due to both data availability and time zone differences, the latest record may not be available during retrieval (as noted by Open-Metro, up to 5-days delay).
          * Missing data 2 - toward in-between data. If any records are missing in between days.
            * Since this is focus on time-series forecasting, a missing record in between days can result in error in calculation.
            * Any missing records, when find, will be handled with linear interpolation method.
          * Outliers - Any extreme records, either too high or too low, based on IQR (interquartile range) method, will be treated as outliers.
            * Outliers record is treated removal, and filled with linear interpolation method.
        * Both line-chart and table of the cleaned data is displayed.
      * EDA
        * This area contains the Exploratory Data Analysis process.
        * It is an iterative process of analyzing data sets, primarily using visual and statiscal methods, to summarize their main characteristics, uncover patterns, detect anomalies, and formulate hypotheses.
        * A few visualization is included in this area
          * Monthly Min/Max/Mean Temperature
          * Monthly Temperature Histogram
          * Annual Temperature in daily presentation
          * Annual 30-days Rolling Temperature
          * ADF (Augmented Dickey-Fuller) Test and Ljung Box test.
            * They are test for data stationarity and seasonality.
          * Seasonal decomposition chart
            * To display data level, trend, seasonality, and residual.
      * Holt-Winters
        * This area contains the Holt-Winters algorithm.
        * It contains a line chart time-series witih original data and forecast result.
        * Forecast result is also dispalyed in a table format.
        * For educational purpose, the optimized parameters used are also displayed.
      * ARIMA
        * This area contains the ARIMA algorithm.
        * It contains a line chart time-series witih original data and forecast result.
        * Forecast result is also dispalyed in a table format.
        * For educational purpose, the optimized parameters used are also displayed.
      * Forecast
        * This is to display the forecast result from both Holt-Winters and ARIMA algorithms.
        * A line chart time-series with last 14 days of original data and forecast result is displayed.
          * An average calculation between both forecasts result are included, for an optimal forecasted result for users.
        * A download feature is included for users to download the original data, forecast parameters and forecast result.
    """)

    st.markdown("""
    Source for the API use:
    * GeoPy: [GeoPy](https://geopy.readthedocs.io/en/stable/)
    * Open-Metro: [Open-Metro](https://open-meteo.com/)
    """)

    st.markdown("---")
