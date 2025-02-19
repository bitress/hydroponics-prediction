import mysql.connector
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, jsonify, request
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

def create_connection():
    try:
        connection = mysql.connector.connect(
            host="139.99.97.250",
            user="hydroponics",
            password=")[ZEy032Zy_oe8C8",
            database="hydroponics"
        )
        if connection.is_connected():
            print("Successfully connected to the database")
        return connection
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def fetch_sensor_data(connection, sensor_id, limit=300):
    query = f"SELECT reading_time, value FROM sensor_data WHERE sensor_id = {sensor_id} LIMIT {limit};"
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        if not result:
            raise ValueError("No data found for the specified sensor ID.")
        return result
    except mysql.connector.Error as e:
        print(f"Error executing query: {e}")
        return None
    except ValueError as e:
        print(e)
        return None
    finally:
        cursor.close()

def preprocess_data(data, lag_days=3):
    df = pd.DataFrame(data, columns=['reading_time', 'sensor_value'])
    df['reading_time'] = pd.to_datetime(df['reading_time'])
    df = df.sort_values('reading_time')
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df['sensor_value'].shift(i)
    df = df.dropna()
    if len(df) < lag_days:
        print("Not enough data after preprocessing")
        return pd.DataFrame()
    return df

def train_xgboost(df):
    features = ['lag_1', 'lag_2', 'lag_3']
    target = 'sensor_value'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror'), param_grid, cv=3,
                               scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'R-squared: {r2 * 100:.2f}%')
    print(f'Best hyperparameters: {best_params}')

    return best_model, scaler

def predict_next_day(model, scaler, df, hours_ahead=24):
    last_data = df[['lag_1', 'lag_2', 'lag_3']].iloc[-1]
    last_data_df = pd.DataFrame([last_data], columns=['lag_1', 'lag_2', 'lag_3'])
    last_data_scaled = scaler.transform(last_data_df)

    predictions = [model.predict(last_data_scaled)[0]]

    for _ in range(1, hours_ahead):
        next_data = [predictions[-1], last_data['lag_1'], last_data['lag_2']]
        next_data_df = pd.DataFrame([next_data], columns=['lag_1', 'lag_2', 'lag_3'])
        next_data_scaled = scaler.transform(next_data_df)

        predictions.append(model.predict(next_data_scaled)[0])
        last_data = {'lag_1': predictions[-1], 'lag_2': last_data['lag_1'], 'lag_3': last_data['lag_2']}

    future_time = pd.date_range(df['reading_time'].iloc[-1], periods=hours_ahead + 1, freq='H')[1:]

    return predictions, future_time

@app.route('/predict', methods=['GET'])
def get_prediction():
    target_value = request.args.get('target_value', default=7.5, type=int)

    connection = create_connection()

    if connection:
        sensor_data = fetch_sensor_data(connection, 1)

        if not sensor_data:
            return jsonify({'error': 'No sensor data fetched'}), 400

        df = preprocess_data(sensor_data)

        if df.empty:
            return jsonify({'error': 'Dataframe is empty after preprocessing'}), 400

        model, scaler = train_xgboost(df)

        predictions, future_time = predict_next_day(model, scaler, df, hours_ahead=72)  
        
        predictions = [float(p) for p in predictions]
        future_time = [str(time) for time in future_time]
        

        result = {
            'predicted_value_day_3': predictions[72-1], 
            'predicted_time_day_3': future_time[72-1],
            'target_value': target_value,
            'status': get_ph_status(predictions[72-1])
        }

        connection.close()
        return jsonify(result)

    return jsonify({'error': 'No connection to the database'}), 500

def get_ph_status(ph_value):
    if ph_value == 'N/A':
        return None
    
    ph_status_ranges = [
        (-float('inf'), 5.5, "Too Acidic"),
        (5.5, 6.0, "Acidic"),
        (6.0, 6.5, "Suboptimal"),
        (6.5, 7.5, "Optimal"),
        (7.5, 8.0, "Slightly Alkaline"),
        (8.0, float('inf'), "Too Alkaline")
    ]
    
    for min_val, max_val, status_text in ph_status_ranges:
        if min_val <= ph_value < max_val:
            return status_text
    
    return None 

if __name__ == "__main__":
    app.run(debug=True)
