import mysql.connector
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.model_selection import GridSearchCV

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

def fetch_sensor_data(connection, sensor_id, limit=1000):
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

def predict_when_full(model, scaler, df, target_value=30):
    last_data = df[['lag_1', 'lag_2', 'lag_3']].iloc[-1]
    last_data_df = pd.DataFrame([last_data], columns=['lag_1', 'lag_2', 'lag_3'])
    last_data_scaled = scaler.transform(last_data_df)

    predictions = [model.predict(last_data_scaled)[0]]
    time_step = 1

    while predictions[-1] < target_value:
        next_data = [predictions[-1], last_data['lag_1'], last_data['lag_2']]
        next_data_df = pd.DataFrame([next_data], columns=['lag_1', 'lag_2', 'lag_3'])
        next_data_scaled = scaler.transform(next_data_df)

        predictions.append(model.predict(next_data_scaled)[0])
        last_data = {'lag_1': predictions[-1], 'lag_2': last_data['lag_1'], 'lag_3': last_data['lag_2']}
        time_step += 1

    return predictions, time_step

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Water Level Prediction and Forecast", style={'textAlign': 'center'}),

    dcc.Graph(id='prediction-graph'),

    html.Div([
        html.Label("Target Value for Prediction:"),
        dcc.Input(id='target-input', type='number', value=30, min=1)
    ], style={'textAlign': 'center', 'marginTop': '20px'})
])

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

@app.callback(
    Output('prediction-graph', 'figure'),
    Input('target-input', 'value')
)
def update_graph(target_value):
    print(f"Target Value: {target_value}")

    connection = create_connection()

    if connection:
        sensor_data = fetch_sensor_data(connection, 5)

        if not sensor_data:
            print("No sensor data fetched")
            return go.Figure()

        df = preprocess_data(sensor_data)

        if df.empty:
            print("Dataframe is empty after preprocessing")
            return go.Figure()

        model, scaler = train_xgboost(df)

        predictions, future_time = predict_next_day(model, scaler, df, hours_ahead=24)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['reading_time'], y=df['sensor_value'], mode='lines', name='Actual Sensor Values',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=future_time, y=predictions, mode='lines+markers', name='Predicted Sensor Values for Next Day',
            line=dict(color='red', dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=[df['reading_time'].iloc[0], future_time[-1]], y=[target_value, target_value], mode='lines',
            name=f'Target Sensor Value ({target_value})', line=dict(color='green', dash='dash')
        ))

        fig.update_layout(
            title="Water Level Prediction and Forecast (Next Day)",
            xaxis_title="Time",
            yaxis_title="Sensor Value",
            template='plotly_dark',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )

        connection.close()
        return fig

    print("No connection to the database")
    return go.Figure()

if __name__ == "__main__":
    app.run_server(debug=True)
