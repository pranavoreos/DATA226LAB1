import requests
import pandas as pd
import snowflake.connector
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.base_hook import BaseHook
from datetime import datetime, timedelta

# Constants
ALPHA_VANTAGE_API_KEY = 'your_alpha_vantage_api_key'
SNOWFLAKE_CONN_ID = 'your_snowflake_connection_id'
STOCK_SYMBOLS = ['AAPL', 'NVDA']

def fetch_stock_data(stock_symbol):
    """
    Fetch stock data from Alpha Vantage API for the specified symbol.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        time_series = data.get("Time Series (Daily)", {})
        
        stock_data = []
        for date, values in time_series.items():
            stock_data.append({
                'stock_symbol': stock_symbol,
                'date': date,
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            })
        return pd.DataFrame(stock_data)
    else:
        raise Exception(f"Failed to fetch data for {stock_symbol}: {response.content}")

def load_to_snowflake(df):
    """
    Load stock data DataFrame to Snowflake.
    """
    
    snowflake_conn = BaseHook.get_connection(SNOWFLAKE_CONN_ID)

    conn = snowflake.connector.connect(
        user=snowflake_conn.login,
        password=snowflake_conn.password,
        account=snowflake_conn.host,
        warehouse='COMPUTE_WH',
        database='stock_data_db',
        schema='finance_schema'
    )

    cursor = conn.cursor()

    try:
        
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO stock_prices (stock_symbol, date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (row['stock_symbol'], row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))
        conn.commit()
        print(f"Data loaded into Snowflake for {df['stock_symbol'].iloc[0]}")
    except Exception as e:
        print(f"Error occurred while loading data to Snowflake: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def ingest_stock_data():
    """
    Ingest stock data for all specified symbols and load into Snowflake.
    """
    for symbol in STOCK_SYMBOLS:
        try:
            stock_df = fetch_stock_data(symbol)
            load_to_snowflake(stock_df)
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('ingest_stock_data_dag', default_args=default_args, schedule_interval='@daily')


ingest_task = PythonOperator(
    task_id='ingest_stock_data',
    python_callable=ingest_stock_data,
    dag=dag
)

ingest_task
