from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.models import Variable
import logging
import snowflake.connector

USE_ROLE_SQL = "USE ROLE ACCOUNTADMIN;"
USE_WAREHOUSE_SQL = "USE WAREHOUSE COMPUTE_WH;"
USE_DATABASE_SQL = "USE DATABASE STOCK_PRICE;"
USE_SCHEMA_SQL = "USE SCHEMA RAW_DATA;"

TRAINING_DATA_SQL = """
SELECT * FROM STOCK_PRICES LIMIT 10;
"""

CREATE_VIEW_SQL = """
CREATE OR REPLACE VIEW STOCK_PRICES_v1 AS SELECT
    TO_TIMESTAMP_NTZ(DATE) AS DATE_v1,
    CLOSE,
    SYMBOL
FROM STOCK_PRICES;
"""

GENERATE_PREDICTIONS_SQL = """
CREATE OR REPLACE TABLE My_forecasts_2024_10_13 AS
SELECT * FROM TABLE(
    ML.FORECAST(
        INPUT_DATA => (SELECT * FROM STOCK_PRICES_v1),
        SERIES_COLNAME => 'SYMBOL',
        TIMESTAMP_COLNAME => 'DATE_v1',
        TARGET_COLNAME => 'CLOSE',
        FORECASTING_PERIODS => 7,
        CONFIDENCE_LEVEL => 0.95
    )
);
"""

VIEW_PREDICTIONS_SQL = """
SELECT * FROM My_forecasts_2024_10_13;
"""

UNION_PREDICTIONS_WITH_HISTORICAL_SQL = """
SELECT SYMBOL, DATE, CLOSE AS actual, NULL AS forecast, NULL AS lower_bound, NULL AS upper_bound
FROM STOCK_PRICES
UNION ALL
SELECT SYMBOL, DATE_v1 AS DATE, NULL AS actual, forecast, lower_bound, upper_bound
FROM My_forecasts_2024_10_13;
"""

INSPECT_ACCURACY_SQL = """
CALL lab_1_forecast!SHOW_EVALUATION_METRICS();
"""

FEATURE_IMPORTANCE_SQL = """
CALL lab_1_forecast!EXPLAIN_FEATURE_IMPORTANCE();
"""

def execute_snowflake_query(sql_query):
    try:
        conn = snowflake.connector.connect(
            user=Variable.get("snowflake_user"),
            password=Variable.get("snowflake_password"),
            account=Variable.get("snowflake_account"),
            warehouse=Variable.get("snowflake_warehouse"),
            database=Variable.get("snowflake_database"),
            schema=Variable.get("snowflake_schema")
        )
        
        cursor = conn.cursor()
        cursor.execute(sql_query)
        logging.info(f"Successfully executed query: {sql_query}")
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to execute query: {sql_query} with error: {e}")
        raise

with DAG(
    dag_id='stock_price_forecasting',
    default_args={'owner': 'airflow'},
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    description='A DAG for stock price forecasting using Snowflake ML',
) as dag:

    @task
    def setup_snowflake_environment():
        execute_snowflake_query(USE_ROLE_SQL)
        execute_snowflake_query(USE_WAREHOUSE_SQL)
        execute_snowflake_query(USE_DATABASE_SQL)
        execute_snowflake_query(USE_SCHEMA_SQL)

    @task
    def inspect_training_data():
        execute_snowflake_query(TRAINING_DATA_SQL)

    @task
    def create_stock_prices_view():
        execute_snowflake_query(CREATE_VIEW_SQL)

    @task
    def generate_predictions():
        execute_snowflake_query(GENERATE_PREDICTIONS_SQL)

    @task
    def view_predictions():
        execute_snowflake_query(VIEW_PREDICTIONS_SQL)

    @task
    def union_predictions_with_historical_data():
        execute_snowflake_query(UNION_PREDICTIONS_WITH_HISTORICAL_SQL)

    @task
    def inspect_model_accuracy():
        execute_snowflake_query(INSPECT_ACCURACY_SQL)

    @task
    def explain_feature_importance():
        execute_snowflake_query(FEATURE_IMPORTANCE_SQL)

    setup_task = setup_snowflake_environment()
    inspect_training_data_task = inspect_training_data()
    create_view_task = create_stock_prices_view()
    generate_predictions_task = generate_predictions()
    view_predictions_task = view_predictions()
    union_predictions_task = union_predictions_with_historical_data()
    inspect_accuracy_task = inspect_model_accuracy()
    explain_feature_importance_task = explain_feature_importance()

    setup_task >> inspect_training_data_task >> create_view_task >> generate_predictions_task
    generate_predictions_task >> view_predictions_task >> union_predictions_task
    generate_predictions_task >> inspect_accuracy_task >> explain_feature_importance_task
