from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from training_app.etl import store_raw_data, clear_raw_data, get_raw_data, clear_clean_data, save_clean_data, get_clean_data


with DAG (dag_id= "training_dag",
          description="Entrenando modelos",
          schedule_interval="@once",
          start_date=datetime (2023,5,1)) as dag:
    t1 = PythonOperator(task_id="clear_raw_data",
                      python_callable=clear_raw_data)
        
    t2 = PythonOperator(task_id="store_raw_data",
                      python_callable=store_raw_data)

    t3 = PythonOperator(task_id="get_raw_data",
                      python_callable=get_raw_data)

    t4 = PythonOperator(task_id="clear_clean_data",
                      python_callable=clear_clean_data)

    t5 = PythonOperator(task_id="save_clean_data",
                      python_callable=save_clean_data)

    t1 >> t2 >> t3 >> t4 >> t5