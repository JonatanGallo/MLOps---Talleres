from airflow import DAG
from datetime import datetime
from training_app.etl import store_raw_data, clear_raw_data, get_raw_data, clear_clean_data, save_clean_data, get_clean_data
from training_app.train import trainModel
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.models import Variable

def check_first_run():
    if Variable.get("dag_first_run_done", default_var="false") == "false":
        return True
    return False

def mark_first_run_done():
    print("Marking first run as done")
    Variable.set("dag_first_run_done", "true")

with DAG (dag_id= "training_dag",
          description="Entrenando modelos",
          schedule_interval="@once",
          start_date=datetime (2023,5,1)) as dag:

    is_first_run = ShortCircuitOperator(
        task_id="is_first_run",
        python_callable=check_first_run
    )

    first_time_task = PythonOperator(
        task_id="first_time_task",
        python_callable=lambda: print("This runs only on first execution")
    )

    mark_done = PythonOperator(
        task_id="mark_done",
        python_callable=mark_first_run_done
    )

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

    t6 = PythonOperator(task_id="train_model",
                      python_callable=trainModel)

    [is_first_run >> first_time_task >> [t1, t4] >> mark_done] >> t2 >> t3 >> t5 >> t6