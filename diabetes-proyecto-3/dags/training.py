from airflow import DAG, settings
import subprocess
from datetime import datetime, timedelta
from airflow.utils.state import State
from training_app.etl import store_raw_data, clean_all_data, get_raw_data, save_clean_data, get_clean_data
from training_app.train import train_and_publish_best
from training_app.dataController import store_raw_data, save_clean_data
from airflow.operators.python import PythonOperator, ShortCircuitOperator, BranchPythonOperator
from airflow.models import Variable, DagRun
from airflow.operators.empty import EmptyOperator


def pause_dag_if_failed(**context):
    dag_id = context['dag'].dag_id
    session = settings.Session()
    dag_runs = session.query(DagRun).filter(DagRun.dag_id == dag_id).order_by(DagRun.execution_date.desc()).all()
    subprocess.run(["airflow", "dags", "pause", dag_id])

def check_run_count(**context):
    # keep track of how many runs have happened
     key = "training_dag_run_count"
     count = int(Variable.get(key, default_var="0"))
     print(f"Current run count: {count}")
     if count >=11:
        raise ValueError("Reached maximum number of runs (11).")
    # increment counter
     Variable.set(key, str(count + 1))
     print(f"Run {count+1}/11")

def check_first_run():
    if Variable.get("dag_first_run_done", default_var="false") == "false":
        return True
    return False

def check_run_count(**context):
    # keep track of how many runs have happened
    key = "training_dag_run_count"
    count = int(Variable.get(key, default_var="0"))

    if count >=11:
        raise ValueError("Reached maximum number of runs (11).")

    # increment counter
    Variable.set(key, str(count + 1))
    print(f"Run {count+1}/11")


def set_run_count(value):
    key = "training_dag_run_count"
    
    Variable.set(key, str(value))
    print(f"Run {value}/11")

def choose_branch():
    first_run = Variable.get("dag_first_run_done", default_var="false") == "false"
    return "clean_all_data" if first_run else "skip_first_time"

def mark_first_run_done():
    print("Marking first run as done")
    Variable.set("dag_first_run_done", "true")

with DAG (dag_id="training_dag",
        description="Entrenando modelos",
        schedule_interval=timedelta(minutes=5, seconds=20),   # every 5 minutes and 20 seconds
        start_date=datetime(2025, 10, 3, 0, 0, 0),   # change as needed
        catchup=False,
        max_active_runs=11,
        is_paused_upon_creation=False
) as dag:

    mark_done = PythonOperator(
        task_id="mark_done",
        python_callable=mark_first_run_done
    )

    pause_dag_task = PythonOperator(
        task_id="pause_dag_if_failed",
        python_callable=pause_dag_if_failed,
        provide_context=True,
        trigger_rule="one_failed",  # Se ejecuta si alguna tarea falla
    )
     
    branch = BranchPythonOperator(
        task_id="branch_first_run",
        python_callable=choose_branch,
    )

    skip_first_time = EmptyOperator(
        task_id="skip_first_time",
    )
    check = PythonOperator(
        task_id="check_run_count",
        python_callable=check_run_count,
        provide_context=True
    )

    clean_all_data = PythonOperator(task_id="clean_all_data",
                      python_callable=clean_all_data)
        
    t2 = PythonOperator(task_id="store_raw_data",
                      python_callable=store_raw_data,
                      )

    t3 = PythonOperator(task_id="get_raw_data",
                      python_callable=get_raw_data)

    t5 = PythonOperator(task_id="save_clean_data",
                      python_callable=save_clean_data)

    t6 = PythonOperator(task_id="train_model",
                      python_callable=train_and_publish_best)

    join_after_branch = EmptyOperator(
        task_id="join_after_branch",
        trigger_rule="none_failed_min_one_success",
    )

    #check >> branch >> [clean_all_data, skip_first_time]>> join_after_branch >> mark_done >> t2 >> t3 >> t5 >> t6
    check >> branch >> [clean_all_data, skip_first_time]>> join_after_branch >> mark_done >> t2 >> t3 >> t5 >> t6 >> pause_dag_task
    