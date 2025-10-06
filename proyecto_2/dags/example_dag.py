from airflow import DAG
from datetime import datetime, timedelta
from airflow.utils.state import State
from airflow import settings
import subprocess
from airflow.operators.python import PythonOperator, ShortCircuitOperator, BranchPythonOperator
from airflow.models import Variable, DagRun
from airflow.operators.empty import EmptyOperator


def pause_dag_if_failed(**context):
    dag_id = context['dag'].dag_id
    session = settings.Session()
    dag_runs = session.query(DagRun).filter(DagRun.dag_id == dag_id).order_by(DagRun.execution_date.desc()).all()
    subprocess.run(["airflow", "dags", "pause", dag_id])
    if dag_runs and dag_runs[0].state == State.FAILED:
     #    subprocess.run(["airflow", "dags", "pause", dag_id])
          dag_model.is_paused = True
          session.commit()
          session.close()
          print(f"DAG '{dag_id}' paused due to failure.")


def check_run_count(**context):
    # keep track of how many runs have happened
     key = "training_dag_run_count"
     count = int(Variable.get(key, default_var="0"))
     print(f"Current run count: {count}")
     if count >=3:
        raise ValueError("Reached maximum number of runs (10).")

    # increment counter
     Variable.set(key, str(count + 1))
     print(f"Run {count+1}/10")

def set_run_count(value):
    key = "training_dag_run_count"
    
    Variable.set(key, str(value))
    print(f"Run {value}/10")




# with DAG (dag_id="dummy_dag",
#         description="Entrenando modelos",
#         schedule_interval=timedelta(minutes=0, seconds=10),   # every 5 minutes and 20 seconds
#         start_date=datetime(2025, 10, 2, 0, 0, 0),   # change as needed
#         catchup=False,
#         max_active_runs=10,
#         is_paused_upon_creation=False

# ) as dag:

#      pause_dag_task = PythonOperator(
#           task_id="pause_dag_if_failed",
#           python_callable=pause_dag_if_failed,
#           provide_context=True,
#           trigger_rule="one_failed",  # Se ejecuta si alguna tarea falla
#      )

#      check = PythonOperator(
#         task_id="check_run_count",
#         python_callable=check_run_count,
#         provide_context=True
#      )
#      t1 = EmptyOperator(task_id="dummy")
#      t2 = EmptyOperator(task_id="dummy2")
    
    

# check >> t1 >> t2 >> pause_dag_task