import mysql.connector
import time
import sys
import os
import pandas as pd

def get_db_connection():
    host = os.getenv('DB_HOST', 'mysql_db')  # service name in docker-compose
    port = int(os.getenv('DB_PORT', '3306'))
    user = os.getenv('DB_USER', 'user')
    password = os.getenv('DB_PASSWORD', 'password')
    database = os.getenv('DB_NAME', 'training')

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database
    )

def create_table(table_name, columns_df):
  connection = None
  cursor = None
  try:
    # columns_df = columns_df.to_string(index=False)
    print(columns_df.head())
    columns = columns_df.columns.tolist()
    columns_str = ", ".join([f"{col} VARCHAR(255)" for col in columns])
    print("columns_str", columns_str)
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {columns_str}
        )
    """)
    connection.commit()
  except mysql.connector.Error as err:
    print(f"❌ Database error: {err}")
  finally:
    if connection.is_connected():
      cursor.close()
      connection.close()
      print("🔒 MySQL connection closed")

def get_headers(columns_df):
  columns = columns_df.columns.tolist()
  return columns

def insert_data(table_name, data):
  connection = None
  cursor = None
  try:
      connection = get_db_connection()
      cursor = connection.cursor()
      columns = get_headers(data)

      df = data[columns].copy().astype(object)
      df = df.where(pd.notna(df), None)
      rows_clean = df.values.tolist()

      placeholders = ", ".join(["%s"] * len(columns))
      columns_str = ", ".join(columns)

      print("rows_clean", rows_clean)
      print("columns_str", columns_str)
      print("placeholders", placeholders)
      sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
      cursor.executemany(sql, rows_clean)  # insert many rows at once
      connection.commit()
      print(f"✅ Inserted {cursor.rowcount} rows into {table_name}")
  except mysql.connector.Error as err:
      print(f"❌ Database error: {err}")
  finally:
      if connection and connection.is_connected():
          cursor.close()
          connection.close()
          print("🔒 MySQL connection closed")

def get_rows(table_name):
  connection = None
  cursor = None
  try:
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    results = cursor.fetchall()
    return results
  except mysql.connector.Error as err:
    print(f"❌ Database error: {err}")
  finally:
    if connection.is_connected():
      cursor.close()
      connection.close()
      print("🔒 MySQL connection closed")

def clear_table(table_name):
  connection = None
  cursor = None
  try:
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute(f"DELETE FROM {table_name}")
    connection.commit()
  except mysql.connector.Error as err:
    print(f"❌ Database error: {err}")
  finally:
    if connection.is_connected():
      cursor.close()
      connection.close()
      print("🔒 MySQL connection closed")

def test_connection():
  connection = None
  cursor = None
  try:
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT 1")
    results = cursor.fetchall()
    print(results)
  except mysql.connector.Error as err:
    print(f"❌ Database error: {err}")
  finally:
    if connection.is_connected():
      cursor.close()
      connection.close()
      print("🔒 MySQL connection closed")
 