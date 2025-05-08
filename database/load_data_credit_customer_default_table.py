import pandas as pd
import psycopg2
from sqlalchemy import create_engine

db_config = {
    "user": "INSERT",
    "password": "INSERT",
    "host": "localhost",
    "port": 5432,
    "database": "credit_etl"
}

file_path = "/Users/johnryan/Downloads/default of credit card clients.csv"

#load the csv
df = pd.read_csv(file_path)

#Rename columns to match SQL naming convention
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

#load data to postgres DB

engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
df.to_sql("raw_customer_default_payment", engine, schema='raw', if_exists="replace", index=False)
