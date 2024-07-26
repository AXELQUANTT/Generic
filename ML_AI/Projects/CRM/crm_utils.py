"""
Package devoted to import data from a sql database and create
a final dataset containing all the relevant information
"""

import datetime
import pandas as pd
import sqlite3
from sqlite3 import Cursor
from typing import Any, Tuple

# Steps to follow.
# 1. Import data from sql
# 2. Clean data (Nans and missing data handling)
# 3. Generate the output csv

class sql_data():
    def __init__(self, db_file:str):
        self.db = db_file
    
    def _create_connection(self) -> Cursor:
        conn = sqlite3.connect(self.db)
        curr = conn.cursor()
        return conn, curr
    
    def query_db(self, query:str) -> Tuple[list[Any], list[str]]:
        conn, curr = self._create_connection()
        cursor = curr.execute(query)
        results = cursor.fetchall()
        colnames = [description[0] for description in cursor.description]
        conn.close()
        return results, colnames
    
    def create_df(self, query:str) -> pd.DataFrame:
        data, colnames = self.query_db(query)
        df = pd.DataFrame(data, columns=colnames)
        return df
    
    def check_schema(self) -> list[Any]:
        return self.query_db("SELECT name FROM sqlite_master")
    
    def get_columns_for_table(self, table_name:str) -> str:
        return self.query_db(f"PRAGMA table_info({table_name});")
    
    # So the logic will be
    # 0. Query all tables and get a small sampel of data to understand
    # the relationship among them.
    # 1. Get all the tables from the schema (check_schema)
    # 2. For each table, select only the columns that are present
    # in the wanted fields.
    # 3. 

    


path = "/home/axelbm23/Code/ML_AI/Projects/CRM/customer_data.sqlite3"
wanted_fields = {'loan_status':bool,
                 'loan_amnt':int,
                 'term':int,
                 'int_rate':float,
                 'installment':float,
                 'sub_grade':int,
                 'emp_length':int,
                 'home_ownership':int,
                 'is_mortgage':bool,
                 'is_rent':bool,
                 'is_own':bool,
                 'is_any':bool,
                 'is_other':bool,
                 'annual_inc':int,
                 'verification_status':int,
                 'is_verified':bool,
                 'is_not_verified':bool,
                 'is_source_verified':bool,
                 'issue_d':datetime.datetime,
                 'purpose':int,
                 'addr_state':int,
                 'dti':float,
                 'fico_range_low':int,
                 'fico_range_high':int,
                 'open_acc':int,
                 'pub_rec':int,
                 'revol_bal':int,
                 'revol_util':float,
                 'mort_acc':int,
                 'pub_rec_bankruptcies':int,
                 'age':int,
                 'pay_status':int} 

db = sql_data(path)
content,_ = db.check_schema()

# Perform some exploratory analysis on each of the tables
for table_info in content:
    table_name = table_info[0]
    query = f"SELECT * from {table_name} limit 10;"
    df = db.create_df(query)
    print(table_name)
    print(df.head())
#sqlite_content = db.query_db("SELECT * from api_oldcustomer limit 10")
#table_cols = db.get_table_for_col()

# The goal is, given a set of colum name, find those column names
# within the database and create a single dataframe containing all
# the information.

# For each field, we will get the table in which it is found. Then
# it's just a matter of merging tables using a unique identifier (like
# a client id or something like that)

# Exploratory analysis

# api_homeonership is a table containing equivalencies between
# api_oldcustomer.home_ownership_id and a categorical value.
# (can be merged with api_oldcustomer.home_ownership_id=api_homeownership.id)
# We will have to create boolean variables for each categorical value found in
# api_purpose, so we will need to merge


# api_purpose contains categorical values for the purpose
# of the loan. There are no categorical variables we need
# to create out of this variable, so we can ignore it.
# The purpose_id we need is in new_customer.purpose_id


# api_state contains categorical values for the addr_state
# column in the old_customer table, we do not need it either.

# api_subgrade contains categorical values for sub_grade in old_customer
# table. We do not need to create dummy variables out of this one,
# so we can skip.

# api_verificationstatus contains categorical values we will need to unfold
#. The id is both in new_customer and old_customer tables, decide which
# to pick.








# Things to solve:
# 1. What is the difference between data in api_oldcustomer
#    and data in api_newcustomer? Check that the shared
#    columns have the same information when both values are present.
#    What to do with the values when are only present in one of the tables?


