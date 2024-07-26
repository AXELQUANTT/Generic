"""
Package devoted to import data from a sql database and create
a final dataset containing all the relevant information
"""

import datetime
import pandas as pd
import sqlite3
from sqlite3 import Cursor, OperationalError
from typing import Any, Tuple, Optional

# Steps to follow.
# 1. Import data from sql
# 2. Clean data (Nans and missing data handling)
# 3. Generate the output csv

class sql_data():
    def __init__(self, db_file:str):
        self.db = db_file
    
    def _create_connection(self) -> Cursor:
        """
        Creates sql connection to our db
        """ 
        conn = sqlite3.connect(self.db)
        curr = conn.cursor()
        return conn, curr
    
    def _query_db(self, query:str) -> Tuple[list[Any], list[str]]:
        """
        Opens a connection to the db and performs a query
        """
        conn, curr = self._create_connection()
        cursor = curr.execute(query)
        results = cursor.fetchall()
        colnames = [description[0] for description in cursor.description]
        conn.close()
        return results, colnames
    
    def create_df(self, query:str) -> pd.DataFrame:
        """
        Creates a pandas dataframe out of a query result (list)
        """
        data, colnames = self._query_db(query)
        df = pd.DataFrame(data, columns=colnames)
        return df
    
    def get_all_tables(self) -> list[Any]:
        """
        Retrieves all table names in the db
        """
        tables,_ = self._query_db("SELECT name FROM sqlite_master")

        # ignore sqlite_sequence, which only contains info about the autoincrement
        return [table[0] for table in tables if table[0]!='sqlite_sequence']
    
    def get_columns_for_table(self, table_name:str, des_cols:list[str]) -> list[str]:
        """
        Retrieves all column names for a specific table name
        """

        # This will return a tuple, where the first element is the name of the column
        table_cols, _ = db._query_db(f"SELECT name FROM PRAGMA_TABLE_INFO('{table}');")
        
        return [col[0] for col in table_cols if col[0] in des_cols]
    
    def get_table_for_column(self, col:str) -> Optional[list[str]]:
        """
        Retrieves the table name containing column=col
        """ 
        try:
            res, _ = self._query_db(f"SELECT DISTINCT name FROM sqlite_master"\
                        f" WHERE sql like '%{col}%';")
        except OperationalError:
            print(f'column name = {col} is not in the db')
            res = None
        
        return res
    
class Standarize_Data():
    def __init__(self, tables:list[str], db:sql_data, wrong_labels:dict):
        self.tables = tables
        self.db = db
        self.wrong_labels = wrong_labels
        self.df = pd.DataFrame([])

    
    
    def _load_data(self) -> pd.DataFrame:
        data = []
        for table in self.tables:
            # Create the table from the database
            query = f'SELECT * from {table} join api_homeownership on '
            df = db.create_df(query)

            # Homogenize column names to easier recognition
            df = self._homogenize_colnames(df)

            # Create the dummy variables that are not present
            # in the dataframe
            self.create_dummies


            data.append(df)

        df = pd.concat(data)
        return df
    
    def _homogenize_colnames(self, df:pd.DataFrame):

        # Remove ending '_id' from any column name'
        faulty_cols = [col for col in df.columns if col.endswith('_id')]
        df.rename(columns={col:col.split('_id')[0] for col in faulty_cols}, inplace=True)

        df.rename(columns=self.wrong_labels,inplace=True)
        
        # Finally check that all columns are equal to the desired ones
        # and report only those
        for col in wanted_in_db.keys():
            if col not in df.columns:
                raise ValueError(f'Col={col} is not in our final dataframe, check')

        return df[[col for col in wanted_in_db.keys()]]

    def _homogenize_values(self) -> pd.DataFrame:

    def _nan_treatment(self) -> pd.DataFrame:

        
    def create_standarized_data(self) -> pd.DataFrame:
        # Import the data
        df = self._load_data()

        # Homogenize values
        df = self._homogenize_values()

        # Nan treatment
        df = self._nan_treatment()

        return df

    def write_to_csv(self):
        self.df.to_csv('standarized_data.csv')


def identify_missing_columns():
    """
    Computes which columns can not be obtained directly
    from the databases
    """
    missing = []
    for coln, coltype in wanted_fields.items():
        col_tables = db.get_table_for_column(coln)
        if not col_tables:
            missing.append(coln)
        wanted_fields[coln] = (coltype,col_tables)
    print(missing)
    return missing


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

# PIPELINE
# 1. Perform some explanatory analysis on the content of the tables
# 2. From each table, select only the desired columns
# 3. Create the columns for the ones missing.
# 4. Merge all data into a single dataframe. Perform NaN treatment, etc
# 5. Write the data into a csv file.

# TO-DO: Change this so that these fields are read from a config_file
# Upon manual inspection, we see that data we need is in api_newcustomer
# and api_oldcustomer

# This will be a file read from the same folder that contains
# the name of the column and the factor we need to apply it to

db = sql_data(path)
all_tables = db.get_all_tables()

# 1. Exploratory analysis (useful for the notebook, can be ignored in the actual script)
for table in all_tables:
    query = f"SELECT * from {table} limit 10;"
    df = db.create_df(query)
    #print(table)
    #print(df.head(10))

# Check which columns are not in our db
missing_cols = identify_missing_columns()
wanted_in_db = dict((key,val) for key,val in wanted_fields.items() if key not in missing_cols)

# 2. Load data from the desired tables and inspect the columns
all_data = []
desired_tables = ['api_newcustomer', 'api_oldcustomer']
for table in desired_tables:
    # Import the table
    query = f"SELECT * from {table};"
    df = db.create_df(query)

    # Create dummies


    # Homogenize col names
    if table=='api_newcustomer':
        df = homogenize_colnames(df, wrong_labels)
    else:
        df = homogenize_values(df)
    
    # Homogenize values
    def homogenize_values(df:pd.DataFrame) -> pd.DataFrame:
        # Set the right type for each column name
        for col in df.columns:
            try:
                df[col].astype(wanted_fields[col])
            except:
                print(f'column = {col} can not be coded as type={wanted_fields[col]}')
    
    homogenize_values(df)

    # Aggregation
    all_data.append(df)

# Once data is aggregated, perform some nan treatment on the different columns
# The main 
all_data = pd.concat(all_data)










# With this 

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




# We have an id in api_oldcustomer, is that the same id
# as the api_newcustomer. Okay, seems clear at this point
# that api_oldcustomer and api_newcustomer just contain different
# loans