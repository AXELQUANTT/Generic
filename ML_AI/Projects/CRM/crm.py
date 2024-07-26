
import datetime
import pandas as pd
import sqlite3
from sqlite3 import Cursor, OperationalError
from typing import Any, Tuple, Optional

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
        table_cols, _ = self._query_db(f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');")
        
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
    
    def _query_for_all_data(self, table:str) -> Optional[str]:
        """
        Creates query to get all the relevant information
        """ 
        match table:
            case 'api_newcustomer':
                org_suff = '_id'
                targ_suff = 'id'
            case 'api_oldcustomer':
                org_suff = ''
                targ_suff = 'name'
            case _:
                raise ValueError(f'table has to be one of [api_newcustomer, api_oldcustomer] '
                                 f', {table} provided instead')
            
        query = f'SELECT {table}.*, '\
        f"api_homeownership.name AS api_homeownership_name, "\
        f"api_homeownership.id AS api_homeownership_id, "\
        f"api_purpose.name AS api_purpose_name, "\
        f"api_purpose.id AS api_purpose_id, "\
        f"api_state.name AS api_state_name, "\
        f"api_state.id AS api_state_id, "\
        f"api_subgrade.name AS api_subgrade_name, "\
        f"api_subgrade.id AS api_subgrade_id, "\
        f"api_verificationstatus.id AS api_verificationstatus_id, "\
        f"api_verificationstatus.name AS api_verificationstatus_name "\
        f"FROM {table} LEFT JOIN api_homeownership "\
        f"ON {table}.home_ownership{org_suff}=api_homeownership.{targ_suff} LEFT JOIN "\
        f"api_purpose ON {table}.purpose{org_suff}=api_purpose.{targ_suff} LEFT JOIN "\
        f"api_state ON {table}.addr_state{org_suff}=api_state.{targ_suff} LEFT JOIN "\
        f"api_subgrade ON {table}.sub_grade{org_suff}=api_subgrade.{targ_suff} LEFT JOIN "\
        f"api_verificationstatus ON {table}.verification_status{org_suff}="\
        f"api_verificationstatus.{targ_suff};"

        return query

    def import_all_data(self, table:str) -> pd.DataFrame:
        query = self._query_for_all_data(table)
        df = self.create_df(query)
        return df

        

path = "/home/axelbm23/Code/ML_AI/Projects/CRM/customer_data.sqlite3"
wrong_labels = {'employment_length':'emp_length',
                'issued':'issue_d',
                'payment_status':'pay_status'}

db = sql_data(path)
all_tables = db.get_all_tables()

for table in all_tables:
    query = f"SELECT * from {table} limit 10;"
    df = db.create_df(query)
    print(df.columns)


# Upon inspection, tables containing most of the data are newcustomer and oldcustomer 
desired_tables = ['api_newcustomer', 'api_oldcustomer']

for table in desired_tables:

    # Import the data
    df = db.import_all_data(table)

    # Fix wrong labels
    df.replace(wrong_labels, inplace=True)

    # Create dummies
    
# Merge dataframes
    
# Perform Nan substitution

# Write to csv



# Notes
# 1. Newcustomers has a lot of columns with id. All those columns need
# to be changed to their respective values

# # PIPELINE
# 1. Perform some explanatory analysis on the content of the tables
# 2. For each dataset:
    # 2.1 Standarize columns if there are any missing 
    # 2.2 Create the columns for the ones missing (id into normal variables)
    # 3.3 Create the dummies
# 4. Merge all data into a single dataframe. Perform NaN treatment, etc
# 5. Write the data into a csv file.

# HOMEOWNERSHIP => integer, but we need the categorical values for the classy
# PURPOSE => integer,
# SUBGRADE => integer, so we need the id (we can skip querying it for the newcust)
# VERIFICATIONSTATUS => integer, but we need the categoricals as well
# STATE => integer