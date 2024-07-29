import dateparser
import datetime
import os
import numpy as np
import pandas as pd
import re
from scipy.stats import zscore
import sqlite3
from sqlite3 import Cursor, OperationalError
from sklearn import linear_model
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
        tables,_ = self._query_db("SELECT name FROM sqlite_master;")

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
        
        # This query is where most of the standarization of the columns happens
        # In the case of the api_newcustomer, all columns on it are the ids, so
        # we only need to retrieve the names from the other ones
        # In the case of the oldcustomer, all columns corresponds to the names,
        # so we only need to retrieve the ids from the other ones
        if table=='api_oldcustomer':
            org_suff = ''
            targ_suff = 'name'
            query = f'SELECT {table}.loan_status, '\
            f'{table}.loan_amnt, {table}.term, '\
            f'{table}.int_rate, {table}.installment, '\
            f'{table}.emp_length, {table}.annual_inc, '\
            f'{table}.dti, {table}.issue_d, '\
            f'{table}.fico_range_low, {table}.fico_range_high, '\
            f'{table}.open_acc, {table}.pub_rec, '\
            f'{table}.revol_bal, {table}.revol_util, '\
            f'{table}.mort_acc, {table}.pub_rec_bankruptcies, '\
            f'{table}.age, {table}.pay_status, '\
            f'{table}.home_ownership, '\
            f'{table}.verification_status, '\
            f'{table}.purpose, '\
            f'{table}.addr_state, '\
            f'api_subgrade.id AS sub_grade_id, '\
            f'api_homeownership.id AS home_ownership_id, '\
            f'api_verificationstatus.id AS verification_status_id '\
            f'FROM {table} LEFT JOIN api_homeownership '\
            f'ON {table}.home_ownership{org_suff}=api_homeownership.name LEFT JOIN '\
            f'api_subgrade ON {table}.sub_grade{org_suff}=api_subgrade.name LEFT JOIN '\
            f'api_verificationstatus ON {table}.verification_status{org_suff}='\
            f'api_verificationstatus.name;'
        else:
            org_suff = '_id'
            targ_suff = 'id'
            query = f'SELECT {table}.loan_status, '\
            f'{table}.loan_amnt, {table}.term, '\
            f'{table}.int_rate, {table}.installment, '\
            f'{table}.employment_length AS emp_length, '\
            f'{table}.annual_inc, '\
            f'{table}.issued as issue_d, {table}.dti, '\
            f'{table}.fico_range_low, {table}.fico_range_high, '\
            f'{table}.open_acc, {table}.pub_rec, '\
            f'{table}.revol_bal, {table}.revol_util, '\
            f'{table}.mort_acc, {table}.pub_rec_bankruptcies, '\
            f'{table}.age, {table}.payment_status AS pay_status, '\
            f'{table}.home_ownership_id, '\
            f'{table}.purpose_id, {table}.sub_grade_id, '\
            f'{table}.verification_status_id, '\
            f'{table}.addr_state_id, '\
            f'api_homeownership.name AS home_ownership, '\
            f'api_verificationstatus.name AS verification_status '\
            f'FROM {table} LEFT JOIN api_homeownership '\
            f'ON {table}.home_ownership{org_suff}=api_homeownership.{targ_suff} LEFT JOIN '\
            f'api_verificationstatus ON {table}.verification_status{org_suff}='\
            f'api_verificationstatus.{targ_suff};'
        
        return query
    
    def _add_faulty_data(self, df:pd.DataFrame, table:str, col:str):
        if table=='api_newcustomer':
            return df
        
        if col=='purpose':
            # TO-DO: Instead of a dict, use a library to create the plural
            # and then just do upper case
            correct_labels = {'debt_consolidation':'DEBT_CONSOLIDATIONS',
                              'other':'OTHER', 
                              'credit_card':'CREDIT_CARDS', 
                              'major_purchase':'MAJOR_PURCHASE',
                              'medical':'MEDICAL',
                              'car':'CARS', 
                              'home_improvement':'HOME_IMPROVEMENTS',
                              'small_business':'SMALL_BUSINESS',
                              'house':'HOUSES',
                              'wedding':'WEDDINGS',
                              'vacation':'VACATIONS',
                              'moving':'MOVING',
                              'renewable_energy':'RENEWABLE_ENERGY',
                              'educational':'EDUCATIONAL'}
            df['purpose'] = df['purpose'].apply(lambda x: correct_labels[x])
            api_purpose = self.create_df("SELECT name as purpose, id as purpose_id FROM api_purpose;")
            df = pd.merge(df, api_purpose, left_on='purpose', right_on='purpose', how='left')
            # Finally remove the purpose column as we do not need it
            df.drop(['purpose'], axis=1, inplace=True)
            return df
        
        if col=='addr_state':
            # TO-DO: Add USA suffix to that col and merge
            df['addr_state'] = df['addr_state'].apply(lambda x: f'USA_{x.upper()}')
            api_state = self.create_df("SELECT name as addr_state, id as addr_state_id from api_state;")
            df = pd.merge(df, api_state, left_on='addr_state', right_on='addr_state', how='left')
            # Finally remove the addr_state column as we do not need it
            df.drop(['addr_state'], axis=1, inplace=True)
            return df
        
    def import_all_data(self, table:str) -> pd.DataFrame:
        query = self._query_for_all_data(table)
        df = self.create_df(query)

        # Upon inspection I've noticed that there are two
        # columns inside oldcustomer data
        # that do not match the values in their respective tables.
        # One is api_oldcustomer.purpose, whose values
        # are in lower case and do not exactly match the values
        # in api_purpose.name. Some of them are in singular, some
        # of them in plural, so need standarization.

        # The second ones are api_oldcustomer.addr_state, which
        # do not have the USA suffix present in api_state.
        df = self._add_faulty_data(df, table, 'purpose')
        df = self._add_faulty_data(df, table, 'addr_state')

        # Make sure we don't have duplicated columns
        if any(df.columns.duplicated()):
            raise ValueError(f'dataframe from table {table} contains duplicated columns, '\
                             f'{",".join(df.columns[df.columns.duplicated()])}')
        
        # If all data has been correctly retrieve, we should have 26 columns, assert that
        if len(df.columns)!=len(wanted_fields.keys()):
            raise ValueError(f'table = {table} is missing the following fields => '\
                             f'{",".join([str(x) for x in set(wanted_fields)-set(df.columns)])}')

        # TO-DO: Make sure all the data from external columns
        # has the correct values
        return df
    
# Create dummies for homeownership and verification_status
def create_dummies(df:pd.DataFrame, col:str) -> pd.DataFrame:
    # For each of different values in col, create a dummy column
    dummies = pd.get_dummies(df[col])
    # Format the columns according to the table 1
    dummies.columns = [f'is_{col.lower().replace(" ","_")}' for col in dummies.columns]
    df = df.join(dummies)
    df.drop([col], inplace=True, axis=1)

    # In the case of home_ownership, there is a specific column
    # we do not want to keep, is_none, so remove it
    if col=='home_ownership':
        df.drop(['is_none'], inplace=True, axis=1)

    return df

def is_tuple(value) -> bool:
    if isinstance(value, str):
        return '(' in value or ')' in value or '[' in value or ']' in value
    return False

def mean_from_tuple(value) -> float:
    value = re.sub(r"[\([{})\]]", "", value)
    return 0.5*sum([float(x) for x in value.split(', ')])

def fix_wrong_units(df:pd.DataFrame, want_fields:dict, fix_qties:dict) -> pd.DataFrame:
    """
    Function devoted to fix the values of those quantities that are wrong
    """
    # TO-DO: Clean this logic, quite convoluted
    for col in want_fields.keys():
        des_type = want_fields[col]
        if des_type==datetime.datetime:
            try:
                df[col] = df[col].apply(lambda x: dateparser.parse(x))
            except ValueError as e:
                print(f'col={col} with type datetime could not be transformed, check')

        elif col=='annual_inc':
            try:
                df[col] = df[col].apply(lambda x: x if pd.isnull(x) else des_type(mean_from_tuple(x)) if is_tuple(x) else des_type(x))
            except ValueError as e:
                print(f'col={col} could not be transformed, check')

        elif col in fix_qties:
            error = fix_qties[col][0]
            multiplier = fix_qties[col][1]
            try:
                df[col] = df[col].apply(lambda x: x if pd.isnull(x) else des_type(float(x.lower()[:-1])*multiplier)
                                         if (isinstance(x, str) and x.lower().endswith(error)) else des_type(x))
                
            except ValueError as e:
                print(f'col={col} with type {des_type} could not be transformed, check')

        else:
            try:
                df[col] = df[col].apply(lambda x: x if pd.isnull(x) else des_type(x))
            except ValueError as e:
                print(f'col={col} with type {des_type} could not be transformed, check')
    return df

def fix_fico(org_df:pd.DataFrame, tgt_df:pd.DataFrame) -> pd.DataFrame:
    fico_range = (org_df['fico_range_high']-org_df['fico_range_low']).value_counts()
    print(f'fico ranges are {fico_range}')

    # In case both values are nan, we may need another tactic. Check if those
    # cases actually happpen
    both_missing = sum(org_df['fico_range_low'].isnull() & org_df['fico_range_high'].isnull())
    print(f'Number of records with both fico_low and fico_high missing = {both_missing}')

    tgt_df.loc[tgt_df['fico_range_low'].isnull(),'fico_range_low'] = tgt_df.loc[tgt_df['fico_range_low'].isnull(),'fico_range_high'] - 4.0
    tgt_df.loc[tgt_df['fico_range_high'].isnull(),'fico_range_high'] = tgt_df.loc[tgt_df['fico_range_high'].isnull(),'fico_range_low'] + 4.0

    return tgt_df

def create_quantiles(df:pd.DataFrame, bin_type:str='qcut') -> pd.DataFrame:
    # Select columns that are continous, contain float
    # numbers, and generate quantiles for them. Note that
    # the bins generated for these quantiles are not in terms
    # of frequency, equally spacing the values of that column.
    # Cut will generate basically the bins of a historigram,
    # whereas qcut will generate bins that contain the same
    # number of data points.

    # Add to this list the ones that contain integers but
    # have a semi-continous range of values. We are including
    # these as the idea is that the groups generated have
    # big enough size

    des_qtiles = [col for col,coltype in wanted_fields.items() if coltype==float]
    des_qtiles.extend(['loan_amnt', 'annual_inc', 'fico_range_high', 'open_acc',
                   'age', 'revol_bal'])

    for col in des_qtiles:
        new_col = f'{col}_qtile'
        if bin_type=='cut':
            df[new_col] = pd.cut(df[col], 10, labels=False)
        else:
            df[new_col] = pd.qcut(df[col], 10, labels=False)
    
    return df

path = "/home/axelbm23/Code/ML_AI/Projects/CRM/customer_data.sqlite3"

# TO-DO: Read wrong quantities from file
fix_qties = {'loan_amnt':('k',1_000),
               'term':('y',12)}

wanted_fields = {'loan_status':bool,
                 'loan_amnt':int,
                 'term':int,
                 'int_rate':float,
                 'installment':float,
                 'sub_grade_id':int, # need to remove the id at the end
                 'emp_length':int,
                 'home_ownership':str,
                 'home_ownership_id':int, # need to remove the id at the end
                 'annual_inc':int,
                 'verification_status':str,
                 'verification_status_id':int, # need to remove the id at the end
                 'issue_d':datetime.datetime,
                 'purpose_id':int, # need to remove the id at the end
                 'addr_state_id':int, # need to remove the id at the end
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
all_tables = db.get_all_tables()

for table in all_tables:
    query = f"SELECT * from {table} limit 10;"
    df = db.create_df(query)
    print(df.columns)


# Upon inspection, tables containing most of the data are newcustomer and oldcustomer 
desired_tables = ['api_oldcustomer', 'api_newcustomer']
dfs = []
oldcustomer = db.create_df("SELECT * from api_oldcustomer;")
newcustomer = db.create_df("SELECT * from api_newcustomer;")

# So for the oldcustomer table, there are some case labels that
# do not match exactly the ones in purpose table. Format them
# so that they match
purpose = db.create_df("SELECT * from api_purpose;")

# api_state values start with USA, whereas
# the values in oldcustomer database do not start with that
# suffix. 
addr_state = db.create_df("SELECT * from api_state;")
subgrade = db.create_df("SELECT * from api_subgrade;")
home_ownership = db.create_df("SELECT * from api_homeownership;")

for table in desired_tables:

    # Import the data
    df = db.import_all_data(table)

    # Upon inspection we see that there are three columns
    # that contain wrong quantities.
    # 1. loan_amnt, some rows are in string format with 'xx.k'
    # 2. term, some rows are in years. We assume payments are
    # monthly, so we can multiply that factor for 12 to obtain
    # the number of payments in the loan.
    # 3. issued has different datetime formats, one contain
    # Days, others just Years and Months. Standarize them.
    # 4. annual_inc has a range for some observations instead
    # of a single value (e.g (72000.0, 80000.0]) 
    # For those cases, compute the mean.
    # Auxiliary column to debug
    #df['table'] = table
    dfs.append(df)
    
df = pd.concat(dfs)

# Fix wrong quantities and assign proper format to columns
df = fix_wrong_units(df, wanted_fields, fix_qties)

# Compute percentage of NaN values for each column
nans_per_col = df.isnull().sum()
nans_per_col_dist = nans_per_col*100.0/len(df)
cols_with_nans = df.columns[df.isnull().any()]

# For the number of public_rec_bankruptcies and
# revol_util, the numbers are below the 0.05%, a number
# of observations so small that we assume no statistical
# difference will be imposed by removing these faulty records

# The whole concept of NaN replacement is to ensure
# that the original distributions of values for each
# of the missing targets remains unaffected once we
# have replaced them with the synthetic data.
# Therefore, we do not want to add a bias on the
# the distribution of values with out choice


# TO-DO! DO NOT THINK ANYTHING, JUST IMPLEMENT THIS!
# So what we will generate quantiles for the continous
# variables, then groupby all the categories and
# replace the nans on each category with the medians
# of those groups. Compute the distributions before
# and after the replacement to ensure we did not add
# any artifact. Also, do the replacement at the end, 
# to ensure that the order at which we replace the NaNs
# is irrelevant (keep the indexs of row,col to replace it
# later)

df = create_quantiles(df)
gbpy_cols = ['loan_status', 'term', 'emp_length',
             'pub_rec', 'mort_acc', 
             'pub_rec_bankruptcies', 'pay_status',
             'home_ownership_id', 'verification_status_id', 'purpose_id',
             'addr_state_id']

gbpy_cols.extend([col for col in df.columns if col.endswith('_qtile')])

# For each of the missing cols, we will select which columns we want to groupby
# with
# TO-DO: Assing columns that are related with the missing qties. For now we will use
# all.

# emp_length => age_qtile,  annual_inc_qtile, verification_status_id, addr_state (demographic factors)
# fico_range_low/high => solved
# revol_util => revol_bal_qtile, open_acc_qtile, dti => measure of how much credit we are using from all available credit
# mort_acc => home_ownership, purpose, open_acc => number of mortatge accounts
# pub_rec_bankruptcies => loan_status, dti, pub_rec => measures of late payments

demographic_fact = ['age_qtile', 'annual_inc_qtile']
credit_usage_fact = ['revol_bal_qtile', 'open_acc_qtile', 'annual_inc_qtile']
mort_fact = ['annual_inc_qtile']
default_factors = ['pub_rec', 'pay_status']

predictors = {'emp_length':demographic_fact,
              'revol_util':credit_usage_fact,
              'mort_acc':mort_fact,
              'pub_rec_bankruptcies':default_factors}

def fix_nans(df:pd.DataFrame, faulty_magn:dict, method:str='median') -> pd.DataFrame:
    org_table = df.copy()

    # In the case of nans in the FICO column, we can easily extract
    # its value from the low-high, as it is shown, the difference
    # between those two values remains constant over the entire dataset
    df = fix_fico(org_table, df)

    for target_col, predictors in faulty_magn.items():
        print(target_col)
        # Select the desired columns to groupby with.
        # Make sure we do not groupby the column we are trying
        # to predict.
        new_values = org_table.groupby(predictors).agg({target_col:method}).reset_index()
        
        # Change the name of the predicted value to ease further manipulation
        col_est = f'{target_col}_estimated'
        new_values.rename(columns={target_col:col_est}, inplace=True)

        df = pd.merge(df, new_values, left_on=predictors, right_on=predictors, how='left')
        
        # Finally assing the new value generated only to the rows that have a missing value
        df.loc[df[target_col].isnull(), target_col] = df.loc[df[target_col].isnull(), col_est]
        # Check if there are any nans in the computed target
        if any(df[target_col].isnull()):
            raise ValueError(f'target_col={target_col} still contains Nans, try to increase'
                  f'the group size')
        # Make sure compute value is type conformant
        df[target_col] = df[target_col].astype(wanted_fields[target_col])

        # Remove the auxiliar column
        df.drop([col_est], axis=1, inplace=True)

    return df

df = fix_nans(df, predictors)

# Remove the extra columns
df = df.loc[:, df.columns.isin(wanted_fields.keys())]

# After the NaN treatment, no NaNs should be present, assert it
if df.isnull().sum().sum()!=0:
    raise ValueError(f'DataFrame still contains Nans, check!')

# Create the dummy variables
df = create_dummies(df, 'verification_status')
df = create_dummies(df, 'home_ownership')

# Rename _id columns to make them conformant with the instructions
renamed = dict((col, col[:-3]) for col in df.columns if col.endswith('_id'))
df.rename(columns=renamed, inplace=True)

# Check that the output dataframe has exactly 32 columns
print(f'df correct columns = {len(df.columns)==32}')

# Write to csv
output_file = f'{os.getcwd()}/final_dataset.csv'
df.to_csv(output_file, index=False)