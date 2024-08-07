import dateparser
import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from scipy.stats import ks_2samp
import sqlite3
from sqlite3 import Cursor
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
        """
        Function devoted to fix wrong values present in columns
        purpose and addr_state
        """

        if table=='api_newcustomer':
            return df
        
        if col=='purpose':
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
            df['addr_state'] = df['addr_state'].apply(lambda x: f'USA_{x.upper()}')
            api_state = self.create_df("SELECT name as addr_state, id as addr_state_id from api_state;")
            df = pd.merge(df, api_state, left_on='addr_state', right_on='addr_state', how='left')
            # Finally remove the addr_state column as we do not need it
            df.drop(['addr_state'], axis=1, inplace=True)
            return df
        
    def import_all_data(self, table:str) -> pd.DataFrame:
        """
        Main function of the class, gets the relevant
        data from the two tables and fixes the wrong values
        """
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

        return df
    

def create_dummies(df:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Creates a dummy variable for dataframe df for the categorical
    column specified by col
    """

    # For each of different values in col, create a dummy column
    dummies = pd.get_dummies(df[col])
    # Format the columns according to the table 1
    dummies.columns = [f'is_{col.lower().replace(" ","_")}' for col in dummies.columns]
    df = df.join(dummies)
    df.drop([col], inplace=True, axis=1)

    # In the case of home_ownership, there is a specific column
    # we do not want to keep (acc to instructions), is_none, 
    # so remove it
    if col=='home_ownership':
        df.drop(['is_none'], inplace=True, axis=1)

    return df

def is_tuple(value) -> bool:
    """
    Returns True if value contains a range as a string
    """

    if isinstance(value, str):
        return '(' in value or ')' in value or '[' in value or ']' in value
    return False

def mean_from_tuple(value) -> float:
    """
    Computes mean from two string values
    """

    value = re.sub(r"[\([{})\]]", "", value)
    return 0.5*sum([float(x) for x in value.split(', ')])

def fix_wrong_units(df:pd.DataFrame, want_fields:dict, fix_qties:dict) -> pd.DataFrame:
    """
    Function devoted to fix the values of those quantities that are wrong
    """

    for col in want_fields.keys():
        des_type = want_fields[col]
        if des_type==datetime.datetime:
            try:
                df[col] = df[col].apply(lambda x: dateparser.parse(x))
            except ValueError:
                print(f'col={col} with type datetime could not be transformed, check')

        elif col=='annual_inc':
            try:
                df[col] = df[col].apply(lambda x: x if pd.isnull(x) else des_type(mean_from_tuple(x)) 
                                        if is_tuple(x) else des_type(x))
            except ValueError:
                print(f'col={col} could not be transformed, check')

        elif col in fix_qties:
            error = fix_qties[col][0]
            multiplier = fix_qties[col][1]
            try:
                df[col] = df[col].apply(lambda x: x if pd.isnull(x) else des_type(float(x.lower()[:-1])*multiplier)
                                         if (isinstance(x, str) and x.lower().endswith(error)) else des_type(x))
                
            except ValueError:
                print(f'col={col} with type {des_type} could not be transformed, check')

        else:
            try:
                df[col] = df[col].apply(lambda x: x if pd.isnull(x) else des_type(x))
            except ValueError:
                print(f'col={col} with type {des_type} could not be transformed, check')
    return df

def fix_fico(org_df:pd.DataFrame, tgt_df:pd.DataFrame) -> pd.DataFrame:
    """
    Function computes the fico value from the low/high, depending
    which one is valid
    """

    # In case both values are nan, we may need another tactic. Check if those
    # cases actually happpen
    both_missing = sum(org_df['fico_range_low'].isnull() & org_df['fico_range_high'].isnull())
    print(f'Number of records with both fico_low and fico_high missing = {both_missing}')

    tgt_df.loc[tgt_df['fico_range_low'].isnull(),'fico_range_low'] = tgt_df.loc[tgt_df['fico_range_low'].isnull(),
                                                                                'fico_range_high'] - 4.0
    tgt_df.loc[tgt_df['fico_range_high'].isnull(),'fico_range_high'] = tgt_df.loc[tgt_df['fico_range_high'].isnull(),
                                                                                  'fico_range_low'] + 4.0

    return tgt_df

def create_quantiles(df:pd.DataFrame, bin_type:str='qcut') -> pd.DataFrame:
    """
    Creates quantiles for a set of specified columns
    """

    des_qtiles = [col for col,coltype in wanted_fields.items() if coltype==float]
    des_qtiles.extend(['loan_amnt', 'annual_inc', 'fico_range_high', 'open_acc',
                   'age', 'revol_bal', 'pub_rec'])

    for col in des_qtiles:
        new_col = f'{col}_qtile'
        if bin_type=='cut' or col=='pub_rec':
            # Compute quantiles on the dat => diff obs per group
            bins = pd.cut(df.loc[~df[col].isnull(),col], 10, labels=False)
            df.loc[~df[col].isnull(),new_col] = bins
        else:
            # Computes qtiles on the freq => same obs per group
            df[new_col] = pd.qcut(df[col], 10, labels=False, duplicates='drop')
    
    return df

def check_dis_equality(samp_1:pd.Series, samp_2:pd.Series, ci:float=0.05) -> None:
    """
    Function that prints True if sample_1 and sample_2
    come from the same distribution
    ci: confidence interval
    """

    result = ks_2samp(samp_1, samp_2)

    pval = result.pvalue

    if pval > ci:
        print(f"sample1 and sample2 come from the same distribution")
    else:
        print(f"sample1 and sample2 DO NOT come from the same distribution")
    pass

def fix_nans(org_df: pd.DataFrame, df:pd.DataFrame, nan_col:str, 
             pred:list[str], method:str='median') -> pd.DataFrame:
    """
    Takes an original dataframe, and fills the nans present in nan_col
    with the median of values for that column specified in list of columns
    in pred. The method used to replace those nans is specified by method
    """

    # Select the desired columns to groupby with.
    new_values = org_df.groupby(pred).agg({nan_col:method}).reset_index()
    
    # Change the name of the predicted value to ease further manipulation
    col_est = f'{nan_col}_estimated'
    new_values.rename(columns={nan_col:col_est}, inplace=True)

    df = pd.merge(df, new_values, left_on=pred, right_on=pred, how='left')
    
    # Finally assing the new value generated only to the rows that have a missing value
    df.loc[df[nan_col].isnull(), nan_col] = df.loc[df[nan_col].isnull(), col_est]
    # Check if there are any nans in the computed target
    if any(df[nan_col].isnull()):
        raise ValueError(f'target_col={nan_col} still contains Nans, '\
                         f'try to increase the group size')
    # Make sure compute value is type conformant
    df[nan_col] = df[nan_col].apply(lambda x: round(x) if wanted_fields[nan_col]==int else 
                                    wanted_fields[nan_col](x))

    # Remove the auxiliar column
    df.drop([col_est], axis=1, inplace=True)

    return df

def print_nans_per_col(df:pd.DataFrame) -> None:
    """
    Prints NaNs per column, purely informative
    """

    nans_per_col = df.isnull().sum()
    nans_per_col_dist = nans_per_col*100.0/len(df)
    print('Distribution of NaNs per columns')
    print(nans_per_col_dist[nans_per_col_dist!=0.0])

def plot_hist(org_df:pd.DataFrame, df:pd.DataFrame, nan_tgt:str) -> None:
    """
    Computes histogram for a specific column over two different
    dataframes
    """

    plt.hist(df[nan_tgt], label='after', alpha=0.7, color='skyblue')
    plt.hist(org_df.loc[~org_df[nan_tgt].isnull(), nan_tgt], label='before', 
             alpha=0.5, color='red')
    plt.title(f'Histogram of {nan_tgt} before/after NaN imputation')
    plt.show()

def plot_scatter(df:pd.DataFrame, x:str, y:str, gpby:list[str]):
    """
    Computes a scatter plot x,y grouping by df by gpby columns
    """

    fig,ax = plt.subplots()
    if gpby==[]:
        ax.plot(df[x],df[y], linestyle='-', marker='o')
    else:
        for key,group in df.groupby(gpby):
            ax.plot(group[x],group[y], linestyle='-', marker='o', label=key)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=gpby)
    ax.grid(True)
    plt.title(f'{y} vs {x}')
    plt.show()

def show_relationship(df:pd.DataFrame, nantgt:str, x:str, 
                      gpby_data:list[str]=[], gpby_plot:list[str]=[]) -> None:
    """
    Pre-computes dataframes grouping them by gpby data and plots
    them 
    """

    nonans_df = df.loc[~df[nantgt].isnull(),:]
    gpby_df = nonans_df.groupby(gpby_data).agg({nantgt:'median'}).reset_index()

    plot_scatter(gpby_df, x=x, y=nantgt, gpby=gpby_plot)

def merge_db_tables() -> pd.DataFrame:
    """
    Function devoted to query the db tables containing the information
    and generate a single dataframe
    """

    desired_tables = ['api_oldcustomer', 'api_newcustomer']
    dfs = []
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
        dfs.append(df)

    return pd.concat(dfs)

def perform_nan_imputation(df:pd.DataFrame,
                           predictors:dict) -> Optional[pd.DataFrame]:
    """
    Function imputes Nan values for columns specified
    in predictors.keys() using predictors.values() as resources
    """

    org_df = df.copy()
    df = fix_fico(org_df, df)
    print('Checking fico sanity')
    check_dis_equality(org_df.loc[~org_df['fico_range_high'].isnull(), 'fico_range_high'],
                    df['fico_range_high'])
    plot_hist(org_df, df, 'fico_range_high')

    for nantgt,pred in predictors.items():
        df = fix_nans(org_df, df, nantgt, pred, method='median')
        print(f'Checking {nantgt} sanity')
        check_dis_equality(org_df.loc[~org_df[nantgt].isnull(),nantgt],
                    df[nantgt])
        
        # Plot distributuon of values before/after the Nan replacement
        plot_hist(org_df, df, nantgt)

    # Remove the extra columns
    df = df.loc[:, df.columns.isin(wanted_fields.keys())]

    # After the NaN treatment, no NaNs should be present, assert it
    if df.isnull().sum().sum()!=0:
        raise ValueError(f'DataFrame still contains Nans, check!')
    
    return df

def write_to_csv(df)-> None:
    """
    Saves input df as final_dataset.csv in the
    directory where the script runs
    """

    # Rename _id columns to make them conformant with the instructions
    renamed = dict((col, col[:-3]) for col in df.columns if col.endswith('_id'))
    df.rename(columns=renamed, inplace=True)

    # Check that the output dataframe has exactly 32 columns
    # The ones provided in the case
    if len(df.columns)!=32:
        raise ValueError('df does not have the right columns, check!')

    # Write to csv
    output_path = os.path.abspath(os.path.dirname(__file__))
    output_file = f'{output_path}/final_dataset.csv'
    df.to_csv(output_file, index=False)

path = f"{os.path.abspath(os.path.dirname(__file__))}/customer_data.sqlite3"

fix_qties = {'loan_amnt':('k',1_000),
               'term':('y',12)}

wanted_fields = {'loan_status':bool,
                 'loan_amnt':int,
                 'term':int,
                 'int_rate':float,
                 'installment':float,
                 'sub_grade_id':int,
                 'emp_length':int,
                 'home_ownership':str,
                 'home_ownership_id':int,
                 'annual_inc':int,
                 'verification_status':str,
                 'verification_status_id':int,
                 'issue_d':datetime.datetime,
                 'purpose_id':int,
                 'addr_state_id':int,
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

print(f"Querying db...")
db = sql_data(path)
df = merge_db_tables()

print(f'Fixing wrong units')
# Fix wrong quantities and assign proper format to columns
df = fix_wrong_units(df, wanted_fields, fix_qties)
print_nans_per_col(df)

print('Creating quantiles for analysis')
df = create_quantiles(df)

demographic_fact = ['annual_inc_qtile']
credit_usage_fact = ['revol_bal_qtile', 'open_acc_qtile', 'annual_inc_qtile']
mort_fact = ['revol_bal_qtile', 'annual_inc_qtile']
default_factors = ['pub_rec_qtile']

predictors = {'emp_length':demographic_fact,
              'revol_util':credit_usage_fact,
              'mort_acc':mort_fact,
              'pub_rec_bankruptcies':default_factors}

# Show relationship between annual_inc, age and emp_length
show_relationship(df, 'emp_length', 'annual_inc_qtile', ['age_qtile', 'annual_inc_qtile'], ['age_qtile'])
show_relationship(df, 'emp_length', 'age_qtile', ['age_qtile', 'annual_inc_qtile'], ['annual_inc_qtile'])

# Print relationships of revol_util with revol_balance, open_acc and annual_income
show_relationship(df, 'revol_util', 'revol_bal_qtile', ['revol_bal_qtile','open_acc_qtile'], ['open_acc_qtile'])
show_relationship(df, 'revol_util', 'open_acc_qtile', ['revol_bal_qtile','open_acc_qtile'], ['revol_bal_qtile'])
show_relationship(df, 'revol_util', 'annual_inc_qtile', ['annual_inc_qtile','open_acc_qtile'], ['open_acc_qtile'])

# Print relationships of mortatge_accounts and income
show_relationship(df, 'mort_acc', 'revol_bal_qtile', predictors['mort_acc'], ['annual_inc_qtile'])
show_relationship(df, 'mort_acc', 'annual_inc_qtile', predictors['mort_acc'], ['revol_bal_qtile'])

# Print relationship of pub_rec_bankruptcies and pub_rec
show_relationship(df, 'pub_rec_bankruptcies', 'pub_rec_qtile', ['pub_rec_qtile'], [])

print('Performing NaN imputation')
# Perform NaN imputation
df = perform_nan_imputation(df, predictors)

# Create the dummy variables
df = create_dummies(df, 'verification_status')
df = create_dummies(df, 'home_ownership')

write_to_csv(df)
print('Program complete')