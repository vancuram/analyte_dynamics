    # %% [markdown]
# ---
# title: Value with volume from AmsterdamUMCdb
# author: Martin Vancura
# date: "`r Sys.Date()`"
# ---

# # Dataset
# Data from Mimic and AmsterdamUMCdb are currently used. Both dataset were extracted with the same method. All relevant test were extracted and converted to default units.
# For every ICU stay some covariets were extracted as well. Fluid volume given in first 24 (iv crystaloid) was calculated as sum of volume of all identified crystaloid in database with star of administration within first 24 h. Albumin as sum of all albumin given in first 24 hours.
# Blood cells (ery), plasma volume, and thrombocytes as sum of all administered with star within first 24 hours.
# CRRT (or dialyses) was set as true when the CRRT were initiated within first 24h. Max CRP was taken as maxmal with first 24 hours and set to -1 if not measured (we handle that as low)


# %%
#| echo: false
#| output: asis



import psycopg2
import pandas as pd
import os
from sqlalchemy import create_engine
#from abr import abr
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from numdifftools import Hessian
from scipy.stats import norm, zscore, normaltest
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import datashader as ds
import datashader.transfer_functions as tf
import plotly.graph_objects as go
import math
import time
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
from IPython.display import Markdown, display
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import shapiro, normaltest, probplot, uniform, t, norm, trimboth, skewnorm, jf_skew_t
from skewt_scipy.skewt import skewt
from scipy.special import gammaln
from properscoring import crps_ensemble
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.insert(0, os.path.expanduser('~/icu_database/lib'))
from probabilitymodel import probabilitymodel as pm
import pickle

#from datashader import plotting
##########3 Settings ###############33


print(f"""\n### Settings\n""")
time_window = 168 #hours
z_score_treshold = 4
relative = True
max_data = None
separating_val_name = 'Volume'
vol_treshold = 5000
crp_treshold = 25
alb_treshold = 0
minimal_set_size = 20
pd.options.mode.copy_on_write = True

# Create connection using SQLAlchemy (for Pandas)
#engine = create_engine(f'postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
pd.option_context('display.max_rows', None, 'display.max_columns', None)

# Define the table names

DATA_DIR = "./"

os.makedirs(DATA_DIR, exist_ok=True)

units = {3024561: 'g/l', 3003458: 'mmol/l', 3006140: 'umol/l', 3012095: 'mmol/l', 3032080: '', 43534077: 'mmol/l', 40762351: 'g/dl',
         3020564: 'umol/l', 3010813: '$10^9$', 3007461: '$10^9$', 3035995: 'IU/l',
         3015377: 'mmol/l', 3000285: 'mmol/l', 3020460: 'mg/dl', 42869588: 'per.'}
#list_of_tests_sql = ", ".join([str(a) for a in test_original])
list_of_tests = [3024561, 3003458, 3006140, 3012095, 3032080, 43534077, 40762351, 3020564, 3010813, 3007461, 3035995, 3015377, 3000285, 3020460, 42869588]
clinical_range = {3024561: [6, 60, 10],
                  3003458: [0.05, 8, 0.6],
                  3006140: [0.1, 1000, 30],
                  3012095: [0.1, 5, 1],
                  3032080: [0.1, 10, 2],
                  43534077: [0.1, 80, 4],
                  40762351: [2, 25, 2],
                  3020564: [0, 1000, 100], 
                  3010813: [0, 300, 10,],
                  3007461: [0, 2000, 100],
                  3035995: [0, 2000, 100],
                  3015377: [0, 10, 2],
                  3000285: [80, 200, 40],
                  3020460: [0, 1000, 100],
                  42869588: [5, 100, 30]
                  }
selected_columns = '*'
test_names = {3024561: 'Albumin', 3003458: 'Phosphate', 3006140: 'Bilirubin total', 3012095: 'Magnesium', 3032080: 'INR',
              43534077: 'Urea', 40762351: 'Hemoglobin', 3020564: 'Creatinine', 3010813: 'WBC', 3007461: 'Platelets', 3035995: 'ALP', 3015377: 'Calcium', 3000285: 'Sodium', 3020460: 'CRP', 42869588: 'Hematocrit'}
#maximal_measurement_error = {3024561: 5, 3003458: 0.5, 3006140: 20, 3012095: 0.4, 3032080: 0.4, 43534077: 0.4, 40762351: 2, 3020564: 20, 3010813: 5, 3007461: 50, 3035995: 5, 3015377: 0.2, 3000285: 2, 3020460: 2, 42869588: 2}
minimal_measurement_error = {3024561: 0.3, 3003458: 0.02, 3006140: 0.1, 3012095: 0.02, 3032080: 0.025, 43534077: 0.05, 40762351: 0.2, 3020564: 2, 3010813: 0.1, 3007461: 1, 3035995: 3, 3015377: 0.02, 3000285: 0.3, 3020460: 1, 42869588: 1}
test_cutoffs = {3024561: 5, 3003458: 0.5, 3006140: 5, 3012095: 0.5, 3032080: 0.5, 43534077: 2, 40762351: 1, 3020564: 20, 3010813: 1, 3007461: 20, 3035995: 10, 3015377: 0.5, 3000285: 3, 3020460: 5, 42869588: 3}
list_of_tests = sorted(list_of_tests, key=lambda x: test_names[x].lower())

ugly_counter = 0

def plt_to_markdown(pltx, name=None, label=None):
    global ugly_counter

    if name == None:
        name = f'{ugly_counter}.png'
        ugly_counter += 1

    name_adres = './fig/' + name 
    out_str = r"""
        \begin""" + """{figure}
        \includegraphics[width=\linewidth]{"""

    out_str += f"{name_adres}" + """}
    """
    if not label is None:
        out_str += "\label{" + label + """}
        """

    out_str += """\end{figure}
    """

    out_str = f"""![{label}]({name_adres} "{label}")"""
    #print(out_str)
    #print(out_str)
    pltx.savefig(name_adres, format='png')
    pltx.close()


    display(Markdown(out_str))
    return 0

def recalculate_relative_to_first(df):
    # recultulates to first value
    # time is calculated as difference from first time
    df = df.sort_values(by=['stayid','charttime']) # it is sorting so doesnt matter which time is used
    df['first'] = df.groupby('stayid')['value'].transform('first')
    df['normalized_value'] = df['value'] / df['first']
    df['difference'] = df['value'] - df['first']
    df['first_time'] = df.groupby('stayid')['charttime'].transform('first')
    df['time'] = df['charttime'] - df['first_time']
    df = df[df['time'] > 0]
    return df[['time', 'value', 'difference', 'normalized_value']]


def get_morning_time(df):
    """
    Returns hour with best alligment

    df: dataframe with data of one patient (times)
    """
    frequency = np.zeros(73) 
    mask = np.zeros(50, dtype=int)
    valid_times = df[(df >= 0) & (df <= 72)]

    # Convert to int bin index
    bin_indices = valid_times.astype(int)

    # Use np.bincount to count occurrences, and add to frequency
    frequency += np.bincount(bin_indices, minlength=73)

    # Convert 1-based positions to 0-based indices
    positions = [0, 1, 23, 24, 25, 47, 48, 49]
    positions = [p for p in positions if 0 <= p <= 49]  # Make sure they are within bounds

    # Set specified positions to 1
    mask[positions] = 1


    # Step 2: Convolve with the frequency array
    convolved = np.correlate(frequency, mask, mode='valid')

    # Step 3: Find the index of the maximum value
    max_index = np.argmax(convolved)


    return max_index


import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.special import gammaln
from numdifftools import Hessian


def generate_model_evaluation_table(models, model_names=None, x=None, y=None):
    """
    Evaluate list of ProbabilityModel objects and generate LaTeX table string.

    Args:
        models: list of ProbabilityModel objects (already fitted)
        model_names: list of names (optional)
        x: list of x arrays for each model (optional, defaults to model.x)
        y: list of y arrays for each model (optional, defaults to model.y)

    Returns:
        LaTeX table string.
    """
    results = []
    for i, model in enumerate(models):
        xi = x[i] if x is not None else None
        yi = y[i] if y is not None else None

        eval_res = model.evaluate_probability_model(model, xi, yi, plot=True)
        eval_res['Model'] = model_names[i] if model_names else f"Model {i+1}"
        results.append(eval_res)

    # Convert to DataFrame for nice formatting
    df_results = pd.DataFrame(results)
    df_results = df_results[['Model', 'Shapiro-Wilk p-value', 'D’Agostino p-value', 
                             'Coverage 95% CI', 'Negative Log Likelihood']]

    # Format floats for publication
    formatted_df = df_results.copy()
    for col in formatted_df.columns[1:]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

    # Generate LaTeX table string
    latex_table = formatted_df.to_latex(index=False, escape=True, caption="Model Evaluation Results",
                                        label="tab:model_evaluation", column_format="lcccc")

    return latex_table


def make_pairs(df, windows_start=[0], windows_length=[24]):

    all_pairs = df[['stayid', 'charttime', 'value']].copy()
    all_pairs['charttime'] = all_pairs['charttime'].astype(float)
    maximal_time = (np.array(windows_start) + np.array(windows_length)).max()
    all_pairs = all_pairs[all_pairs['charttime'] < maximal_time]

    # Merge on stayid to form all pairs
    merged = all_pairs.merge(
        all_pairs,
        on='stayid',
        suffixes=('_1', '_2')
    )

    # Filter: non-identical, charttime_2 > charttime_1, and < 24 hour difference
    result = merged[
        (merged['charttime_1'] < merged['charttime_2']) &
        ((merged['charttime_2'] - merged['charttime_1']) < max(windows_length))
    ]


    # Add new columns
    result = result.reset_index(drop=True)
    result['stayid'] = result.index

    # shifting times se they fit to window
    for start, length in zip(windows_start, windows_length):
        mask = (result['charttime_1'] >= start) & (result['charttime_2'] < start + length)
        time_shift = (result.loc[mask, 'charttime_1'].values - start)
        result.loc[mask, 'charttime_2'] = result.loc[mask, 'charttime_2'].values - time_shift
        result.loc[mask, 'charttime_1'] = start

    first = pd.DataFrame({
        'stayid': result['stayid'],
        'charttime': result['charttime_1'],  # start point
        'value': result['value_1']
    })

    # Second point in each pair
    second = pd.DataFrame({
        'stayid': result['stayid'],
        'charttime': result['charttime_2'],  # time difference
        'value': result['value_2']
    })

    #breakpoint()

    all_pairs = pd.concat([first, second], ignore_index=True)
    return all_pairs

def make_windows(df1, window_starts=(0, 12, 24), window_lengths=(12, 12, 24), morning_tolerance = 1.5,
                   times=['charttime', 'charttime', 'charttime'], 
                ):
    # minimal_count_int is interval for fitting model (preventing fitting if there is not enough
    
    dfs = []


    # All data 
    for indx, start in zip(range(len(window_starts)), window_starts):

        # filtering based on set time
        if times[indx] == 'charttime':
            a = df1[(df1['charttime'] >= start)
                    & (df1['charttime'] < start + window_lengths[indx])].copy()
        else:
            a = df1[(df1['clock_time'] > start - morning_tolerance)
                    & (df1['clock_time'] < start + window_lengths[index])].copy()
            
            # Fitering values without value within set morning tests
            a['morning_check'] = a.groupby('stayid')['clock_time'].transform('first')
            a = a[(a['morning_check'] - start).abs() < morning_tolerance]

        dfs.append(a)

        ## FIRST data frame
        # initialization and fitting
    return dfs

def split_into_quartiles(df, column='first_value'):
    """
    Splits the dataframe into four dataframes based on quartiles of the specified column.
    
    Parameters:
    - df: input dataframe
    - column: column name to compute quartiles on
    
    Returns:
    - dfs: list of 4 dataframes [Q1, Q2, Q3, Q4]
    """
    # Calculate quartiles
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q2 = df[column].quantile(0.50)
    q3 = df[column].quantile(0.75)
    
    # Split into quartile-based dataframes
    df_q1 = df[df[column] <= q1]
    df_q2 = df[(df[column] > q1) & (df[column] <= q2)]
    df_q3 = df[(df[column] > q2) & (df[column] <= q3)]
    df_q4 = df[df[column] > q3]
    
    return [df_q1, df_q2, df_q3, df_q4], [q1, q2, q3]

        
def plot_quartils(dfs, qs, x_col='time', y_col='difference', window_start=0, window_length=12, plot_lengt=6, measurement_std=None, 
                 number_of_windows=10, minimal_set_size=10, cutoff=None):
    """
    Plots four subplots (scatter plots) with fitted ProbabilityModel curve and returns model objects.
    
    Parameters:
    - df1, df2, df3, df4: DataFrames to plot.
    - ProbabilityModel: Class with .fit() and .predict() methods.
    - x_col, y_col: Columns for x and y axes.
    - alpha: Scatter plot transparency.
    
    Returns:
    - models: List of fitted model objects for each dataframe.
    """ 
    models = []
    

    for i, (df, ax) in enumerate(zip(dfs, axes)):
        df = df[(df['charttime'] < plot_lengt)]
        df = make_pairs(df, windows_start=[window_start], windows_length=[window_length]) 
        df = recalculate_relative_to_first(df)
        #df = df[df['difference'].abs() <= df['difference'].abs().quantile(0.99)]

        x = df['time']
        y = df['difference']
        alpha = min(1, 100/max(1,len(y)))
        #breakpoint()

        # Scatter plot
        #ax.scatter(x, y, alpha=alpha)
        ax.set_title(f"{labels[i]}")
        #ax.set_xlabel(x_col)
        #ax.set_ylabel(y_col)
        ax.grid(linestyle='--', color='lightgrey')

        # Fit model
        if len(df) > 10:  # Prevent fitting if too few data points
            model = pm(df['time'], df['difference'])
            model.fit(measurement_std=measurement_std)
        else:
            model = None
        
        models.append(model)
    return models


def initial_filtering(df):
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df['charttime']):
        df['charttime'] = df.groupby('stayid')['charttime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
    else:
        df['charttime'] = df.groupby('stayid')['charttime'].transform(lambda x: (x - x.min()))

    df = df[df['charttime'] < 168]
    df = df[df['itemid'].isin(list_of_tests)]

    df['morning_time'] = df.groupby('stayid')['charttime'].transform(get_morning_time) 
    df['clock_time'] = df['charttime'] - df['morning_time']
    #print(df[df['crrt'] == True])
    #print(df['crrt'])
    df = df.drop_duplicates()

    bins = range(-24, 169, 2)
    df['charttime'].hist(bins=bins, alpha=1.0, label='Time from 1. measurement')
    df['clock_time'].hist(bins=bins, alpha=0.8, label='Correlation adjusted time')
    plt.title('Correlation time adjustment')
    plt.xlabel('hours')
    plt.ylabel('number of tests')
    plt.legend()
    plt.show()


    removal_stats = []
    summary_stats = []

    # Combined plot
    fig, axes = plt.subplots(len(list_of_tests), 3, figsize=(20, 4 * len(list_of_tests)), layout="constrained")

    filtered_data = pd.DataFrame()

    for i, test in enumerate(list_of_tests):
        data_to_process = df[df['itemid'] == test][:max_data].copy()

        original_len = len(data_to_process)

        # Bins for histogram
        bins = np.linspace(data_to_process['value'].min(), data_to_process['value'].max(), 40)

        # HISTOGRAM BEFORE FILTER
        axes[i, 0].hist(data_to_process['value'], bins=bins, alpha=0.6, label='Original', log=True)

        # Apply range filter
        data_to_process = data_to_process[
            (data_to_process['value'] > clinical_range[test][0]) &
            (data_to_process['value'] < clinical_range[test][1])
        ]
        
        # HISTOGRAM AFTER RANGE FILTER
        axes[i, 0].hist(data_to_process['value'], bins=bins, alpha=0.6, label='Range filtered', log=True)
        axes[i, 0].set_title(f"{test_names[test]} Histogram")
        axes[i, 0].legend()
        axes[i, 0].set_xlabel(units[test])

        after_range_len = len(data_to_process)

        # Time difference filtering
        #data_to_process['charttime'] = data_to_process.groupby('stayid')['charttime'].transform(lambda x: (x - x.min()))
        #data_to_process = data_to_process[data_to_process['charttime'] < time_window]
        after_time_len = len(data_to_process)

        # Rate of change filter
        data_to_process.sort_values(by=['stayid', 'charttime'], inplace=True)
        rate_filtering = data_to_process[['stayid', 'charttime', 'value']].copy()
        rate_filtering[['time_diff', 'value_diff']] = data_to_process.groupby('stayid')[['charttime', 'value']].diff()
        rate_filtering.dropna(inplace=True)
        rate_filtering['rate'] = rate_filtering['value_diff'] / rate_filtering['time_diff'].clip(lower=20/60)

        rco = clinical_range[test][2]
        stays_to_remove = rate_filtering[abs(rate_filtering['rate']) > rco]['stayid'].unique()
        data_to_process = data_to_process[~data_to_process['stayid'].isin(stays_to_remove)]

        after_rate_len = len(data_to_process)

        # Histogram of rate of change
        bins = np.linspace(rate_filtering['rate'].min(), rate_filtering['rate'].max(), 40)
        axes[i, 1].hist(rate_filtering['rate'], bins=bins, alpha=0.5, label='Original Rate', log=True)

        rate_filtered = rate_filtering[~rate_filtering['stayid'].isin(stays_to_remove)]
        axes[i, 1].hist(rate_filtered['rate'], bins=bins, alpha=0.5, label='Filtered Rate', log=True)
        axes[i, 1].set_title(f"{test_names[test]} Rate of Change")
        axes[i, 1].legend()
        axes[i, 1].set_xlabel(units[test] + "/h")

        # Histogram of number of measurements over time
        bins = np.arange(-24, 169, 2)
        axes[i, 2].hist(data_to_process['clock_time'], bins=bins)
        axes[i, 2].set_title(f"{test_names[test]} Measurements Over Time")
        axes[i, 2].set_xlabel("Hours")

        # Stats tracking
        removal_stats.append({
            'Test': test_names[test],
            'Removed by range': original_len - after_range_len,
            'Removed by rate': after_range_len - after_rate_len,
            'Remaining': after_rate_len
        })
        
        stats = data_to_process['value'].agg(['mean', 'std', 'min', 'max']).to_dict()
        stats['Test'] = test_names[test]
        summary_stats.append(stats)

        filtered_data = pd.concat([filtered_data, data_to_process], ignore_index=True)

    #plt.tight_layout()
    print('\n')
    plt.show()

    # Tables
    removal_df = pd.DataFrame(removal_stats)
    summary_df = pd.DataFrame(summary_stats)

    display(Markdown(removal_df.to_latex(index=False, float_format="%.2f")))
    display(Markdown(summary_df.to_latex(index=False, float_format="%.2f")))
    return filtered_data

starttime = time.time()
def print_time(event):
    #print(event + (f'\n time of running:{time.time() - starttime}\n'))
    return 0

from google.cloud import bigquery


# %% [markdown]
# # Preprocessing of the data
# All data are preprocessed with two filters. First filter removes values considered as errors. This filter is based on set on cutoffs (values out of reasonable physiological ranges). 
# Second employed filters removes values which change drastically in short time period.
# This is focused mostly as prevention of occurence of erroreus measurement which where inside previous bounderies but where retested because made not sense in clinical setting and where restested
# (eg. hemolysis). Whole dataline fot particular test for the patients is removed in that case. Filtering is done for indicidual database, however this does not affect the data set.
# We were also considering influance of rountine vs non-routine test and because the time in AmsterdamUMCdb is stored relativily to some arbitrary moment and not as real time we used correlation
# with time matrix to extimated the time of routine tests and set the time of first routine test as 0 (this corrected time is later refered as clock time).


# %%
#| echo: false
#| output: asis



cash_file = "cash_lab_stability.parquet"
cash_file_path = os.path.join(DATA_DIR, cash_file)


print(f"\n### Loading data from database\n")
# Loading Data
if os.path.exists(cash_file_path) and True:
    print(f"FIltered data loaded from file {cash_file}.")
    data = pd.read_parquet(cash_file_path)
else:
    raise Exception('Data file missing')




n_bootstrap = 500
report_every = 50  # print progress every 100 iterations
#results = pd.DataFrame(columns=['test', 'time_window', 'note', 'bootstrap_idx', 'model_type', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6'])
results = []

# 4. Bootstrap loop
for test, test_df in data.groupby('itemid'):
    print(f"Bootstrapping {test_names[test]} ...")
    window_starts = [0, 24, 72]
    window_lengths = [24, 48, 96]

    fit_lengths = [8, 16, 24] #h
    step = 1 #h
    number_of_bins = 1
    
    df_pairs = make_pairs(test_df, windows_start=window_starts, windows_length=window_lengths) # making pairs of all values
    dfs = make_windows(df_pairs, window_starts=window_starts, window_lengths=window_lengths) # making df with individual windows for processing
    
    model = pm(None, None)
        # Resample with replacement
        
    for df, ws, fl, wl in zip(dfs, window_starts, fit_lengths, window_lengths):
        df = recalculate_relative_to_first(df)
        df = df[df['time'] < fl]
        #df = df[df['difference'].abs() <= df['difference'].abs().quantile(0.99)]

        #making pairs

        #for binid in range(number_of_bins):
            #Filtering is based on previously set time, however later time is calculated with the set
            #time window
            #a_sub = a[((step*i) < a['time']) & (a['time'] <= (step*(i+1)))]


        # Make full data fit
        model.fit(df['time'], df['difference'], measurement_std=minimal_measurement_error[test], method='kinetic_jfskewt')

        params = list(model.params) + [np.nan] * (6 - len(model.params))  # pad to ensure 6 parameters

        
        #results = pd.concat([results, pd.DataFrame({
        results.append({
            'test': test,
            'time_window': f'{ws}-{ws+wl}h',
            'note': 'full',  # modify as needed
            'bootstrap_idx': None,
            'model_type': model.model_type,
            'p1': params[0],
            'p2': params[1],
            'p3': params[2],
            'p4': params[3],
            'p5': params[4],
            'p6': params[5],
            'p7': params[6],
            #'sdc': model.x_for_confidence_exceeds_abs(sdcs[test]),
            '1h': model.ci_max(1),
            '2h': model.ci_max(2),
            '3h': model.ci_max(3),
            '4h': model.ci_max(4),
            '5h': model.ci_max(5),
            '6h': model.ci_max(6),
        })

        for i in range(n_bootstrap):
        # Assuming m has attributes: model_type and params (tuple of p1...p6 or fewer)
            sample = df.sample(n=min(len(df), 100000), replace=True)
            model.fit(sample['time'], sample['difference'], measurement_std=minimal_measurement_error[test], method='kinetic_jfskewt')

            params = list(model.params) + [np.nan] * (6 - len(model.params))  # pad to ensure 6 parameters

            
            #results = pd.concat([results, pd.DataFrame({
            results.append({
                'test': test,
                'time_window': f'{ws}-{ws+wl}h',
                'note': '',  # modify as needed
                'bootstrap_idx': i,
                'model_type': model.model_type,
                'p1': params[0],
                'p2': params[1],
                'p3': params[2],
                'p4': params[3],
                'p5': params[4],
                'p6': params[5],
                'p7': params[6],
                #'sdc': model.x_for_confidence_exceeds_abs(sdcs[test]),
                '1h': model.ci_max(1),
                '2h': model.ci_max(2),
                '3h': model.ci_max(3),
                '4h': model.ci_max(4),
                '5h': model.ci_max(5),
                '6h': model.ci_max(6),
            })

            if False:
                plt.scatter(sample['time'], sample['difference'], marker='.', alpha=0.2)
                x_time = np.linspace(sample['time'].min(), sample['time'].max(), 200)
                lower_ci, upper_ci = model.ci(x_time)
                plt.fill_between(x_time, upper_ci, lower_ci,
                    alpha=0.2, edgecolor='#1E8925', facecolor='#6BD572')
                plt.show()
                

            #})], ignore_index=True)

            if (i + 1) % report_every == 0:
                print(f"[{test_names[test]}] time {ws} Bootstrap iteration {i + 1} / {n_bootstrap} completed., n={len(sample)}")

results = pd.DataFrame(results)


# 5. Summary reporting after bootstrapping

# Convert parameter columns to numeric (if they are not already)
param_cols = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', '1h', '4h']
results[param_cols] = results[param_cols].apply(pd.to_numeric, errors='coerce')

# Group by test and time_window and compute mean and std for each parameter
grouped = results.groupby(['test', 'time_window'])[param_cols]

# Loop over groups to print
for (test, time_window), group_df in grouped:
    print(f"\nSummary for test: {test_names[test]}, time window: {time_window}h")
    for param in param_cols:
        mean_val = group_df[param].mean()
        std_val = group_df[param].std()
        print(f"  {param}: mean = {mean_val:.4f}, std = {std_val:.4f}")

def get_nonexisting_filename(base_name, ext):
    counter = 0
    final_name = f"{base_name}.{ext}"
    while os.path.exists(final_name):
        counter += 1
        final_name = f"{base_name}_{counter}.{ext}"
    return final_name

output_filename = get_nonexisting_filename("bootstrap_results", "parquet")
results.to_parquet(output_filename)

print(f"Bootstrap results saved to {output_filename}")
