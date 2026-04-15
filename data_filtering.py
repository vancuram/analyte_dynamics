#%% [markdown]
# ---
# title: "How long can I trust laboratory results in critically ill patients: a large-scale  study on temporal validity of laboratory analytesi"
# subtitle: "Data extraction" 
# author:
#    - name: Martin Vančura
#    - name: Petr Waldauf
#    - name: Santje Slot
#    - name: Martijn Otten
#    - name: Ameet R. Jagesar
#    - name: Laurens A. Biesheuvel
#    - name: Martin Krbec
#    - name: Wendy P.J. den Elzen
#    - name: Pieter Roel Tuinman
#    - name: František Duška
#    - name: Paul Elbers
#    - name: Micah L. A. Heldeweg
#format:
#  pdf:
#    tbl-float: true
#    tbl-position: h!
#    include-in-header:
#      text: |
#        \usepackage{multirow}
#        \usepackage{float}
#---
# This supplementary file contains information about the data extraction and filtering process. For each database, individual processing and filtering were performed. Data from University Hospital Královské Vinohrady were stored in two separate versions of the MetaVision database (versions 5 and 6).
# Data from these databases were processed and filtered independently to identify possible systematic errors; however, the data were subsequently merged for further processing.  


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
import pickle
import statsmodels.api as sm
from IPython.display import Markdown, Latex, display
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import zscore, shapiro, normaltest, probplot, uniform, t, norm, trimboth, skewnorm, jf_skew_t, gaussian_kde
from skewt_scipy.skewt import skewt
from scipy.special import gammaln
from properscoring import crps_ensemble
from matplotlib.colors import LinearSegmentedColormap
import sys
from tableone import TableOne
sys.path.insert(0, os.path.expanduser('~/icu_database/lib'))
from probabilitymodel import probabilitymodel as ProbabilityModel

#from datashader import plotting
##########3 Settings ###############33


#print(f"""\n### Settings\n""")
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


#test_original_conversions = {3024561: 50862, 3003458: 50970}
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
#maximal_measurement_error = {3024561: 1, 3003458: 0.1, 3006140: 5, 3012095: 0.1, 3032080: 0.1, 43534077: 0.2, 40762351: 0.5, 3020564: 5, 3010813: 0.5, 3007461: 5, 3035995: 5, 3015377: 0.2, 3000285: 2, 3020460: 2, 42869588: 2}
minimal_measurement_error = {3024561: 0.3, 3003458: 0.02, 3006140: 0.1, 3012095: 0.02, 3032080: 0.025, 43534077: 0.05, 40762351: 0.2, 3020564: 2, 3010813: 0.1, 3007461: 1, 3035995: 3, 3015377: 0.02, 3000285: 0.3, 3020460: 1, 42869588: 1}
test_cutoffs = {3024561: 5, 3003458: 0.5, 3006140: 5, 3012095: 0.5, 3032080: 0.5, 43534077: 2, 40762351: 1, 3020564: 20, 3010813: 1, 3007461: 20, 3035995: 10, 3015377: 0.5, 3000285: 3, 3020460: 5, 42869588: 3}
list_of_tests = sorted(list_of_tests, key=lambda x: test_names[x].lower())

ugly_counter = 0

def plt_to_markdown(pltx, name=None, label=None, reference=None):
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
    if not reference is None:
        out_str += "\ref{" + reference + """}
        """

    out_str += """\end{figure}
    """
    if reference is None:
        reference = label
    out_str = f"""![{label}]({name_adres} "{reference}")"""
    #print(out_str)
    #print(out_str)
    pltx.savefig(name_adres, format='png')
    pltx.close()


    display(Markdown(out_str))
    return 0
 
def recalculate_relative_to_first(df):
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

        
def make_fit_and_density_plot(df1, df2=None, l1='less severe', l2='more severe', title='Some plot', unit='',
               window_starts=(0, 24, 72), window_length=(24, 48, 96), plot_lengths=[12, 12, 24], number_of_bins=12, morning_tolerance = 1.5,
               times=['charttime', 'charttime', 'charttime'], minimal_set_size=10, measurement_std=None,
                 minimal_count_int=3, fit_minimal_count=10, cutoff=None, titles=['Early', 'Intermediate', 'Late']):
# minimal_count_int is interval for fitting model (preventing fitting if there is not enough

    if df2 is None:
        second = False
    else:
        second = True

    rows = len(window_starts)
    # setting plot
    fig, axes = plt.subplots(len(window_starts), 1, figsize=(12, rows * 5), sharex=False, sharey="all", layout="constrained")  # Adjust figure size
    if rows == 1: # beacouse of just on rows return only one axe withou indexing option
        axes = [axes]
    #plt.tight_layout()
    #    print('Df1 is:')
    #   print(df1)

    models_a = []
    models_b = []
    max_range = 0


    # All data 
    for indx, start, plot_length in zip(range(len(window_starts)), window_starts, plot_lengths):

        # filtering based on set time
        a = df1[(df1['charttime'] >= start)
                & (df1['charttime'] < start + window_length[indx])].copy()
            
        a = recalculate_relative_to_first(a)
        a = a[a['time'] < plot_length]


        if max_range < a['difference'].abs().quantile(0.99):
            max_range = a['difference'].abs().quantile(0.99)


        #axes[indx].scatter(a['time'] + t, a['difference'], c='g', label=l1, alpha=a_alpha, s=2)

        ## SECOND dataframe
        if second:
            if len(b[b['time'] < minimal_count_int]) > fit_minimal_count:
                model_b = ProbabilityModel(b['time'], b['difference'])
                model_b.fit(measurement_std=measurement_std, method='kinetic_jfskewt')
                #prediction
                tf = np.linspace(start, start + plot_length, 200)
                tf_zerod = tf - start
                m_t,  sigmal, sigmau = model_b.predict(tf_zerod) 
                #axes[indx].scatter(b['time'] + t, b['difference'], c='r', label=l1, alpha=b_alpha, s=2)
                axes[indx].fill_between(tf_zerod,sigmau, sigmal,
                alpha=0.2, facecolor='#FF6666')
                #alpha=0.2, edgecolor='#CC0000', facecolor='#FF6666')
                axes[indx].plot(tf_zerod, m_t, c= '#CC0000', alpha=0.3, label=model_b.model_equation())
            else:
                model_b = None

            models_b.append(model_b)


        # Making scatter 




        # Mean and deviation plot
        axes[indx].text(0.02, 0.98, titles[indx] + f', n={len(a)}' , transform=axes[indx].transAxes,
            ha='left', va='top', fontsize=12)

        sub = a.sample(min(1000, len(a)))

        # Fit KDE on 2D subsample
        kde = gaussian_kde([sub["time"], sub["difference"]])

        # Evaluate on full data
        densities = kde([a["time"], a["difference"]])
        a["density"] = densities

        #blue_violet = LinearSegmentedColormap.from_list("blue_violet", ["darkblue", "magenta"])
        blue_violet = LinearSegmentedColormap.from_list("blue_violet", ["green", "green", "darkorange"])
        #axes[indx].scatter(a['time'], a['difference'], color='g', marker='.', s=25, label=l1, alpha=max(0.05, min(1, 1000/len(a))))
        axes[indx].scatter(a['time'], a['difference'], c=a['density'], cmap=blue_violet, marker='.', s=15, alpha=max(0.1, min(0.3, 10000/len(a))))


        # initialization and fitting
        if len(a[a['time'] < minimal_count_int]) > fit_minimal_count:
            model_a = ProbabilityModel(a['time'], a['difference'])
            model_a.fit(measurement_std=measurement_std, method='kinetic_jfskewt')
            #prediction
            tf = np.linspace(start, start + plot_length, 200)
            tf_zerod = tf - start
            m_t,  sigmal, sigmau = model_a.predict(tf_zerod) 
            breakpoint()

            #axes[indx].fill_between(tf, m_t  + sigma_t, m_t - sigma_t,
            #alpha=0.2, ls='--', edgecolor='#1E8925', facecolor='#6BD572')

            axes[indx].fill_between(tf_zerod, sigmau, sigmal,
            #alpha=0.15, edgecolor='k', facecolor='k')
            #alpha=0.2, edgecolor='#1E8925', facecolor='#6BD572')
            alpha=0.2, edgecolor='orangered', facecolor='#6BD572')
            #axes[indx].plot(tf_zerod, m_t, c= 'white', ls='-', alpha=0.8, label=model_a.model_equation(), linewidth=2)
            axes[indx].plot(tf_zerod, m_t, c= 'orangered', ls='--', alpha=0.7, label=model_a.model_equation(), linewidth=3)

            #m_t, delta_t = model_a.predict(a['time'])
            #outliers = a[(a['difference'] > m_t+delta_t) | (a['difference'] < m_t - delta_t)]
            #plt.scatter(outliers['time'] + start, outliers['difference'], marker='.', s=1, alpha=0.1)
        else:
            model_a = None
        models_a.append(model_a)

        if cutoff != None:
            axes[indx].axhline(cutoff, c='grey', ls='--')
            axes[indx].axhline(-cutoff, c='grey', ls='--')


        axes[indx].legend(fontsize='small', loc="lower left")
        axes[indx].set_xlabel(f'Time (h)')
        #axes[indx].set_xlabel(f'{times[indx]} (h)')
        axes[indx].grid(linestyle='--', color='lightgrey')


    plt.setp(axes, ylim=(-max_range*1.05, max_range*1.05))

    #plt.xlabel('time [h]')
    plt.ylabel(units[test])
    #plt.legend(fontsize='small')

    plt.suptitle(title)


    #plt.tight_layout()
    #plt.show()
    #plt.close()
    plt_to_markdown(plt)
    if second:
        return models_a, models_b
    else:
        return models_a


def initial_filtering(df, report=True, db=''):
    df = df.copy()



    if pd.api.types.is_datetime64_any_dtype(df['charttime']):
        df['charttime'] = df.groupby('stayid')['charttime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)
    else:
        df['charttime'] = df.groupby('stayid')['charttime'].transform(lambda x: (x - x.min()))

    df = df[df['charttime'] < 168]
    df = df[df['itemid'].isin(list_of_tests)]
    df = df.drop_duplicates()


    patient_summary = [{'phase': 'Extracted', 'Stays': len(df['stayid'].unique()), 'Test': len(df)}]
    
    # Filtering patiens without any pair of test within 24 hours + 2h tolarence
    df_ids = df.sort_values(by=['stayid', 'charttime'])
    df_ids['time_diff'] = df.groupby(['stayid', 'itemid'])[['charttime']].diff()
    df_ids.dropna(inplace=True)
    df_ids = df_ids[df_ids['time_diff'] < 26]
    df = df[df['stayid'].isin(df_ids['stayid'])]

    patient_summary.append({'phase': 'Has enough data', 'Stays': len(df['stayid'].unique()), 'Test': len(df)})

    df['morning_time'] = df.groupby('stayid')['charttime'].transform(get_morning_time) 
    df['clock_time'] = df['charttime'] - df['morning_time']
    #print(df[df['crrt'] == True])
    #print(df['crrt'])

    bins = range(-24, 169, 2)
    df['charttime'].hist(bins=bins, alpha=1.0, label='Time from 1. measurement')
    df['clock_time'].hist(bins=bins, alpha=0.8, label='Correlation adjusted time')
    if report:
        plt.title('Correlation time adjustment')
        plt.xlabel('Hours')
        plt.ylabel('Number of tests')
        plt.legend()
        plt_to_markdown(plt, label=f'Comparison of distribution of original test times after admission and correlation adjusted times in {db}', reference=f'fig:{db}_time_adjustment')
    else:
        plt.close()


    removal_stats = []
    summary_stats = []

    # Combined plot
    #fig, axes = plt.subplots(len(list_of_tests), 3, figsize=(20, 4 * len(list_of_tests)), layout="constrained")

    filtered_data = pd.DataFrame()

    # Split into two halves
    midpoint = math.ceil(len(list_of_tests) / 3)
    test_groups = [list_of_tests[:midpoint], list_of_tests[midpoint:midpoint*2], list_of_tests[midpoint*2:]]

    for group_index, test_group in enumerate(test_groups, start=1):
        fig, axes = plt.subplots(len(test_group), 3, figsize=(12, 3 * len(test_group)), layout="constrained")

        # Ensure axes is always 2D array for consistent indexing
        if len(test_group) == 1:
            axes = np.array([axes])

        for i, test in enumerate(test_group):
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
            axes[i, 0].hist(data_to_process['value'], bins=bins, alpha=0.6, label='Filtered', log=True)
            axes[i, 0].set_title(f"{test_names[test]} histogram of values")
            axes[i, 0].legend()
            axes[i, 0].set_xlabel(units[test])

            after_range_len = len(data_to_process)

            # Rate of change filter
            data_to_process.sort_values(by=['stayid', 'charttime'], inplace=True)
            rate_filtering = data_to_process[['stayid', 'charttime', 'value']].copy()
            rate_filtering[['time_diff', 'value_diff']] = data_to_process.groupby('stayid')[['charttime', 'value']].diff()
            rate_filtering = rate_filtering[rate_filtering['time_diff'] < 3]
            rate_filtering.dropna(inplace=True)
            rate_filtering['rate'] = rate_filtering['value_diff'] / rate_filtering['time_diff'].clip(lower=30/60)

            #rco = clinical_range[test][2]

            rate_z = zscore(rate_filtering['rate'], nan_policy='omit')

            # Filter rows where |z| < 4
            stays_to_remove = rate_filtering[abs(rate_z) >4 ]['stayid'].unique()
            data_to_process = data_to_process[~data_to_process['stayid'].isin(stays_to_remove)]
            
            # absolute value
            #rco = clinical_range[test][2]
            #stays_to_remove = rate_filtering[abs(rate_filtering['rate']) > rco]['stayid'].unique()
            #data_to_process = data_to_process[~data_to_process['stayid'].isin(stays_to_remove)]

            after_rate_len = len(data_to_process)

            # Histogram of rate of change
            bins = np.linspace(rate_filtering['rate'].min(), rate_filtering['rate'].max(), 40)
            axes[i, 1].hist(rate_filtering['rate'], bins=bins, alpha=0.5, label='Original', log=True)
            rate_filtered = rate_filtering[~rate_filtering['stayid'].isin(stays_to_remove)]
            axes[i, 1].hist(rate_filtered['rate'], bins=bins, alpha=0.5, label='Filtered', log=True)
            axes[i, 1].set_title(f"{test_names[test]} rate of change histogram")
            axes[i, 1].legend()
            axes[i, 1].set_xlabel(units[test] + "/h")

            # Histogram of number of measurements over time
            bins = np.arange(-24, 169, 2)
            axes[i, 2].hist(data_to_process['clock_time'], bins=bins)
            axes[i, 2].set_title(f"{test_names[test]} measurements time distribution")
            axes[i, 2].set_xlabel("Hours")

            # Stats tracking
            removal_stats.append({
                'Test': test_names[test],
                'Removed by range': original_len - after_range_len,
                'Removed by rate': after_range_len - after_rate_len,
                'Remaining': after_rate_len
            })
            stats= pd.DataFrame()
            stats = {'Test' : f"{test_names[test]}, {units[test]}"}
            stats.update(data_to_process['value'].agg(['mean', 'std', 'min', 'max']).to_dict())
            summary_stats.append(stats)

            filtered_data = pd.concat([filtered_data, data_to_process], ignore_index=True)

        #print(f"\nShowing Figure {group_index} of 2")
        print('\n')
        if report:
            plt_to_markdown(plt, label=f'Visualization of filtered data in {db}. Histograms of values and rates are logarithmic. Measurement time distribution plots show histogram of measurement times after adjusting to routine time measurements. Part {group_index} of 3', reference=f'fig-{db}-filtering-{group_index}')
        else:
            plt.close()

    patient_summary.append({'': 'Filtered', 'Stays': len(filtered_data['stayid'].unique()), 'Test': len(filtered_data)})

    # Tables
    if report:
        removal_df = pd.DataFrame(removal_stats)
        summary_df = pd.DataFrame(summary_stats)
        patient_df = pd.DataFrame(patient_summary)

        #print("\n### Removed values\n")
        display(Markdown(removal_df.to_latex(index=False, float_format="%.2f", caption=f'Removed values in database {db}', position='H')))
        #print("\n### Statistics of filtered values")
        display(Markdown(summary_df.to_latex(index=False, float_format="%.2f", caption=f'Statistics of filtered values in database {db}', position='H')))
    #print(\n 
    #display(Markdown(patient_df.to_latex(index=False, float_format="%.2f")))

    return filtered_data

starttime = time.time()
def print_time(event):
    #print(event + (f'\n time of running:{time.time() - starttime}\n'))
    return 0

from google.cloud import bigquery


# Set up environment variables
project_id = ''
if project_id == 'CHANGE-ME':
  raise ValueError('You must change project_id to your GCP project.')
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id 

dataset_project_id = 'amsterdamumcdb' #@param {type:"string"}
dataset_id = 'version1_5_0' #@param {type:"string"}
location = 'eu' #@param {type:"string"}
store_file_ams = "ams_lab_test_vol.parquet"
file_path_ams = os.path.join(DATA_DIR, store_file_ams)


# Read data from BigQuery into pandas dataframes.
def run_query(query, project_id=project_id):
  return pd.io.gbq.read_gbq(
      query,
      project_id=project_id,
      dialect='standard')

# set the dataset
# if you want to use the demo, change this to mimic_demo


# Amsterdam
#print(f"\n### Loading data from database\n")
# Loading Data
if os.path.exists(file_path_ams):
#    print(f"Table data loaded from file {store_file_ams}.")
    lab_results_ams = pd.read_parquet(file_path_ams)
else:
    print(f"file with data from measurement amsterdam UMC not found! Fetching from the database...")

    with open("stability_extraction_amsterdam.sql", "r") as file:
    	sql_query = file.read()

    lab_results_ams = run_query(sql_query)

    lab_results_ams['charttime'] = pd.to_datetime(lab_results_ams['charttime'])
    lab_results_ams['value'] = pd.to_numeric(lab_results_ams['value'], errors='coerce')
    lab_results_ams = lab_results_ams.dropna()
    #lab_results['itemid'] = lab_results['itemid'].apply(lambda x: test_original_conversions[x])
    lab_results_ams.to_parquet(file_path_ams, index=False)  # Save for future use
    #print(f" Saved data to {file_path_ams}\n")


#print(f"Number of lab table row before processing {len(lab_results_ams)}")

# Hirid
#print(f"\n### Loading data from database\n")

store_file_hirid = "hirid_lab_test_vol.parquet"
file_path_hirid = os.path.join(DATA_DIR, store_file_hirid)
# Loading Data
if os.path.exists(file_path_hirid):
#    print(f"Table data loaded from file {store_file_hirid}.")
    lab_results_hirid = pd.read_parquet(file_path_hirid)
else:
    print(f"file with data from measurement hiridUMC not found! Fetching from the database...")

    with open("stability_extraction_hirid.sql", "r") as file:
    	sql_query = file.read()

    lab_results_hirid = run_query(sql_query)

    lab_results_hirid['charttime'] = pd.to_datetime(lab_results_hirid['charttime'])
    lab_results_hirid['value'] = pd.to_numeric(lab_results_hirid['value'], errors='coerce')
    lab_results_hirid = lab_results_hirid.dropna()
    #lab_results['itemid'] = lab_results['itemid'].apply(lambda x: test_original_conversions[x])
    lab_results_hirid.to_parquet(file_path_hirid, index=False)  # Save for future use
    #print(f" Saved data to {file_path_hirid}\n")


#print(f"Number of lab table row before processing {len(lab_results_hirid)}")


# MIMIC_________________
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""
DB_HOST = ""
DB_PORT = ""  # Default PostgreSQL port
engine = create_engine(f'postgresql:///{DB_NAME}')

store_file_mimic = "mimic_lab_test_vol.parquet"
file_path_mimic = os.path.join(DATA_DIR, store_file_mimic)


#print(f"\n### Loading data from database\n")
# Loading Data
if os.path.exists(file_path_mimic):
 #   print(f"Table data loaded from file {store_file_mimic}.")
    lab_results_mimic = pd.read_parquet(file_path_mimic)
else:
    print(f"file with data from measurement MIMIC not found! Fetching from the database...")

    with open("stability_extraction_mimic.sql", "r") as file:
    	sql_query_mimic = file.read()

    lab_results_mimic = pd.read_sql(sql_query_mimic, con=engine)

    lab_results_mimic['charttime'] = pd.to_datetime(lab_results_mimic['charttime'])
    lab_results_mimic['value'] = pd.to_numeric(lab_results_mimic['value'], errors='coerce')
    lab_results_mimic = lab_results_mimic.dropna()
    #lab_results['itemid'] = lab_results['itemid'].apply(lambda x: test_original_conversions[x])
    lab_results_mimic.to_parquet(file_path_mimic, index=False)  # Save for future use
    #print(f" Saved data to {file_path_mimic}\n")


#print(f"Number of lab table row before processing {len(lab_results_mimic)}")


# Prague ICU MV 5_________________


store_file_mv5 = "MV5_lab_stability.csv"
file_path_mv5 = os.path.join(DATA_DIR, store_file_mv5)


#print(f"\nLoading data from {store_file_mv5}\n.")


lab_results_mv5= pd.read_csv(file_path_mv5)

# datime conversion
#lab_results_mv5['charttime'] = pd.to_datetime(lab_results_mv5['charttime'])
# conv to numeric
lab_results_mv5['value'] = pd.to_numeric(lab_results_mv5['value'], errors='coerce')
# removing nulls from covariets


# dropping lines with missing informations
lab_results_mv5 = pd.read_csv(file_path_mv5, dtype={
    'fluid_volume_within_24h': 'float64',
    'other_decimal_column': 'float64'
})
lab_results_mv5 = lab_results_mv5.apply(pd.to_numeric, errors='coerce')

#print(f"Number of lab table row before processing {len(lab_results_mv5)}")

# Prague ICU MV 6_________________

conversions_mv = {3035995: 59.99, 40762351: 0.1}

store_file_mv6 = "MV6_lab_stability.csv"
file_path_mv6 = os.path.join(DATA_DIR, store_file_mv6)


#print(f"\nLoading data from {store_file_mv6}\n.")


lab_results_mv6= pd.read_csv(file_path_mv6)
lab_results_mv6 = lab_results_mv6.apply(pd.to_numeric, errors='coerce')

# datime conversion
#lab_results_mv6['charttime'] = pd.to_datetime(lab_results_mv6['charttime'])
# conv to numeric
lab_results_mv6['value'] = pd.to_numeric(lab_results_mv6['value'], errors='coerce')

# dropping lines with missing informations
lab_results_mv6= lab_results_mv6.dropna()

#print(f"Number of lab table row before processing {len(lab_results_mv6)}")

# %% [markdown]
# # Preprocessing of the data
# ## Filtering erroneous measurements
# All data were preprocessed using two filters. The first filter removed values considered erroneous based on predefined cutoffs (see @tbl-cutoffs), defined as values outside reasonable physiological ranges.
# The second filter removed values that changed drastically over a short time period. This filter was primarily intended to prevent inclusion of erroneous measurements that fell within physiological bounds but were clinically implausible and subsequently retested (e.g., due to hemolysis).
# For this purpose, the rate of change between consecutive measurements was calculated, and z-score filtering with a cutoff of 4 was applied.
# If this threshold was exceeded, the entire data line for the given test and patient was removed. Rate of change $r$ was calculated as follows:
# $$ r = \frac{x_1 - x_0} {max(t_1-t_0, 30 min)}, $$
# where  $x_0$   and  $x_1$  are the measured values and $t_0$ and $t_1$ are the corresponding measurement times. The time difference used in the calculation was set to a minimum of 30 minutes to prevent unrealistically large rates resulting from very short time intervals

# ## Estimating time of routine measurements
# We also considered the influence of routine versus non-routine testing. Because time in the AmsterdamUMCdb is stored relative to an arbitrary reference point rather than as absolute time, and because the exact timing of routine testing is unknown and may vary across ICUs, we estimated routine testing times using correlation with a time mask.  
# This procedure was performed separately for each patient. We assumed that routine testing occurs regularly with a 24-hour period. For each patient, test times were converted into a vector of length 73, where each element represented the number of tests performed within a specific one-hour interval since admission (e.g., the first element corresponds to 0–1 hours after admission, the second to 1–2 hours, etc.). For example, a patient with seven tests at 0.5 hours and five tests at 3.2 hours after admission would have the following vector:
# $$ [7, 0, 0, 5, 0, ..., 0] $$
# This vector was then correlated with a 24-hour mask (a vector of length 50 with values of one at positions 0, 1, 23, 24, 25, 47, 48, and 49, and zeros elsewhere). The position with the highest correlation was selected as the estimated time of routine testing, and all time values were shifted such that this time corresponded to zero.

# ## Reporting of data preprocessing
# The filtering process for each individual database is reported in a separate section. For each database, plots showing the temporal distribution of tests were generated. Histograms of test times were created with times shifted so that the first test was set to zero (time since first measurement) and were compared with histograms in which times were corrected such that the first routine measurement was set to zero.  
# For each individual analyte in every database, logarithmic histograms were generated to illustrate the effect of data filtering. Histograms of the time distribution using correlation-adjusted times were also created for each test.  
# Tables summarizing the number of tests removed by each filter for each analyte, basic statistical characteristics of the analytes, and the number of admissions are also included.


# %%
#| echo: false
#| output: asis

#print('### Clinical cutoffs used for filtering:\n')

ranges_to_print = []
for t in list_of_tests:
    ranges_to_print.append({'Test': test_names[t], 'Units': units[t], 'Low': clinical_range[t][0], 
                           'High': clinical_range[t][1]})#, 'Rate': clinical_range[t][2]})
display(Latex(pd.DataFrame(ranges_to_print).to_latex(index=False, float_format="%.2f", caption="Clinical cutoffs used for filtering", label='tbl-cutoffs', position='h!')))


cash_file = "cash_lab_stability.parquet"
cash_file_path = os.path.join(DATA_DIR, cash_file)


#print(f"\n### Loading data from database\n")
# Loading Data
if os.path.exists(cash_file_path) and False:
    print(f"Filtered data loaded from file {cash_file}.")
    filtered_data = pd.read_parquet(cash_file_path)
else:

    print('## AmsterdamUMCdb\n')
    filtered_data_ams = initial_filtering(lab_results_ams, db='AmsterdamUMCdb')

    print('## HiRID\n')
    filtered_data_hirid= initial_filtering(lab_results_hirid, db='HiRID')

    print('## MIMIC-IV\n')
    test = 40762351
    filtered_data_mimic_ego_included = initial_filtering(lab_results_mimic, report=False)
    filtered_data_mimic = initial_filtering(lab_results_mimic[~(lab_results_mimic['icu_unit'] == 'Cardiac Vascular Intensive Care Unit (CVICU)')], db='MIMIC-IV')

    print('\n### Cardiosurgery patients in MIMIC-IV\n')
    print("""
We identified a unique pattern in the MIMIC-IV database with respect to hemoglobin laboratory results. Several patients demonstrated increases of 2–4 g/dL within the first two hours after ICU admission (see plots below). Upon further investigation, we found that these cases were primarily associated with the Cardiac Vascular ICU (CVICU). Some of these patients had documented administration of blood products or use of a cell saver device; however, others had no corresponding records of transfusion or related interventions. We hypothesize that these individuals were postoperative cardiac surgery patients admitted directly to the ICU. In many cases, the first hemoglobin measurement may have been obtained intraoperatively or immediately at the end of surgery, followed by a second measurement after aggressive correction with blood products.

Importantly, some patients with hemoglobin changes greater than 3 g/dL lacked any record of transfusion, cell saver use, or other relevant procedures. This suggests the presence of systematic bias related to the transition from the operating room to the ICU and possibly missing data in the database. As a result, we could not reliably use blood product administration as a confounder to stratify patients by therapy received. We therefore considered this phenomenon to reflect a database artifact rather than a genuine clinical characteristic, and decided to exclude all patients from this particular ICU to avoid introducing systematic error.

Finally, we note that in some cases there were no clear markers to identify these patients as cardiac surgery cases apart from their admission to the CVICU—a designation that may itself be somewhat misleading.
          """)

    make_fit_and_density_plot(filtered_data_mimic_ego_included[filtered_data_mimic_ego_included['itemid'] == 40762351],
                              title='Hemoglobin data with cardiosurgery patients', window_starts=[0],
                              measurement_std=minimal_measurement_error[40762351]
                             )

    make_fit_and_density_plot(filtered_data_mimic[filtered_data_mimic['itemid'] == 40762351],
                              title='Hemoglobin data without cardiosurgery patients', window_starts=[0],
                              measurement_std=minimal_measurement_error[40762351])



    print('\n')


    print('## UNKV Metavision 5\n')
    filtered_data_mv5 = initial_filtering(lab_results_mv5, db='UNKV Metavision 5')

    print('## UNKV Metavison 6\n')
    filtered_data_mv6 = initial_filtering(lab_results_mv6, db='UNKV Metavison 6')

    # merging
    filtered_data_ams['stayid'] = 'ams_' + filtered_data_ams['stayid'].astype(str)
    filtered_data_hirid['stayid'] = 'hirid_' + filtered_data_hirid['stayid'].astype(str)
    filtered_data_mimic['stayid'] = 'mimic_' + filtered_data_mimic['stayid'].astype(str)
    filtered_data_mv5['stayid'] = 'mv5_' + filtered_data_mv5['stayid'].astype(str)
    filtered_data_mv6['stayid'] = 'mv6_' + filtered_data_mv6['stayid'].astype(str)
    
    numeric_cols = ['fluid_volume_within_24h', 'albumin_dose_within_24h', 'blood_cells_volume', 'plasma_volume', 'platelets_volume', 'crrt', 'max_crp']
    filtered_data = pd.concat([filtered_data_mimic, filtered_data_hirid, filtered_data_ams, filtered_data_mv5, filtered_data_mv6], ignore_index=True)
    filtered_data[numeric_cols] = filtered_data[numeric_cols].apply(pd.to_numeric)

    #breakpoint()
    filtered_data.to_parquet(cash_file_path, index=False)  # Save for future use

    #print(f" Saved data to {cash_file_path}\n")


#%% [markdown]
# # Methods of measurements
# Data used in this study are collected from multiple medical centers acquired over approximatly 15 years in open databases with
# limited information on devices and analythical methods used. However, most of the variation of the test result is likely atributed to physiological
# change and preanalytical error rather then analytical process itself. Further, the results compaired within one patients are highly likely compared on the
# same device and method, so this should compansate possible systematics errors, ofsets etc.  
# We provided information on methods and devices used in period of data acquisition from Amsterdam and Prague databases in @tbl-prague-methods and @tbl-ams-methods.
# We did not obtain these information for MIMIC-IV and HiRID.
# 

# %%
#| echo: false
#| output: asis
methods_list = pd.read_csv('methods.csv') 
latex_table = methods_list.to_latex(index=False, label="tbl-prague-methods", caption= 'Laboratory devices and methods used in Prague UNKV')
latex_table = latex_table.replace("\\begin{tabular}", "\\small\n\\begin{tabular}")
display(Markdown(latex_table))

methods_list = pd.read_csv('methods_ams.csv')
latex_table = methods_list.to_latex(index=False, label="tbl-ams-methods", caption= 'Laboratory devices and methods used in AmsterdamUMCdb')
latex_table = latex_table.replace("\\begin{tabular}", "\\small\n\\begin{tabular}")
display(Markdown(latex_table))

# [markdown]
# # Population characteristics
# This parts focuses on obtaining data about population. We worked with all individual admission and decided to describe
# those.

# %%
#| echo: false
#| output: asis


# AMS POPULATION


def approx_value(interval_str):
    if pd.isna(interval_str) or interval_str.strip() == '':
        return np.nan

    interval_str = interval_str.strip()

    # Handle "80+" (open-ended upper bound)
    if interval_str.endswith('+'):
        min_val = float(interval_str[:-1])
        # You can define midpoint as min_val + half of a typical range (say 5)
        return min_val + 5

    # Handle "159-" (open-ended lower bound)
    if interval_str.endswith('-'):
        max_val = float(interval_str[:-1])
        # Midpoint as max_val - 5 (or any reasonable estimate)
        return max_val - 5

    # Handle ranges like "60-69"
    if '-' in interval_str:
        parts = interval_str.split('-')
        try:
            min_val = float(parts[0])
            max_val = float(parts[1])
            return (min_val + max_val) / 2
        except:
            return np.nan

    # If just a number (unlikely), convert directly
    try:
        return float(interval_str)
    except:
        return np.nan
# loading second file with patient characteristec
store_file_pop_ams = "ams_population.parquet"
file_path = os.path.join(DATA_DIR, store_file_pop_ams)

if os.path.exists(file_path):
    #print(rf"Table data loaded from file {store_file_pop_ams}.")
    population_ams = pd.read_parquet(file_path)
else:
    #print(rf"file with data for amsterdam UMC population not found! Fetching from the database...")
    with open("population_ams.sql", "r") as file:
    	sql_query = file.read()
# Run the query
    population_ams = run_query(sql_query)
    # List of lab columns 
    # Convert them to float
    population_ams['stayid'] = 'ams_' + population_ams['stayid'].astype(str)
    population_ams.to_parquet(file_path, index=False)  # Save for future use
    #print(f" Saved data to {file_path}\n")
#print(lab_results.notna().sum())


#select only those patients used in previous analysis
population_ams = population_ams[population_ams['stayid'].isin(filtered_data['stayid'].unique())]

# Conversion of speciality from Ducth to clustered groups
specialty_conversion = {
    'Cardiology': ['Cardiologie'],
    'Neurosurgery': ['Neurochirurgie'],
    'Cardiac and Vascular Surgery': ['Cardiochirurgie', 'Vaatchirurgie'],
    'Internal Medicine': ['Inwendig', 'Oncologie Inwendig', 'Hematologie', 'Reumatologie'],
    'General and Oncological Surgery': [
        'Heelkunde Gastro-enterologie',
        'Heelkunde Oncologie',
        'Heelkunde Longen/Oncologie',
        'Heelkunde'
    ],
    'Pulmonology': ['Longziekte'],
    'Trauma Surgery': ['Traumatologie', 'Orthopedie'],
    'Neurology': ['Neurologie'],
    'ICU': ['Intensive Care Volwassenen'],
    'ENT and Dental': ['Keel, Neus & Oorarts', 'Mondheelkunde'],
    'Urology and Nephrology': ['Urologie', 'Nefrologie'],
    'Gastroenterology and Hepatology': ['Maag-,Darm-,Leverziekten'],
    'Gynecology and Obstetrics': ['Gynaecologie', 'Verloskunde', 'Obstetrie'],
    'Plastic Surgery': ['Plastische chirurgie'],
    'Ophthalmology': ['Oogheelkunde'],
    'Unspecified/Other': ['None', 'ders']
}


flat_mapping = {specialty: group for group, specialties in specialty_conversion.items() for specialty in specialties}

# Map the specialties in the DataFrame
population_ams['specialty group'] = population_ams['specialty'].map(flat_mapping)

# Optional: Fill unmatched values with 'Unspecified/Other'
population_ams['specialty group'] = population_ams['specialty group'].fillna('Unspecified/Other')
population_ams = population_ams.rename(columns={"died_within_2w": 'death'})


# making conversion of ranges to approx values
def approx_value(interval_str):
    if pd.isna(interval_str) or interval_str.strip() == '':
        return np.nan

    interval_str = interval_str.strip()

    # Handle "80+" (open-ended upper bound)
    if interval_str.endswith('+'):
        min_val = float(interval_str[:-1])
        # You can define midpoint as min_val + half of a typical range (say 5)
        return min_val + 5

    # Handle "159-" (open-ended lower bound)
    if interval_str.endswith('-'):
        max_val = float(interval_str[:-1])
        # Midpoint as max_val - 5 (or any reasonable estimate)
        return max_val - 5

    # Handle ranges like "60-69"
    if '-' in interval_str:
        parts = interval_str.split('-')
        try:
            min_val = float(parts[0])
            max_val = float(parts[1])
            return (min_val + max_val) / 2
        except:
            return np.nan

    # If just a number (unlikely), convert directly
    try:
        return float(interval_str)
    except:
        return np.nan


#population_description["Number of ICU admissions after filtering"] = population['stayid'].nunique()


#print('\n')
#for x, y in population_description.items(): print(f'**{x}**: {y}\n') 

population_ams['age'] = population_ams['agegroup'].apply(approx_value)
population_ams['height'] = population_ams['heightgroup'].apply(approx_value)
population_ams['weight'] = population_ams['weightgroup'].apply(approx_value)
population_ams['bmi'] = population_ams['weight'] / (population_ams['height']/100)**2

columns = ['sex', 'age', 'height', 'weight', 'bmi',  'specialty group', 'death']
categorical = ['sex', 'specialty group', 'death']



# HiRID

store_file_pop_hirid = "hirid_population.parquet"
file_path = os.path.join(DATA_DIR, store_file_pop_hirid)

if os.path.exists(file_path):
    #print(rf"Table data loaded from file {store_file_pop_hirid}.")
    population_hirid = pd.read_parquet(file_path)
else:
    #print(rf"file with data for hiridterdam UMC population not found! Fetching from the database...")
    with open("population_hirid.sql", "r") as file:
    	sql_query = file.read()
# Run the query
    population_hirid = run_query(sql_query)
    # List of lab columns 
    # Convert them to float
    population_hirid['stayid'] = 'hirid_' + population_hirid['stayid'].astype(str)
    population_hirid.to_parquet(file_path, index=False)  # Save for future use
    #print(f" Saved data to {file_path}\n")
#print(lab_results.notna().sum())


#select only those patients used in previous analysis
population_hirid = population_hirid[population_hirid['stayid'].isin(filtered_data['stayid'].unique())]
population_hirid['bmi'] = population_hirid['weight'] / (population_hirid['height']/100)**2

# MIMIC

store_file_pop_mimic = "mimic_population.parquet"
file_path = os.path.join(DATA_DIR, store_file_pop_mimic)

if os.path.exists(file_path):
    #print(rf"Table data loaded from file {store_file_pop_mimic}.")
    population_mimic = pd.read_parquet(file_path)
else:
    print(rf"file with data for MIMIC population not found! Fetching from the database...")
    with open("population_mimic.sql", "r") as file:
    	sql_query_mimic = file.read()

    population_mimic = pd.read_sql(sql_query_mimic, con=engine)

    # List of lab columns 
    # Convert them to float
    population_mimic['stayid'] = 'mimic_' + population_mimic['stayid'].astype(str)
    population_mimic.to_parquet(file_path, index=False)  # Save for future use
    #print(f" Saved data to {file_path}\n")
#print(lab_results.notna().sum())


#select only those patients used in previous analysis
population_mimic = population_mimic[population_mimic['stayid'].isin(filtered_data['stayid'].unique())]

population_mimic['bmi'] = population_mimic['weight'] / (population_mimic['height'] / 100) **2
population_mimic = population_mimic.rename(columns={"died_within_2w": 'death'})


# Metavision

store_file_pop_mv5 = "MV5_lab_population.csv"
file_path_mv5 = os.path.join(DATA_DIR, store_file_pop_mv5)
#select only those patients used in previous analysis

#print(f"\nLoading data from {store_file_mv5}\n.")

population_mv5 = pd.read_csv(file_path_mv5)
population_mv5['stayid'] = 'mv5_' + population_mv5['stayid'].astype(str)
population_mv5['patientid'] = 'mv5_' + population_mv5['patientid'].astype(str) # This is to secure uniquenes between mv5 and mv6

store_file_pop_mv6 = "MV6_lab_population.csv"
file_path_mv6 = os.path.join(DATA_DIR, store_file_pop_mv6)
#select only those patients used in previous analysis

#print(f"\nLoading data from {store_file_mv5}\n.")

population_mv6 = pd.read_csv(file_path_mv6)
population_mv6['stayid'] = 'mv6_' + population_mv6['stayid'].astype(str)
population_mv6['patientid'] = 'mv6_' + population_mv6['patientid'].astype(str)

population_mv = pd.concat([population_mv5, population_mv6])

population_mv = population_mv[population_mv['stayid'].isin(filtered_data['stayid'].unique())]

population_mv['bmi'] = population_mv['weight'] / (population_mv['height']) **2
population_mv.loc[(population_mv['bmi'] < 5) | (population_mv['bmi'] > 100), 'bmi'] = np.nan # there are some unreasonable values...

mv_errors = population_mv[~population_mv['stayid'].isin(filtered_data['stayid'].unique())]
mv_errors2 = filtered_data[filtered_data['stayid'].str.contains('mv') & ~filtered_data['stayid'].isin(population_mv['stayid'])]
#breakpoint()

#population_mv= population_mv.rename(columns={"died_within_2w": 'death'})

# Merge datasets to get actual std from merged data
# Columns to join
join_cols = [
    'stayid', 'albumin_dose_within_24h', 'fluid_volume_within_24h',
    'blood_cells_volume', 'plasma_volume', 'platelets_volume',
    'crrt', 'max_crp'
]

# Prepare filtered_data subset
filtered_subset = filtered_data[join_cols].drop_duplicates()

# Join to each dataset
population_ams = population_ams.join(filtered_subset.set_index('stayid'), on='stayid', how='left')
population_mimic = population_mimic.join(filtered_subset.set_index('stayid'), on='stayid', how='left')
population_hirid = population_hirid.join(filtered_subset.set_index('stayid'), on='stayid', how='left')
population_mv = population_mv.join(filtered_subset.set_index('stayid'), on='stayid', how='left')

population_total = pd.concat([population_ams, population_mimic, population_hirid, population_mv], ignore_index=True)

def get_summary(df):
    patients = df['patientid'].nunique()
    died_count = df['death'].sum()
    died_percent = died_count / patients * 100 if patients > 0 else 0

    # IV fluids [ml] → convert to liters
    iv_vals = df['fluid_volume_within_24h'] / 1000
    iv_median = iv_vals.median()
    iv_q1 = iv_vals.quantile(0.25)
    iv_q3 = iv_vals.quantile(0.75)

    # Binary treatments
    albumin_count = (df['albumin_dose_within_24h'] > 0).sum()
    albumin_percent = albumin_count / patients * 100 if patients > 0 else 0

    blood_cells_count = (df['blood_cells_volume'] > 0).sum()
    blood_cells_percent = blood_cells_count / patients * 100 if patients > 0 else 0

    plasma_count = (df['plasma_volume'] > 0).sum()
    plasma_percent = plasma_count / patients * 100 if patients > 0 else 0

    platelets_count = (df['platelets_volume'] > 0).sum()
    platelets_percent = platelets_count / patients * 100 if patients > 0 else 0

    crrt_count = df['crrt'].sum()
    crrt_percent = crrt_count / patients * 100 if patients > 0 else 0

    # Max CRP (only non-negative)
    crp_filtered = df.loc[df['max_crp'] >= 0, 'max_crp']
    crp_median = crp_filtered.median()
    crp_q1 = crp_filtered.quantile(0.25)
    crp_q3 = crp_filtered.quantile(0.75)

    # Age, BMI, LoS: median [IQR]
    age_median = df['age'].median()
    age_q1 = df['age'].quantile(0.25)
    age_q3 = df['age'].quantile(0.75)

    bmi_median = df['bmi'].median()
    bmi_q1 = df['bmi'].quantile(0.25)
    bmi_q3 = df['bmi'].quantile(0.75)

    los_median = df['los'].median()
    los_q1 = df['los'].quantile(0.25)
    los_q3 = df['los'].quantile(0.75)

    return {
        'patients': patients,
        'admissions': df['stayid'].nunique(),

        'age_str': f"{age_median:.1f} [{age_q1:.1f}-{age_q3:.1f}]",
        'bmi_str': f"{bmi_median:.1f} [{bmi_q1:.1f}-{bmi_q3:.1f}]",
        'los_str': f"{los_median:.2f} [{los_q1:.2f}-{los_q3:.2f}]",

        'male': (df['sex'] == 'male').sum(),
        'female': (df['sex'] == 'female').sum(),
        'unknown': (~df['sex'].isin(['male', 'female'])).sum(),

        'died_str': f"{died_count} ({died_percent:.1f}\%)",

        'iv_str': f"{iv_median:.1f} [{iv_q1:.1f}-{iv_q3:.1f}]",

        'albumin_str': f"{albumin_count} ({albumin_percent:.1f}\%)",
        'blood_cells_str': f"{blood_cells_count} ({blood_cells_percent:.1f}\%)",
        'plasma_str': f"{plasma_count} ({plasma_percent:.1f}\%)",
        'platelets_str': f"{platelets_count} ({platelets_percent:.1f}\%)",
        'crrt_str': f"{crrt_count} ({crrt_percent:.1f}\%)",

        'crp_str': f"{crp_median:.1f} [{crp_q1:.1f}-{crp_q3:.1f}]"
    }


# Generate summaries
summary_total = get_summary(population_total)
summary_ams   = get_summary(population_ams)
summary_mimic = get_summary(population_mimic)
summary_hirid = get_summary(population_hirid)
summary_mv    = get_summary(population_mv)

# Build rows with the new format
rows = [
    ["Patients", "", summary_total['patients'], summary_ams['patients'], summary_mimic['patients'], summary_hirid['patients'], summary_mv['patients']],
    ["Admissions", "", summary_total['admissions'], summary_ams['admissions'], summary_mimic['admissions'], summary_hirid['admissions'], summary_mv['admissions']],
    ["Age [y]", "", summary_total['age_str'], summary_ams['age_str'], summary_mimic['age_str'], summary_hirid['age_str'], summary_mv['age_str']],
    ["BMI [kg/m$^2$]", "", summary_total['bmi_str'], summary_ams['bmi_str'], summary_mimic['bmi_str'], summary_hirid['bmi_str'], summary_mv['bmi_str']],
    ["LoS [d]", "", summary_total['los_str'], summary_ams['los_str'], summary_mimic['los_str'], summary_hirid['los_str'], summary_mv['los_str']],
    [r"\multirow{3}{*}{Sex}", "M.", summary_total['male'], summary_ams['male'], summary_mimic['male'], summary_hirid['male'], summary_mv['male']],
    ["", "F.", summary_total['female'], summary_ams['female'], summary_mimic['female'], summary_hirid['female'], summary_mv['female']],
    ["", "Unk.", summary_total['unknown'], summary_ams['unknown'], summary_mimic['unknown'], summary_hirid['unknown'], summary_mv['unknown']],
    ["Died", "", summary_total['died_str'], summary_ams['died_str'], summary_mimic['died_str'], summary_hirid['died_str'], summary_mv['died_str']],
    ["IV Fluids [l]", "", summary_total['iv_str'], summary_ams['iv_str'], summary_mimic['iv_str'], summary_hirid['iv_str'], summary_mv['iv_str']],
    ["Albumin", "", summary_total['albumin_str'], summary_ams['albumin_str'], summary_mimic['albumin_str'], summary_hirid['albumin_str'], summary_mv['albumin_str']],
    ["Blood Cells", "", summary_total['blood_cells_str'], summary_ams['blood_cells_str'], summary_mimic['blood_cells_str'], summary_hirid['blood_cells_str'], summary_mv['blood_cells_str']],
    ["Plasma", "", summary_total['plasma_str'], summary_ams['plasma_str'], summary_mimic['plasma_str'], summary_hirid['plasma_str'], summary_mv['plasma_str']],
    ["CRRT", "", summary_total['crrt_str'], summary_ams['crrt_str'], summary_mimic['crrt_str'], summary_hirid['crrt_str'], summary_mv['crrt_str']],
    ["Max CRP [mg/L]", "", summary_total['crp_str'], summary_ams['crp_str'], summary_mimic['crp_str'], summary_hirid['crp_str'], summary_mv['crp_str']]
]

df = pd.DataFrame(rows, columns=["Metric", "", "Total", "AmsterdamUMCdb", "MIMIC IV", "HiRID", "Prague MV"])
df.to_csv('population_table.csv', index=False)


# Convert to LaTeX
latex_table = df.to_latex(
    index=False,
    escape=False,
    column_format="llrrrrr",  # One more r for the new column
    caption="Population summary statistics for AmsterdamUMCdb, MIMIC IV, HiRID, and Prague MV",
    label="tbl-population_summary",
    position='h!'

)
latex_table = latex_table.replace("\\begin{tabular}", "\\footnotesize\n\\begin{tabular}")
#display(Latex(latex_table))



# 1. Assign database based on stayid
def assign_db(stayid):
    if stayid.startswith('ams_'):
        return 'AmsterdamUMCdb'
    elif stayid.startswith('mimic_'):
        return 'MIMIC IV'
    elif stayid.startswith('hirid_'):
        return 'HiRID'
    elif stayid.startswith('mv5') or stayid.startswith('mv6'):
        return 'Prague MV'
    else:
        return 'Unknown'

population_total['database'] = population_total['stayid'].apply(assign_db)
# 2. Convert treatment variables to binary (0=no, >0=yes)
population_total['albumin_bin'] = (population_total['albumin_dose_within_24h'] > 0).astype(int)
population_total['blood_cells_bin'] = (population_total['blood_cells_volume'] > 0).astype(int)
population_total['plasma_bin'] = (population_total['plasma_volume'] > 0).astype(int)
population_total['platelets_bin'] = (population_total['platelets_volume'] > 0).astype(int)

# 3. Define columns for TableOne
continuous_vars = [
    'age',
    'bmi',
    'los',
    'fluid_volume_within_24h',
    'max_crp'
]

categorical_vars = [
    'sex',
    'died',
    'albumin_bin',
    'blood_cells_bin',
    'plasma_bin',
    'platelets_bin',
    'crrt'
]

# 4. Rename columns for nicer appearance in the table
rename_dict = {
    'age': 'Age [y]',
    'bmi': 'BMI [kg/m²]',
    'los': 'Length of Stay [d]',
    'fluid_volume_within_24h': 'IV Fluids [L]',
    'max_crp': 'Max CRP [mg/L]',

    'albumin_bin': 'Albumin',
    'blood_cells_bin': 'Blood Cells',
    'plasma_bin': 'Plasma',
    'platelets_bin': 'Platelets',
    'crrt': 'CRRT',
    'sex': 'Sex',
    'death': 'Died'
}

population_total = population_total.rename(columns=rename_dict)

# 5. Build TableOne with nonnormal treatment of continuous variables
t = TableOne(
    population_total,
    columns=list(rename_dict.values()),
    categorical=[
        'Sex',
        'Died',
        'Albumin',
        'Blood Cells',
        'Plasma',
        'Platelets',
        'CRRT'
    ],
    groupby='database',
    nonnormal=[
        'Age [y]',
        'BMI [kg/m²]',
        'Length of Stay [d]',
        'IV Fluids [L]',
        'Max CRP [mg/L]'
    ],
    missing=False
)

# 6. LaTeX output
latex_table = t.to_latex(
    #index=False,
    escape=True,
    caption="Population summary statistics for AmsterdamUMCdb, MIMIC IV, HiRID, and Prague MV",
    label="tab:population_summary2"
)
latex_table = latex_table.replace("\\begin{tabular}", "\\tiny\n\\begin{tabular}")
#display(Markdown(latex_table))
