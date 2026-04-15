#%% [markdown]
# ---
# title: "How long can I trust laboratory results in critically ill patients: a large-scale  study on temporal validity of laboratory analytes"
# subtitle: "SDC calculation and plots" 
# bibliography: library.bib
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
#    include-in-header:
#      text: |
#        \usepackage{multirow}
#---
# This supplementary file contains details for three parts of the main text. In the first chapter, the data used to calculate SDC are presented. The second chapter contains a description of the analyte dynamics model used and plots for all analytes.
# Results of bootstrapping with estimated uncertainties are presented in the third chapter. 
# %%
#| echo: false
#| output: asis



import psycopg2
import warnings
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
from matplotlib.colors import LinearSegmentedColormap
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
from IPython.display import Markdown, display
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import shapiro, normaltest, probplot, uniform, t, norm, trimboth, skewnorm, jf_skew_t, chi2, gaussian_kde
from skewt_scipy.skewt import skewt
from scipy.special import gammaln
from properscoring import crps_ensemble
from matplotlib.colors import LinearSegmentedColormap
import sys 
import pdb
sys.path.insert(0, os.path.expanduser('~/icu_database/lib'))
from probabilitymodel import probabilitymodel as ProbabilityModel
import rpy2
from rpy2.robjects import FloatVector, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import io
import logging

psych = importr("psych")
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

units = {3024561: 'g/l', 3003458: 'mmol/l', 3006140: 'umol/l', 3012095: 'mmol/l', 3032080: '', 43534077: 'mmol/l', 40762351: 'g/dl',
         3020564: 'umol/l', 3010813: '$10^9$/l', 3007461: '$10^9/l$', 3035995: 'IU/l',
         3015377: 'mmol/l', 3000285: 'mmol/l', 3020460: 'mg/dl', 42869588: '\%'}
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
test_names = {3024561: 'Albumin', 3003458: 'Phosphate', 3006140: 'Bilirubin', 3012095: 'Magnesium', 3032080: 'INR',
              43534077: 'Urea', 40762351: 'Hemoglobin', 3020564: 'Creatinine', 3010813: 'WBC', 3007461: 'Platelets', 3035995: 'ALP', 3015377: 'Calcium', 3000285: 'Sodium', 3020460: 'CRP', 42869588: 'Hematocrit'}
#maximal_measurement_error = {3024561: 5, 3003458: 0.5, 3006140: 20, 3012095: 0.4, 3032080: 0.4, 43534077: 0.4, 40762351: 2, 3020564: 20, 3010813: 5, 3007461: 50, 3035995: 5, 3015377: 0.2, 3000285: 2, 3020460: 2, 42869588: 2}
minimal_measurement_error = {3024561: 0.3, 3003458: 0.02, 3006140: 0.1, 3012095: 0.02, 3032080: 0.025, 43534077: 0.05, 40762351: 0.2, 3020564: 2, 3010813: 0.1, 3007461: 1, 3035995: 3, 3015377: 0.02, 3000285: 0.3, 3020460: 1, 42869588: 1}
test_cutoffs = {3024561: 5, 3003458: 0.5, 3006140: 5, 3012095: 0.5, 3032080: 0.5, 43534077: 2, 40762351: 1, 3020564: 20, 3010813: 1, 3007461: 20, 3035995: 10, 3015377: 0.5, 3000285: 3, 3020460: 5, 42869588: 3}
list_of_tests = sorted(list_of_tests, key=lambda x: test_names[x].lower())



ugly_counter = 0

def plt_to_markdown(pltx, name=None, label=None, reference=None, pformat='jpg'):
    global ugly_counter

    if name == None:
        name = f'{ugly_counter}.{pformat}'
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

    out_str = f"""![{label}]({name_adres} "{reference if reference is not None else label}")"""
    #print(out_str)
    #print(out_str)
    pltx.savefig(name_adres, format=pformat, dpi=150)
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


def generate_model_evaluation_table(models, model_names=None, x=None, y=None, test=None):
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

        if model is not None:
            eval_res = evaluate_probability_model(model, xi, yi, plot=True, test=test, model_name=model_names[i])
            eval_res['Model'] = model_names[i] if model_names else f"Model {i+1}"
            results.append(eval_res)
        else:
            eval_res = {'Model': model_names[i] if model_names else f"Model {i+1}"}
            results.append

    # Convert to DataFrame for nice formatting
    df_results = pd.DataFrame(results)
    df_results = df_results[['Model', 'Shapiro-Wilk p-value', 'D’Agostino p-value', 
                             'Coverage 95% CI', 'Negative Log Likelihood']]

    # Format floats for publication
    formatted_df = df_results.copy()
    for col in formatted_df.columns[1:]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

    # Generate LaTeX table string
    latex_table = formatted_df.to_latex(index=False, escape=True, caption=f"Model evaluation results for {test_names[test]}",
                                        label="tab:model_evaluation", column_format="lcccc")

    return latex_table

def evaluate_probability_model(model, x=None, y=None, plot=True, test=None, model_name=None):
    """
    Evaluate ProbabilityModel object for:
    - Residual normality (Shapiro-Wilk and D’Agostino’s test)
    - QQ plot
    - Histogram of standardized residuals
    - Empirical coverage of 95% CI
    - Negative Log Likelihood (NLL)
    - Probability Integral Transform (PIT) histogram

    Returns dictionary with results.
    """
    import numpy as np
    from scipy.stats import shapiro, normaltest
    from statsmodels.graphics.gofplots import qqplot
    import matplotlib.pyplot as plt

    if x is None: x = model.x
    if y is None: y = model.y

    mu, sigma = model.calculate_mean_scale(x)
    nll = model.nll(x=x, y=y)

    # Standardized residuals
    standardized_residuals = (y - mu) / sigma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress all warnings inside
        # Normality tests
        shapiro_stat, shapiro_p = shapiro(standardized_residuals)
        dagostino_stat, dagostino_p = normaltest(standardized_residuals)

    # Coverage
    lower, upper = model.ci(x, alpha=0.05)
    coverage = np.mean((y >= lower) & (y <= upper))

    # PIT
    pit_values = model.cdf(x, y)
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        # 1. QQ plot using model-based theoretical quantiles
        plt.sca(axs[0])
        
        # Convert x and y to numpy arrays if they are DataFrames or Series
        x_array = np.asarray(x).flatten()
        y_array = np.asarray(y).flatten()

        # Calculate PIT values for all data points (model CDF evaluated at observed y)
        pit_values = model.cdf(x_array, y_array)

        # Sort PIT values for PP plot
        sorted_pit = np.sort(pit_values)

        # Calculate empirical cumulative probabilities
        n = len(sorted_pit)
        empirical_probs = np.arange(1, n + 1) / n

        # Plot PP plot
        axs[0].plot(sorted_pit, empirical_probs, 'o')
        axs[0].plot([0, 1], [0, 1], 'r--', label='45° line')
        axs[0].text(0.02, 0.98, model_name, transform=axs[0].transAxes,
            ha='left', va='top', fontsize=12)

        axs[0].set_title(f"PP plot of PIT values")
        axs[0].set_xlabel('Theoretical cumulative probability (PIT value)')
        axs[0].set_ylabel('Empirical cumulative probability')
        axs[0].legend()
        # 2. Histogram of standardized residuals
        axs[1].hist(standardized_residuals, bins=80, edgecolor='k', density=True)
        axs[1].set_title('Histogram of standardized residuals')
        axs[1].set_xlabel('Standardized residual')
        axs[1].set_ylabel('Density')

        # 3. PIT histogram
        axs[2].hist(pit_values, bins=20, edgecolor='k', density=True)
        axs[2].set_title('PIT histogram')
        axs[2].set_xlabel('PIT value')
        axs[2].set_ylabel('Density')

        plt.tight_layout()
        plt_to_markdown(plt, pformat='jpg', label=f'{test_names[test]} - plots showing model alignment')

    results = {
        'Shapiro-Wilk p-value': shapiro_p,
        'D’Agostino p-value': dagostino_p,
        'Coverage 95% CI': coverage,
        'Negative Log Likelihood': nll,
    }

    return results

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

def make_fit_and_plot(df1, df2=None, l1='less severe', l2='more severe', title='Some plot', unit='',
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
    fig, axes = plt.subplots(len(window_starts), 1, figsize=(8, rows * 4), sharex=False, sharey="all", layout="constrained")  # Adjust figure size
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
        if times[indx] == 'charttime':
            a = df1[(df1['charttime'] >= start)
                    & (df1['charttime'] < start + window_length[indx])].copy()
            if second:
                b = df2[(df2['charttime'] >= start)
                        & (df2['charttime'] < start + window_length[indx])].copy()
    #          print("a is charttime:")
    #         print(a)
        else:
            a = df1[(df1['clock_time'] > start - morning_tolerance)
                    & (df1['clock_time'] < start + window_length[indx])].copy()
            if second:
                b = df2[(df2['clock_time'] > start - morning_tolerance)
                        & (df2['clock_time'] < start + window_length[indx])].copy()
            
            # Fitering values without value within set morning tests
            a['morning_check'] = a.groupby('stayid')['clock_time'].transform('first')
            a = a[(a['morning_check'] - start).abs() < morning_tolerance]
            if second:
                b['morning_check'] = b.groupby('stayid')['clock_time'].transform('first')
                b = b[(b['morning_check'] - start).abs() < morning_tolerance]
    #        print("a is clocktime:")
     #       print(a)

        a = recalculate_relative_to_first(a)
        a = a[a['time'] < plot_length]


        if max_range < a['difference'].abs().quantile(0.99):
            max_range = a['difference'].abs().quantile(0.99)


        if second:
            b = recalculate_relative_to_first(b)
            b = b[b['time'] < plot_length]
            #b = b[b['difference'].abs() <= b['difference'].abs().quantile(0.99)]
            if max_range < b['difference'].abs().quantile(0.99):
                max_range = b['difference'].abs().quantile(0.99)

        ## FIRST data frame
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
            alpha=0.2, edgecolor='#1E8925', facecolor='#6BD572')
            axes[indx].plot(tf_zerod, m_t, c= '#1E8925', alpha=0.3, label=model_a.model_equation())

            #m_t, delta_t = model_a.predict(a['time'])
            #outliers = a[(a['difference'] > m_t+delta_t) | (a['difference'] < m_t - delta_t)]
            #plt.scatter(outliers['time'] + start, outliers['difference'], marker='.', s=1, alpha=0.1)
        else:
            model_a = None
        models_a.append(model_a)
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

        axes[indx].legend(fontsize='small', loc="lower left")
        axes[indx].set_xlabel(f'Time (h)')
        #axes[indx].set_xlabel(f'{times[indx]} (h)')
        axes[indx].grid(linestyle='--', color='lightgrey')


        # Making scatter 


        devia_low = []
        devia_high = []
        mean_low = []
        mean_high = []
        t_low =[]
        t_high = []
        labels= []
        t_labels =[]
        x_labels = [] 
        outliers = []
        for i in range(number_of_bins):
            step = plot_length / number_of_bins * 1.0
            # Filtering is based on previously set time, however later time is calculated with the set
            # time window
            a_sub = a[((step*i) < a['time']) & (a['time'] <= (step*(i+1)))]

            if second:
                b_sub = b[((step*i) < b['time']) & (b['time'] <= (step*(i+1)))]
    #            print('a_sub:')
    #            print(a_sub)


            label_logic = False
            if len(a_sub) >= minimal_set_size:
                outliers.append(a_sub[(a_sub['difference'] < a_sub['difference'].quantile(0.025)) | (a_sub['difference'] > a_sub['difference'].quantile(0.975))])
                mean_low.append(a_sub['difference'].mean())
                devia_low.append([abs(a_sub['difference'].quantile(0.025) - mean_low[-1]), abs(a_sub['difference'].quantile(0.975) - mean_low[-1])])
                t_low.append(a_sub['time'].mean())

                labels.append(f'$n_1$={len(a_sub)}')
                t_labels.append(a_sub['time'].mean())
                x_labels.append( mean_low[-1])
                label_logic = True

            else:
                for time, diff in zip(a_sub['time'], a_sub['difference']):
                    mean_low.append(diff)
                    t_low.append(time)
                    devia_low.append([float('nan'), float('nan')])


            if second:
                if len(b_sub) >= minimal_set_size:
                    mean_high.append(b_sub['difference'].mean())
                    devia_high.append([abs(b_sub['difference'].quantile(0.025) - mean_low[-1]), abs(b_sub['difference'].quantile(0.975) - mean_low[-1])])
                    t_high.append(b_sub['time'].mean())

                    # Adding label base if it is already existing or not
                    if label_logic:
                        labels[-1] =  labels[-1] + f'\n$n_2$={len(b_sub)}'
                    else:
                        labels.append(f'$n_2$={len(b_sub)}')
                        t_labels.append(step*i + step/2)
                        x_labels.append(mean_high[-1])
                else:
                    for time, diff in zip(b_sub['time'], b_sub['difference']):
                        mean_high.append(diff)
                        t_high.append(time)
                        devia_high.append([float('nan'), float('nan')])


        # Mean and deviation plot
        axes[indx].text(0.02, 0.98, titles[indx], transform=axes[indx].transAxes,
            ha='left', va='top', fontsize=12)
        mean_low = np.array(mean_low)
        devia_low = np.transpose(np.array(devia_low))
        axes[indx].errorbar(np.array(t_low), mean_low, yerr=devia_low, fmt='g.', capsize=3.0, label=l1)
        if len(outliers) > 0 and not second:
            outliers = pd.concat(outliers)
            axes[indx].scatter(outliers['time'], outliers['difference'], c='g', marker='.', s=3.0)

        mean_high = np.array(mean_high)
        devia_high = np.transpose(np.array(devia_high))
        
        if second:
            axes[indx].errorbar(np.array(t_high) + 0.003, mean_high, yerr=devia_high, fmt='r.', capsize=3.0, label=l2)

        for i in range(len(t_labels)):
            axes[indx].text(t_labels[i] + 0.2, x_labels[i], labels[i], fontsize=7, ha='left', va='top', **{'color': 'black'})

        #if second:
        #    for i in range(len(t_high)):
        #        axes[indx].text(t_high[i] + 0.1, mean_high[i] - devia_high[i] * 1.15, labels_high[i], fontsize=7, va='bottom', ha='right', **{'color': 'red'})

        if cutoff != None:
            axes[indx].axhline(cutoff, c='grey', ls='--')
            axes[indx].axhline(-cutoff, c='grey', ls='--')

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

def make_fit_and_density_plot(df1, df2=None, l1='less severe', l2='more severe', title='Some plot', unit='',
               window_starts=(0, 24, 72), window_length=(24, 48, 96), plot_lengths=[8, 16, 24], number_of_bins=12, morning_tolerance = 1.5,
               times=['charttime', 'charttime', 'charttime'], minimal_set_size=10, measurement_std=None,
                 minimal_count_int=3, fit_minimal_count=10, cutoff=None, titles=['Early', 'Intermediate', 'Late'], label=None):
# minimal_count_int is interval for fitting model (preventing fitting if there is not enough

    if df2 is None:
        second = False
    else:
        second = True

    rows = len(window_starts)
    # setting plot
    #fig, axes = plt.subplots(len(window_starts), 1, figsize=(10, rows * 4), sharex=False, sharey="all", layout="constrained")  # Adjust figure size
    fig, axes = plt.subplots(len(window_starts), 1, figsize=(8, rows * 3.2), sharex=False, sharey="all", layout="constrained")  # Adjust figure size
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

        # Making scatter 




        # Mean and deviation plot
        axes[indx].text(0.02, 0.98, titles[indx] + f', n={len(a)}' , transform=axes[indx].transAxes,
            ha='left', va='top', fontsize=12)

        sub = a.sample(min(2000, len(a)))

        # Fit KDE on 2D subsample
        kde = gaussian_kde([sub["time"], sub["difference"]])

        # Evaluate on full data
        densities = kde([a["time"], a["difference"]])
        a["density"] = densities

        #blue_violet = LinearSegmentedColormap.from_list("blue_violet", ["darkblue", "magenta"])
        blue_violet = LinearSegmentedColormap.from_list("blue_violet", ["green", "green", "darkorange"])
        #axes[indx].scatter(a['time'], a['difference'], color='g', marker='.', s=25, label=l1, alpha=max(0.05, min(1, 1000/len(a))))
        axes[indx].scatter(a['time'], a['difference'], c=a['density'], cmap=blue_violet, marker='.', s=15, alpha=max(0.05, min(0.25, 10000/len(a))))


        # initialization and fitting
        if len(a[a['time'] < minimal_count_int]) > fit_minimal_count:
            model_a = ProbabilityModel(a['time'], a['difference'])
            model_a.fit(measurement_std=measurement_std, method='kinetic_jfskewt')
            #prediction
            tf = np.linspace(start, start + plot_length, 200)
            tf_zerod = tf - start
            m_t,  sigmal, sigmau = model_a.predict(tf_zerod) 
            #breakpoint()

            #axes[indx].fill_between(tf, m_t  + sigma_t, m_t - sigma_t,
            #alpha=0.2, ls='--', edgecolor='#1E8925', facecolor='#6BD572')

            axes[indx].fill_between(tf_zerod, sigmau, sigmal,
            #alpha=0.15, edgecolor='k', facecolor='k')
            #alpha=0.2, edgecolor='#1E8925', facecolor='#6BD572')
            label='2.5th and 97.5th percentile',
            alpha=0.2, edgecolor='orangered', facecolor='#6BD572')
            #axes[indx].plot(tf_zerod, m_t, c= 'white', ls='-', alpha=0.8, label=model_a.model_equation(), linewidth=2)
            axes[indx].plot(tf_zerod, m_t, c= 'orangered', ls='--', alpha=0.7, label='mean', linewidth=3)
            axes[indx].text(
                0.95, 0.05,               # x and y positions (bottom-right in axes fraction)
                model_a.model_equation(),          # your text
                transform=axes[indx].transAxes,
                fontsize=9,
                va='bottom', ha='right'   # align text to bottom-right
            )

            #m_t, delta_t = model_a.predict(a['time'])
            #outliers = a[(a['difference'] > m_t+delta_t) | (a['difference'] < m_t - delta_t)]
            #plt.scatter(outliers['time'] + start, outliers['difference'], marker='.', s=1, alpha=0.1)
        else:
            model_a = None
        models_a.append(model_a)

        if cutoff != None:
            axes[indx].axhline(cutoff, c='grey', ls='--', label=f'SDC = {cutoff:.2f}')
            axes[indx].axhline(-cutoff, c='grey', ls='--')


        axes[indx].legend(fontsize='small', loc="lower left")
        axes[indx].set_xlabel(f'Time (h)')
        #axes[indx].set_xlabel(f'{times[indx]} (h)')
        axes[indx].grid(linestyle='--', color='lightgrey')


    plt.setp(axes, ylim=(-max_range*1.05, max_range*1.05))

    #plt.xlabel('Time [h]')
    fig.supylabel('Change of ' + title + (f' ({unit})' if unit != '' else ''))
    #plt.legend(fontsize='small')

    #plt.suptitle(title)


    #plt.tight_layout()
    #plt.show()
    #plt.close()
    plt_to_markdown(plt, label=label, pformat='jpg')
    if second:
        return models_a, models_b
    else:
        return models_a

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

    
def plot_quartils(df0, x_col='time', y_col='difference', window_start=0, window_length=12, plot_lengt=12, measurement_std=None, 
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
    dfs, qs = split_into_quartiles(df0)
    labels = [f'<{qs[0]:.3g}', f'[{qs[0]:.3g}, {qs[1]:.3g}]', f'[{qs[1]:.3g}, {qs[2]:.3g}]', f'{qs[2]:.3g}>']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey="all", layout="constrained")
    axes = axes.flatten()

    max_range = 0
    for i, (df, ax) in enumerate(zip(dfs, axes)):
        df = df[(df['charttime'] < plot_lengt)]
        df = make_pairs(df, windows_start=[window_start], windows_length=[window_length]) 
        df = recalculate_relative_to_first(df)

        x = df['time']
        y = df['difference']
        alpha = min(1, 100/max(1,len(y)))
        #breakpoint()
        if max_range < df['difference'].abs().quantile(0.99):
            max_range = df['difference'].abs().quantile(0.99)

        # Scatter plot
        #ax.scatter(x, y, alpha=alpha)
        ax.set_title(f"{labels[i]}")
        #ax.set_xlabel(x_col)
        #ax.set_ylabel(y_col)
        ax.grid(linestyle='--', color='lightgrey')

        # Fit model
        if len(df) > 10:  # Prevent fitting if too few data points
            model = ProbabilityModel(df['time'], df['difference'])
            model.fit(measurement_std=measurement_std, method='kinetic_jfskewt')

            tf = np.linspace(window_start, window_start + plot_lengt, 200)
            tf_zerod = tf - window_start
            m_t,  sigmal_t, sigmau_t  = model.predict(tf_zerod) 
            #breakpoint()

            # Plot fitted curve with 95% CI
            #ax.fill_between(tf, m_t + delta_t, m_t - delta_t,
            #alpha=0.2, facecolor='#FF6666')
            #alpha=0.2, edgecolor='#CC0000', facecolor='#FF6666')
            #ax.plot(tf, m_t, c= '#CC0000', alpha=0.3, label=model.model_equation())
            ax.fill_between(tf_zerod, sigmau_t, sigmal_t,
            alpha=0.2, edgecolor='#1E8925', facecolor='#6BD572')
            ax.plot(tf_zerod, m_t, c= '#1E8925', alpha=0.2, label=model.model_equation())
            
            ax.legend()
        else:
            model = None
        
        models.append(model)

        devia_low = []
        mean_low = []
        t_low =[]
        labels_low = []
        a = df
        outliers = []
        for i in range(number_of_windows):
            step = plot_lengt / number_of_windows
            # Filtering is based on previously set time, however later time is calculated with the set
            # time window
            a_sub = a[((step*i) < a['time']) & (a['time'] <= (step*(i+1)))]

            if len(a_sub) >= minimal_set_size:
                outliers.append(a_sub[(a_sub['difference'] < a_sub['difference'].quantile(0.025)) | (a_sub['difference'] > a_sub['difference'].quantile(0.975))])
                devia_low.append(a_sub['difference'].std() * 1.96)
                mean_low.append(a_sub['difference'].mean())
                labels_low.append(f'n={len(a_sub)}')
                t_low.append(window_start + step*i + step/2)
            else:
                for time, diff in zip(a_sub['time'], a_sub['difference']):
                    mean_low.append(diff)
                    t_low.append(t+ time)
                    labels_low.append(None)
                    devia_low.append(float('nan'))

        # Mean and deviation plot
        mean_low = np.array(mean_low)
        devia_low = np.array(devia_low)
        ax.errorbar(np.array(t_low), mean_low, yerr=devia_low, fmt='g.', capsize=3.0)
        for i in range(len(t_low)):
            ax.text(t_low[i] + 0.1, mean_low[i], labels_low[i], fontsize=7, ha='left', va='top', **{'color': 'green'})

        if len(outliers) > 0:
            outliers = pd.concat(outliers)
            ax.scatter(outliers['time'] + window_start, outliers['difference'], c='g', marker='.', s=3.0)
        if cutoff != None:
            ax.axhline(cutoff, c='grey', ls='--')
            ax.axhline(-cutoff, c='grey', ls='--')

    plt.setp(axes, ylim=(-max_range*1.05, max_range*1.05))
    plt.suptitle("Quartiles of first value", fontsize=16)
    plt.show()
    plt.close()

    return models, labels

def plot_quartils_density(df0, x_col='time', y_col='difference', window_start=0, window_length=24, unit='', plot_lengt=8, measurement_std=None, 
             number_of_windows=10, minimal_set_size=10, cutoff=None, label=None, title=None):
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
    dfs, qs = split_into_quartiles(df0)
    labels = [f'<{qs[0]:.3g}', f'[{qs[0]:.3g}, {qs[1]:.3g}]', f'[{qs[1]:.3g}, {qs[2]:.3g}]', f'>{qs[2]:.3g}']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey="all", layout="constrained")
    #fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), sharex=False, sharey="all", layout="constrained")
    axes = axes.flatten()

    max_range = 0
    for i, (df, ax) in enumerate(zip(dfs, axes)):
        df = df[(df['charttime'] < plot_lengt)]
        df = make_pairs(df, windows_start=[window_start], windows_length=[window_length]) 
        df = recalculate_relative_to_first(df)

        x = df['time']
        y = df['difference']
        alpha = min(1, 100/max(1,len(y)))
        #breakpoint()
        if max_range < df['difference'].abs().quantile(0.99):
            max_range = df['difference'].abs().quantile(0.99)

        # Scatter plot
        #ax.scatter(x, y, alpha=alpha)
        #ax.set_xlabel(x_col)
        #ax.set_ylabel(y_col)
        ax.grid(linestyle='--', color='lightgrey')


        if cutoff != None:
            ax.axhline(cutoff, c='grey', ls='--', label='SDC')
            ax.axhline(-cutoff, c='grey', ls='--')


        # Mean and deviation plot
        ax.text(0.02, 0.98, f"Quartile: {labels[i]} n={len(df)}" , transform=ax.transAxes,
            ha='left', va='top', fontsize=10)

        sub = df.sample(min(2000, len(df)))

        # Fit KDE on 2D subsample
        kde = gaussian_kde([sub["time"], sub["difference"]])

        # Evaluate on full data
        densities = kde([df["time"], df["difference"]])
        df["density"] = densities

        #blue_violet = LinearSegmentedColormap.from_list("blue_violet", ["darkblue", "magenta"])
        blue_violet = LinearSegmentedColormap.from_list("blue_violet", ["green", "green", "darkorange"])
        #axes[indx].scatter(a['time'], a['difference'], color='g', marker='.', s=25, label=l1, alpha=max(0.05, min(1, 1000/len(a))))
        ax.scatter(df['time'], df['difference'], c=df['density'], cmap=blue_violet, marker='.', s=15, alpha=max(0.08, min(0.25, 2500/len(df))))



        # Fit model
        if len(df) > 10:  # Prevent fitting if too few data points
            model = ProbabilityModel(df['time'], df['difference'])
            model.fit(measurement_std=measurement_std, method='kinetic_jfskewt')

            tf = np.linspace(window_start, window_start + plot_lengt, 200)
            tf_zerod = tf - window_start
            m_t,  sigmal_t, sigmau_t  = model.predict(tf_zerod) 
            #breakpoint()

            # Plot fitted curve with 95% CI

            ax.fill_between(tf_zerod, sigmau_t, sigmal_t,
            alpha=0.2, edgecolor='orangered', facecolor='#6BD572', label='2.5 and 97.5th percentile')
            #axes[indx].plot(tf_zerod, m_t, c= 'white', ls='-', alpha=0.8, label=model_a.model_equation(), linewidth=2)
            ax.plot(tf_zerod, m_t, c= 'orangered', ls='--', alpha=0.7, label='mean', linewidth=3)
            ax.text(
                0.95, 0.05,               # x and y positions (bottom-right in axes fraction)
                model.model_equation(),          # your text
                transform=ax.transAxes,
                fontsize=9,
                va='bottom', ha='right'   # align text to bottom-right
            )
            
            ax.legend(fontsize='small', loc="lower left")
        else:
            model = None

        
        models.append(model)


    plt.setp(axes, ylim=(-max_range*1.05, max_range*1.05))
    #plt.suptitle(title + "stratified by quartiles of first value", fontsize=16)
    fig.supxlabel('Time (h)')
    fig.supylabel('Change of ' + title + (f' ({unit})' if unit != '' else ''))
    plt_to_markdown(plt, label=label, pformat='jpg')
    plt.close()

    return models, labels


def print_time(event):
    #print(event + (f'\n time of running:{time.time() - starttime}\n'))
    return 0

from google.cloud import bigquery




# %%
#| echo: false
#| output: asis





cash_file_path = 'cash_lab_stability.parquet'
print(f"\n### Loading data from database\n")
# Loading Data
if os.path.exists(cash_file_path) and True:
#    print(f"FIltered data loaded from file {cash_file}.")
    filtered_data = pd.read_parquet(cash_file_path)
else:

    raise Exception('Missing data file')
    print(f" Saved data to {cash_file_path}\n")

#ids_to_use = np.random.choice(filtered_data['stayid'].unique(), size=20000, replace=False)
#filtered_data=filtered_data[filtered_data['stayid'].isin(ids_to_use)]

# %% [markdown]
# # SDC - Smallest detectable change calculation 
# We tried to exploit possible accidental (or intentional) double sampling of the same test to estimate SDC. We assumed that for most cases and studied analytes there is a low chance of significant clinical change within 5 minutes.
# Therefore, if duplicate measurements within this time window occur, the difference can be attributed to the error of the method (combining all steps from acquiring the blood sample, transport, and processing).
# A major limitation of this approach is that these data pairs can be manually inputted duplicates of the same result, as at least part of the database may include manual entries.
# To filter the data, we removed records with exactly the same time and value. Then, a z-score filter was applied on differences between pairs of measurements, and pairs with z-scores > 4 were removed.
# Calculations were perfomed based on [@de_vet_when_2006]. Formula for Standard error of measurement (SEM):
# $$ SEM = \sigma \cdot \sqrt{1 - ICC}, $$
# where $\sigma$ is pooled standard deviation of data used, and $ICC$ is interclass corrocelation coefficient (Two-way mixed effects, absolute agreement, single rater/measurement) and 1 is one. SDC is then:
# $$ SDC = 1.96 \cdot \sqrt{2} \cdot SEM. $$
# For the calculation of the 95 % CI of the SDC, the value provided by the R psych package was used, and a $\chi^2$-distribution formula was applied to obtain the 95 % interval for the pooled standard deviation. The corresponding standard deviations were subsequently combined as the square root of the sum of squares. This procedure assumes normality of the data, which may not hold for all tests; however, we consider the resulting error to be negligible.  
# To visualize the data, two plots were generated for each analyte. The first shows a histogram of the differences in values between consecutive measurements, and the second shows the distribution of the measurement values.
#
#
# %%
#| echo: false
#| output: asis

sdc_report = pd.DataFrame(columns=['test', 'SDC', 'type', 'ICC', 'ICC CI 95%', 'ICC p-value'])
sdc_export = {}
sdc_cis = {}
for test in list_of_tests:

    print(f'\n## {test_names[test]}\n') 
    data_to_process = filtered_data[filtered_data['itemid'] == test][:max_data]
    twins = data_to_process[['stayid', 'charttime', 'value']].copy()
    twins = twins.sort_values(['stayid', 'charttime']).reset_index(drop=True)

    # Step 2: Create shifted columns
    twins['stayid_next'] = twins['stayid'].shift(-1)
    twins['charttime_next'] = twins['charttime'].shift(-1)
    twins['value_next'] = twins['value'].shift(-1)

    # Step 3: Compute time difference to next row (in hours)
    twins['time_diff'] = twins['charttime_next'] - twins['charttime']
    twins['value_diff'] = twins['value_next'] - twins['value']

    # Step 4: Filter rows where stay_id matches and time_diff ≤ 5 min
    close_twins = twins[
        (twins['stayid'] == twins['stayid_next']) &
        (twins['time_diff'] <= (5 / 60))
    ].copy()

    close_twins = close_twins[
        (close_twins['value_diff'] != 0) |
        (close_twins['time_diff'] != 0)
    ].copy()


    close_twins_filtered = close_twins[np.abs(zscore(close_twins['value_diff'])) <= 4]
    print(f'\n{len(close_twins)} measurements with time difference < 5 min, {len(close_twins) - len(close_twins_filtered)} were removed by z-score filtering.\n')
    #breakpoint()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # First histogram: value_diff
    close_twins['value_diff'].hist(bins=30, ax=axes[0])
    axes[0].set_xlabel("Value difference")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram of close tests differences")

    # Second histogram: value
    working_values = pd.DataFrame()
    working_values['value'] = close_twins_filtered['value_next']
    all_values = pd.concat([close_twins_filtered['value'], working_values['value']], axis=0)
    all_values.hist(bins=30, ax=axes[1])
    axes[1].set_xlabel(f"Value, {units[test]}")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram of values")

    if len(close_twins_filtered) < 23:
        print("Because of small dataset, default values were used\n")

        new_row = pd.DataFrame([{
            'test': test_names[test],
            'type': 'Empirical',
            'SDC': None,
                'SDC CI 95%': '',
                'ICC': np.nan,
                'ICC CI 95%': np.nan,
                'ICC p-value': np.nan,
            }])

        test_cutoffs[test] = None
        sdc_report = pd.concat([sdc_report, new_row], ignore_index=True)
        sdc_cis.update({test: 'insuff. data'})
    else:
        # Z-score fintering

        # plotting gauss
        #mean_clean = close_twins_filtered['value_diff'].mean()
        #std_clean = close_twins_filtered['value_diff'].std()
        #xmin, xmax = plt.xlim()
        #x = np.linspace(xmin, xmax, 100)
        #p = norm.pdf(x, mean_clean, std_clean) 
        #plt.plot(x, p, 'r', linewidth=2)


        # data preparation
        first_measurement = close_twins_filtered[['stayid', 'value']]
        first_measurement['measurement'] = 1
        second_measurement = pd.DataFrame()
        second_measurement[['stayid', 'value']] = close_twins_filtered[['stayid', 'value_next']]
        second_measurement['measurement'] = 2


        icc_data = pd.concat([first_measurement['value'], second_measurement['value']])

        # calculating ICC
        #icc_output = pg.intraclass_corr(data=icc_data, raters='measurement', ratings='value', targets='stayid')


        #icc_matrix = icc_data.pivot(index="stayid", columns="measurement", values="value")

        # Send to R
        # Suppress R warnings
        rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings

        icc_data_r = rpy2.robjects.r.matrix(FloatVector(icc_data), ncol=2)
        icc_results = psych.ICC(icc_data_r)
        icc_output = pandas2ri.rpy2py(icc_results[0])
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")  # suppress all warnings inside
        #    icc_results = psych.ICC( icc_data_r)

        # icc_output is now an R object; convert back to Python DataFrame if needed

        # ICC(2,1) used because its the same formula as "Two-way mixed effects, absolute agreement, single rater/measurement"
        # which we need, source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/
        #icc = icc_output.loc[(icc_output['Type'] == 'ICC2'), 'ICC'].values[0]
        #CI95 = icc_output.loc[(icc_output['Type'] == 'ICC2'), 'CI95%'].values[0]
        #icc_low = CI95[0]
        #icc_high = CI95[1]
        #pval = icc_output.loc[(icc_output['Type'] == 'ICC2'), 'pval'].values[0]

        #print(icc_output)


        row = icc_output.loc[icc_output['type'] == 'ICC2'].iloc[0]
        icc = row['ICC']
        icc_low = row['lower bound']
        icc_high = row['upper bound']
        pval = row['p']

        #display(Markdown(icc_output.to_latex(float_format='%.2f', index=False, escape=True)))

        # calculating SDC
        #SEM = icc_data['value'].std() * np.sqrt(1 - icc)
        # group sizes
        n1 = len(first_measurement['value'])
        n2 = len(second_measurement['value'])

        # sample standard deviations (ddof=1 for unbiased estimator)
        s1 = np.std(first_measurement['value'], ddof=1)
        s2 = np.std(second_measurement['value'], ddof=1)

        # pooled standard deviation (spooled)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

        SEM = pooled_std * np.sqrt(1 - icc)

        SDC = 1.96 * np.sqrt(2) * SEM

        # 95% CI for pooled variance
        dof = n1 + n2 - 2
        var_low = (dof * pooled_std**2) / chi2.ppf(0.975, dof)
        var_up  = (dof * pooled_std**2) / chi2.ppf(0.025, dof)

        # convert to pooled std CI
        pooled_std_low = np.sqrt(var_low)
        pooled_std_up  = np.sqrt(var_up)

        # approximate pooled std SD for delta method
        pooled_std_sd = (pooled_std_up - pooled_std_low) / (2 * 1.96)
        ICC_sd = (icc_high - icc_low) / (2 * 1.96)

        # delta method variance for SEM
        var_SEM = (np.sqrt(1 - icc))**2 * pooled_std_sd**2 + \
                  (pooled_std / (2 * np.sqrt(1 - icc)))**2 * ICC_sd**2

        SEM_CI = (SEM - 1.96 * np.sqrt(var_SEM), SEM + 1.96 * np.sqrt(var_SEM))

        # SDC 95% CI
        SDC_low = 1.96 * np.sqrt(2) * SEM_CI[0]
        SDC_high = 1.96 * np.sqrt(2) * SEM_CI[1]
        # modiing cutoffs
        test_cutoffs[test] = SDC
        #maximal_measurement_error[test] = SDC
        # report
        new_row = pd.DataFrame([{
            'test': test_names[test],
            'type': 'SDC',
            'SDC': test_cutoffs[test],
            'SDC CI 95%': [round(SDC_low, 2), round(SDC_high, 2)],
            'ICC': icc,
            'ICC CI 95%': [round(icc_low, 3), round(icc_high,3)],
            'ICC p-value': pval
        }])

        sdc_report = pd.concat([sdc_report if not sdc_report.empty else None, new_row], ignore_index=True)

        axes[0].axvline(x=-SDC, c='r', ls='--', label=f'SDC = {SDC:.2f}')
        axes[0].axvline(x=SDC, c='r', ls='--')
        axes[0].set_xlabel(f"Value difference, {units[test]}")
        axes[0].legend(fontsize='small')
        sdc_cis.update({test:  [round(SDC_low, 2), round(SDC_high, 2)]})

    sdc_export.update({test: test_cutoffs[test]})

    # plot
    plt.show()

    print('\n')

with open('sdc.pickle', 'wb') as f:
    pickle.dump(sdc_export, f)

print('## Summary\n')

display(Markdown(sdc_report.to_latex(index=False, float_format="%.3f", escape=True, caption="Estimated SDC with estimated ICCs", position='H')))
summary = (
    filtered_data.groupby("itemid")["value"]
      .agg(
        n = 'size',
        mean = "mean",
        std  = "std",
        q1   = lambda s: s.quantile(0.25),
        median = "median",
        q3   = lambda s: s.quantile(0.75),
        min  = "min",
        max  = "max"
      )
      .reset_index()
)

# combine mean and std
summary["mean ± std"] = summary["mean"].apply(lambda x: f'{x:.3g}') + " ± " + summary["std"].apply(lambda x: f'{x:.3g}')

# add SDC columns
summary["SDC"] = summary["itemid"].map(sdc_export)
summary["SDC 95\% CI"] = summary["itemid"].map(sdc_cis)

# map itemid -> test name
summary["test"] = summary["itemid"].map(test_names) +', ' + summary["itemid"].map(units)
summary["n"] = summary['n'].apply(lambda x: f"{x}") 

summary = summary.drop(columns=["mean", "std"])


# drop original id
summary = summary.drop(columns=["itemid"])

# sort alphabetically by test name
summary = summary.sort_values("test", key=lambda x: x.str.lower()).reset_index(drop=True)
# reorder columns
summary = summary.loc[:, ["test", 'n', "mean ± std", "q1", "median", "q3", "min", "max", "SDC", "SDC 95\% CI"]]
latex_table = summary.to_latex(index=False, escape=False, float_format="%.4g", caption='Statistical description of all analytes with estimated SDCs', position='H')
latex_table = latex_table.replace("\\begin{tabular}", "\\small\n\\begin{tabular}")

display(Markdown(latex_table))
summary.to_csv('sdc_table.csv')





# %% [markdown]
# # Time change
# In this section we present plots illustrates the dependence of differences in test results on time. We constructed three types of plots:  
# 1. data within three different time windows (0–24 hours, 24–72 hours, and 72–168 hours),  
# 2. only the first acquired measurement and the first routine measurement,  
# 3. results from the first time window stratified into quartiles according to the first measurement value.  
#
# The plots were constructed as follows. For each test type, all data pairs (same patient, same test) within the given time window (e.g., 0–24 hours) were identified, and differences in values and test 
# times were calculated. Pairs with time differences within the defined fitting/plotting window (e.g., 12) were retained.  
#
# Differences between paired measurements were calculated and plotted. Plot ranges were set to 1.1 times the 99th percentile of the absolute values to prevent rescaling driven by extreme outliers,
# although these outliers were still included in subsequent analyses. All data points were plotted to visualize outlying values.
# Due to the high density of points around the central trend and resulting oversaturation, point colors were adjusted using a density approximation estimated by a Gaussian kernel method.
# Because this density has no direct quantitative interpretation, no color scale was included in the plots.  
#
#
#
# ## Analyte dynamics model
# We fitted the following probability model to the data. We assumed that the difference from the first measurement could be described by a probability density function with a time-varying location 
# parameter $\mu_t$ (for symmetric distributions such as the normal, this corresponds to the mean) and a time-varying scaling factor $\sigma_t'$ characterizing the increasing spread of results over time 
# (equivalent to the standard deviation for the normal distribution).  
#
# The time dependence of the location parameter was assumed to be:
# $$ \mu_t = k \cdot t + d, $$
# where $k$ and $d$ are constants. Constant $d$ is included because for asymetrical distribution location does not correspond to mean value so this allows the model to shift mean value to zero.  
#
# The time-dependent scale factor was modeled as:
# $$ \sigma_t' = \frac{c \cdot t}{\tau + t}, $$
# where $c$ and $\tau$ are constants.  
#
# We further assumed a constant baseline measurement error $\sigma_m$, combined with the scaling factor as:
# $$ \sigma_t = \sqrt{\sigma_t'^2 + \sigma_m^2}. $$  
#
# A skewed Student’s t-distribution was used because the data appeared heavy-tailed and skewed for most tests. This introduced two additional distributional parameters, $a$ and $b$, which were held 
# constant. In total, seven parameters were optimized: $k$, $d$, $c$, $\tau$, $\sigma_m$, $a$, and $b$. Optimization was performed by minimizing the negative log-likelihood (NLL).
# We used Python library scikit-learn for implementation of the model. Boundaries for parameters were set as follows: $k$ was unrestricted, $c>0$, $\tau \in (0, 168)$, $\sigma_m \in > \sigma_t$, $a,b \in (0.6, 50)$  
#
# The first three parameters were restricted based on physiological plausibility. Parameters of the distribution were constrained to a minimum of 0.6, as lower values resulted in numerical instability.
# The measurement error parameter $\sigma_t$ was set to its minimal value to avoid extremely low probabilities near time zero.
# These values were estimated based on the precision of the stored data (e.g., Albumin and Sodium without decimals) and the data used for SDC calculation. You can find these values in table @tbl-minimal_measurement_error
# With the exception of Albumin, these restrictions did not substantially affect the results.

#
#Mean values and the 2.5 and 
# 97.5 percentiles of the fitted distribution were plotted against the data.  
# Since differences are plotted relative to the first measurement, these percentile values are analogous to Limits of Agreement in Bland–Altman plots changing over time.
#
# As there is no simple formal test for overall goodness-of-fit, we employed probability–probability (PP) plots, probability integral transform (PIT) histograms, and histograms of standardized values 
# $x'$, defined as:
# $$ x'(x,t) = \frac{x(t) - \mu_t(t)}{\sigma_t(t)}. $$
#
# Step-like patterns in PP plots or “teeth-like” structures in PIT histograms may arise from the discrete nature of laboratory data (e.g., sodium values often reported without decimals).  
#
# Shapiro–Wilk and D’Agostino tests for normality were performed on the standardized values. Coverage of the 95% confidence interval (percentage of points within the estimated 95% limits of agreement) 
# was also calculated. These numerical characteristics, together with the model’s NLL, are reported in the tables.


# %%
#| echo: false
#| output: asis
hours_to_table = [1, 2, 4, 8]

# Combine dictionaries into DataFrame
df = pd.DataFrame({
    'Test': [test_names[k] for k in test_names],
    'Unit': [units[k] for k in test_names],
    'Minimal measurement error': [minimal_measurement_error[k] for k in test_names]
})

# Optional: sort alphabetically by Test
df = df.sort_values('Test').reset_index(drop=True)

# Convert to LaTeX
latex_table = df.to_latex(
    index=False,
    escape=False,
    column_format='lll',
    float_format='%.2g',
    caption="Minimal measurement error for included laboratory tests.",
    label="tbl-minimal_measurement_error"
)


display(Markdown(latex_table))


time_to_cid = []
time_to_routine = []

#print(f'\n## Analyte plots\n ') 
for test in list_of_tests:
#test_names = {3024561: 'Albumin', 3003458: 'Phosphate', 3006140: 'Bilirubin', 3012095: 'Magnesium', 3032080: 'INR',
#              43534077: 'Urea', 40762351: 'Hemoglobin', 3020564: 'Creatinine', 3010813: 'WBC', 3007461: 'Platelets', 3035995: 'ALP', 3015377: 'Calcium', 3000285: 'Sodium', 3020460: 'CRP', 42869588: 'Hematocrit'}
#for test in [3024561, 40762351]:
#for test in []:

    print(f'\n## {test_names[test]}\n ') 
    #print('\n### multilinear regression\n ')


    # for table to compate 1. time window, admission to routine and quartiles
    individual_test_comp = []

    #print('\n#### Plots\n')
    #print("\n")

    data_to_process = filtered_data[filtered_data['itemid'] == test][:max_data]
    data_to_process['first_value'] = data_to_process.groupby('stayid')['value'].transform('first')
    data_pairs = make_pairs(data_to_process, windows_start=[0, 24, 72], windows_length=[24, 48, 96])

    models = make_fit_and_density_plot(data_pairs,
                  l1=f"all data",
                  title=f'{test_names[test]}',
                     measurement_std=minimal_measurement_error[test],
                    cutoff=test_cutoffs[test],
                   label=f"Plot of {test_names[test]} comparing dynamics in three different phases of stay. Time scales are different for each plot.",
                                      unit=units[test])
    print('\n')
    #breakpoint()

    display(Markdown(generate_model_evaluation_table(models, model_names=['Early', 'Intermediate', 'Late'], test=test)))

    

    # First to first routine
    lowest_charttime_rows = data_to_process.loc[data_to_process.groupby('stayid')['charttime'].idxmin()]
    within_2h_rows = data_to_process[(data_to_process['clock_time'] >= -2) & (data_to_process['clock_time'] <= 2)]

    admission_routine = pd.concat([lowest_charttime_rows, within_2h_rows], axis = 0).drop_duplicates()
     
    model_routine = make_fit_and_density_plot(admission_routine,
                  l1=f"adm to routine",
                  title=f'{test_names[test]}',
                    unit = units[test],
                     measurement_std=minimal_measurement_error[test],
                    cutoff=test_cutoffs[test],
                    times=['charttime'], window_starts=[0],
                    label=f"Plot demonstrating the dynamics when only first measurement and first routine measurement are included.")
    print('\n')
                    
    quartile_models, quantiles = plot_quartils_density(data_to_process, measurement_std=minimal_measurement_error[test], cutoff=test_cutoffs[test], title= f'{test_names[test]}',
                                                      label=f'Plot of {test_names[test]} comparing dynamics of values stratified to quartiles by measured value.', unit=units[test])

    
    # Row with 
    new_row = {'Data': 'All'}
    if models[0] is not None:
        new_row['quartile range'] = ''
        for h in hours_to_table:
            new_row[f'{h}h'] = models[0].ci_max(h)
        new_row['Time to SDC, h'] = models[0].x_for_confidence_exceeds_abs(test_cutoffs[test])
    individual_test_comp.append(new_row)
    
    new_row = {'Data': 'First and routine measurement'}
    if model_routine[0] is not None:
        new_row['quartile range'] = ''
        for h in hours_to_table:
            new_row[f'{h}h'] = model_routine[0].ci_max(h)
        new_row['Time to SDC, h'] = model_routine[0].x_for_confidence_exceeds_abs(test_cutoffs[test])
    individual_test_comp.append(new_row)


    
    for i, qmodel in enumerate(quartile_models):
        new_row = {'Data': f'{i+1}. quartile'}
        new_row['quartile range'] = quantiles[i]
        if qmodel is not None:
            for h in hours_to_table:
                new_row[f'{h}h'] = qmodel.ci_max(h)
            new_row['Time to SDC, h'] = qmodel.x_for_confidence_exceeds_abs(test_cutoffs[test])
        individual_test_comp.append(new_row)

    # Hessian paramaters error estimation, replaced by bootstrap
    #model_report= []

    #for m in model:
    #    params, uncertainty = '', ''
    #    for a in m.params: params += f' {a:.2g}'
    #    for a in m.get_params_uncertainty(): uncertainty += f'  {a:.2g}'
    #    model_report.append({'type': m.model_type, 'params': params, 'std': uncertainty})


    #model_report = pd.DataFrame(model_report)

    #display(Markdown(model_report.to_latex(index=False, escape=True, float_format="%.2f")))


    times_df = pd.DataFrame(individual_test_comp)

    display(Markdown(times_df.to_latex(index=False, float_format="%.2f",
        caption=f'Comparison of estimated percetile values (2.5th/97/th) in given times (in {units[test]}) and times to SDC in early phase.' )))


# %% [markdown]
# # Bootstraping results
# In this section, we present results of bootstrapping. We used bootstrapping to estimate uncertainties of model parameters, as analytical calculation would be complicated. We performed 500 iterations, using a maximum of 100,000 data points per iteration to achieve reasonable computation time. The density of data points is still considered sufficient for the selected model.
#
# Bootstrapping results provide estimates of model stability. Parameters of the model show low standard deviations.
# Parameters with the highest standard deviations are $c$ and $\tau$. It is worth noting that these particular parameters affect the calculation of LoA the most.
# Parameter $c$ can be interpreted as the value of the asymptote, and $\tau$ as the characteristic time to achieve it. These values are considerably high for ALP, bilirubin, CRP, creatinine, hemoglobin, INR, and platelets.
# However, the corresponding estimated LoA at set times (1h, 2h, 4h, 8h) do not show comparable standard deviations. This may be because shifts in $c$ and $\tau$ compensate for each other.


# %%
#| echo: false
#| output: asis
hours_to_table = [1, 2, 4, 8]
file_path_bootstrap = './bootstrap_results_12.parquet'

if os.path.exists(file_path_bootstrap):
    
    # Load the bootstrap results
    df = pd.read_parquet(file_path_bootstrap)

    # Calculate mean and std for each test and time_window
    df = df.rename(columns={'time_window': 'time window'})



    #cols_to_show = ['1h', '2h', '3h', '4h', '5h', '6h', 'sdc']
    cols_to_show = []
    params_cols = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
    calculation_model = ProbabilityModel(None, None, model_type='kinetic_jfskewt')
    for h in hours_to_table:
        col = f"{h}h"
        cols_to_show.append(col)
        #breakpoint()
        df[col] = df[params_cols].apply(lambda x: ProbabilityModel(None, None, params=x.to_list(), model_type='kinetic_jfskewt').ci_max(h), axis=1)

    df['sdc'] = df.apply(lambda x: ProbabilityModel(None, None, params=x[params_cols].to_list(),
                                                    model_type='kinetic_jfskewt').x_for_confidence_exceeds_abs(test_cutoffs[x['test']]) if test_cutoffs[x['test']] != None else np.nan, axis=1)

    cols_to_show.append('sdc')

    df_param = df.copy()
    df['test'] = df['test'].map(test_names).fillna('') + ', ' + df['test'].map(units).fillna('')
    df = df.sort_values(by=['test', 'time window'], key=lambda col: col.str.lower())
    #breakpoint()
    
    df_std = df.groupby(['test', 'time window'])[cols_to_show].std().reset_index()
    df_std = df_std.sort_values(by=['test', 'time window'], key=lambda col: col.str.lower())
    df_full = df[df['note'] == 'full']
    df_full = df_full.set_index(['test', 'time window'])



    latex_table = df_full[cols_to_show].to_latex(
        multirow=True,          # keep multirow index
        multicolumn=True,       # keep multi-level columns
        index=True,             # show row index
        escape=False,           # don't escape special LaTeX chars
        float_format='%.3g',
        caption="Calculated values of LoA for different times and time for LoA to exceed the SDC",
        label="tab:multiindex_summary"
    )
    df_full[cols_to_show].to_csv('time_table.csv')

    # Print or save the LaTeX table
    display(Markdown(latex_table))
    
    df_std = df_std.set_index(['test', 'time window'])
    df_both = df_full.copy()
    def fmt_with_std(val, std):
        # Handle zero std
        if std == 0 or np.isnan(std):
            return f"{val:g} ± {std:g}"
        
        # Determine number of decimal places needed
        # first non-zero digit position of std
        dec = -int(np.floor(np.log10(abs(std))))
        dec = max(dec, 0)  # no negative decimals
        breakpoint()

        return f"{val:.{dec}f} ± {std:.{dec}f}"
    #breakpoint()

    for col in cols_to_show:
        df_both[col] = [
            fmt_with_std(v, s)
            for v, s in zip(df_full[col], df_std[col])
        ]

    #for col in cols_to_show:
    #    df_both[col] = df_full[col].apply(lambda x: f"{x:.2f}") + ' ± ' + df_std[col].apply(lambda x: f"{x:.2f}")
    df_both[cols_to_show].to_csv('boot.csv')

    latex_table = df_both[cols_to_show].to_latex(
        multirow=True,          # keep multirow index
        multicolumn=True,       # keep multi-level columns
        index=True,             # show row index
        escape=False,           # don't escape special LaTeX chars
        float_format='%.3g',
        caption="Calculated values of LoA for different times and time for LoA to exceed the SDC. Data are show in following format: value etimated on full datase ± standard deviation from bootstraping.",
        label="tab:multiindex_summary"
    )
    
    latex_table = latex_table.replace("\\begin{tabular}", "\\footnotesize\n\\begin{tabular}")
    display(Markdown(latex_table))
    print('\n')


    df_param['test'] = df_param['test'].map(test_names)
    df_param = df_param.sort_values(by=['test', 'time window'], key=lambda col: col.str.lower())
    agg_df2 = df_param.groupby(['test', 'time window'])[['p1', 'p7', 'p2', 'p3', 'p4', 'p5', 'p6']].agg(['mean', 'std']).reset_index()
    #rename_parameters = {'p1': '$k$', 'p2': '$c$', 'p3': '$\\tau$', 'p4': '$\sigma_m$', 'p5': '$a$', 'p6': '$b$'} 






    # Flatten MultiIndex columns
    agg_df2.columns = ['test', 'time window',
                       'p1_mean', 'p1_std',
                       'p7_mean', 'p7_std',
                       'p2_mean', 'p2_std',
                       'p3_mean', 'p3_std',
                       'p4_mean', 'p4_std',
                       'p5_mean', 'p5_std',
                       'p6_mean', 'p6_std']

    # Prepare table rows
    table2_rows = []
    for test, group in agg_df2.groupby('test'):
        group = group.sort_values('time window')
        first_row = True
        for _, row in group.iterrows():
            # Format each pX
            p_strs = []
            for p in ['p1', 'p7', 'p2', 'p3', 'p4', 'p5', 'p6']:
                mean = row[f'{p}_mean']
                std  = row[f'{p}_std']
                p_strs.append(f"{mean:.2f} ± {std:.2f}")

            if first_row:
                table2_rows.append(f"\\multirow{{{len(group)}}}{{*}}{{{test}}} & {row['time window']} & {' & '.join(p_strs)} \\\\")
                first_row = False
            else:
                table2_rows.append(f"& {row['time window']}h & {' & '.join(p_strs)} \\\\")
        table2_rows.append("\\midrule")


   # Assemble LaTeX table 2
    latex_table2 = r"""\begin{table}[H]
    \caption{Calculated values model parameters from bootstraping. Data are show in following format: value etimated on full datase ± standard deviation from bootstraping.}
    \tiny
    \begin{tabular}{l l c c c c c c c}
    \toprule
    Test & Phase & $k$ & d & $c$ & $\tau$ & $\sigma_m$ & $a$ & $b$ \\
    \midrule    
    """ + "\n".join(table2_rows) + """\n\\bottomrule\n\\end{tabular}\n\\end{table}"""
    display(Markdown(latex_table2))
else:
    print(f'File {file_path} not found')



#%%[markdown]
# # References
