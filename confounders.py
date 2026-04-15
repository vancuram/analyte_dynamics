#%% [markdown]
# ---
# title: "How long can I trust laboratory results in critically ill patients: a large-scale  study on temporal validity of laboratory analytes"
# subtitle: "Confounders" 
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
# This supplementary file contains covariant matrices for all analytes and plot of subgroups of patients separated by different confounders. To measure the change of analyte over time we used two parameters:
# avarage rate (avarage of $x/t$) and avarage variance (root of average of $x^2/t^2$). We calculated Spearman correlation coefficicent of these parameters with all included confounders,
# We also plotted the change of analyte over time for two groups of patients divided by median of included counfounders. Only data from early time window (0-24h) was used for this analysis.

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
from matplotlib import colors, cm
import sys 
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

def plt_to_markdown(pltx, name=None, label=None, pformat='jpg'):
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

    out_str = f"""![{label}]({name_adres} "{label}")"""
    #print(out_str)
    #print(out_str)
    pltx.savefig(name_adres, format=pformat, dpi=120)
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

class ProbabilityModelBackUp:
    def __init__(self, x, y, params=None):
        self.x = x
        self.y = y
        self.params = params
        self.model_type = None  # 'kinetic_gaussian', 'kinetic_studentt', or 'kinetic_skewnorm'

    def calculate_mean_scale(self, t):
        if self.params is None or self.model_type is None:
            raise RuntimeError("Model not fitted. Call `fit()` first.")

        a = self.params[0]
        c = self.params[1]
        tau = self.params[2]
        measurement = self.params[3]

        mu = a * t
        sigma = np.sqrt((c * t / (tau + t))**2 + measurement**2)
        sigma = np.clip(sigma, 1e-6, None)
        return mu, sigma

    # --- Gaussian kinetics model ---
    def neg_log_likelihood_kinetics(self, params, t, delta):
        a, c, tau, measurement = params
        mu_t = a * t 
        sigma_t = np.sqrt((c * t / (tau + t))**2 + measurement**2)
        sigma_t = np.clip(sigma_t, 1e-6, None)
        nll = 0.5 * np.sum(
            np.log(2 * np.pi * sigma_t**2) + ((delta - mu_t)**2) / (sigma_t**2)
        )
        return nll

    # --- Student's t kinetics model ---
    def neg_log_likelihood_kinetics_t(self, params, x, delta):
        from scipy.stats import t
        a, c, tau, measurement, nu = params
        mu_t = a * x 
        sigma_t = np.sqrt((c * x / (tau + x))**2 + measurement**2)
        sigma_t = np.clip(sigma_t, 1e-6, None)
        log_pdf = t.logpdf(delta, df=nu, loc=mu_t, scale=sigma_t)
        nll = -np.sum(log_pdf)
        return nll

    # --- Skew-normal kinetics model ---
    def neg_log_likelihood_kinetics_skewnorm(self, params, x, delta):
        a, c, tau, measurement, alpha = params
        mu_t = a * x 
        sigma_t = np.sqrt((c * x / (tau + x))**2 + measurement**2)
        sigma_t = np.clip(sigma_t, 1e-6, None)
        log_pdf = skewnorm.logpdf(delta, alpha, loc=mu_t, scale=sigma_t)
        nll = -np.sum(log_pdf)
        return nll

        # --- Skew-t kinetics model ---
    def neg_log_likelihood_kinetics_jfskewt(self, params, x, delta):
        a, c, tau, measurement, ta, tb = params
        mu_t = a * x
        sigma_t = np.sqrt((c * x / (tau + x))**2 + measurement**2)
        sigma_t = np.clip(sigma_t, 1e-6, np.inf)
        log_pdf = jf_skew_t.logpdf(delta, ta, tb, loc=mu_t, scale=sigma_t)
        nll = -np.sum(log_pdf)
        return nll

        # --- Skew-t kinetics model ---
    def neg_log_likelihood_kinetics_skewt(self, params, x, delta):
        a, c, tau, measurement, a, df = params
        mu_t = a * x
        sigma_t = np.sqrt((c * x / (tau + x))**2 + measurement**2)
        sigma_t = np.clip(sigma_t, 1e-6, np.inf)
        log_pdf = skewt.logpdf(delta, a, df, loc=mu_t, scale=sigma_t)
        nll = -np.sum(log_pdf)
        return nll

    def fit(self, x=None, y=None, bounds=None, measurement_std=None, method='best'):
        if x is None: x = self.x
        if y is None: y = self.y

        # Initial guesses
        init_gaussian = [0, 1.0, 1.0, measurement_std / 2]
        init_studentt = [0, 1.0, 1.0, measurement_std / 2, 5.0]
        init_skewnorm = [0, 1.0, 1.0, measurement_std / 2, 0.0]  # alpha init at 0
        init_jfskewt = [0, 1.0, 1.0, measurement_std / 2, 2, 2.0]  # df=5.0, alpha=0.0
        init_skewt = [0, 1.0, 1.0, measurement_std / 2, 0, 3.0]  #a=1 df=5.0
        

        
        # Bounds        
        bounds_jfskewt = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0),
            (measurement_std/5, measurement_std*2), (-50, 50), (-30, 30)
        ]
        bounds_skewt = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0),
            (measurement_std/5, measurement_std*2), (-50, 50), (0, 30)
        ]
        bounds_gaussian = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0), (measurement_std/5, measurement_std*2)
        ]
        
        bounds_studentt = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0), (measurement_std/5, measurement_std*2), (0, 30)
        ]
        bounds_skewnorm = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0), (measurement_std/5, measurement_std*2), (-20, 20)
        ]

        res_gaussian = minimize(
            self.neg_log_likelihood_kinetics, init_gaussian, args=(x, y),
            bounds=bounds_gaussian, method='L-BFGS-B', options={'maxiter': 10000}
        )
        #print('gauss optimized')
        res_studentt = minimize(
            self.neg_log_likelihood_kinetics_t, init_studentt, args=(x, y),
            bounds=bounds_studentt, method='L-BFGS-B', options={'maxiter': 10000}

        )
        #print('t optimized')
        #res_skewnorm = minimize(
        #    self.neg_log_likelihood_kinetics_skewnorm, init_skewnorm, args=(x, y),
        #    bounds=bounds_skewnorm, method='L-BFGS-B', options={'maxiter': 10000}

        #)
        #print('skewnorm optimized')
        
        res_jfskewt = minimize(
            self.neg_log_likelihood_kinetics_jfskewt, init_jfskewt, args=(x, y),
            bounds=bounds_skewt, method='L-BFGS-B', options={'maxiter': 10000}

        )
        #print('jfskewt optimized')
        #res_skewt = minimize(
        #    self.neg_log_likelihood_kinetics_skewt, init_skewt, args=(x, y),
        #    bounds=bounds_skewt, method='L-BFGS-B', options={'maxiter': 100}

        #)
        #print('skewt optimized')

        # Choose model
        if method == 'best':
            nlls = {
            'kinetic_gaussian': res_gaussian.fun,
            'kinetic_studentt': res_studentt.fun,
            #'kinetic_skewnorm': res_skewnorm.fun,
            'kinetic_jfskewt': res_jfskewt.fun,
            #'kinetic_skewt': res_skewt.fun
            }
            best_model = min(nlls, key=nlls.get) if method == 'best' else f"kinetic_{method}"

        else:
            best_model = f"kinetic_{method}"

        #print(f'winner: {best_model}')

        if best_model == 'kinetic_gaussian':
            self.params = res_gaussian.x
        elif best_model == 'kinetic_studentt':
            self.params = res_studentt.x
        elif best_model == 'kinetic_skewnorm':
            self.params = res_skewnorm.x
        elif best_model == 'kinetic_jfskewt':
            self.params = res_jfskewt.x
        elif best_model == 'kinetic_skewt':
            self.params = res_skewt.x

        self.model_type = best_model
        return self

    def get_params_uncertainty(self):
        if self.params is None or self.model_type is None:
            raise RuntimeError("Model not fitted. Call `fit()` first.")

        nll_func = lambda params: self.nll(params=params)

        hessian_nll = Hessian(nll_func)
        hess_val = hessian_nll(self.params)

        try:
            cov_matrix = np.linalg.inv(hess_val)
            standard_errors = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            standard_errors = np.full(len(self.params), np.nan)
            print("Hessian is singular. Standard errors could not be computed.")

        return standard_errors

    def predict(self, x_vec):
        if self.params is None or self.model_type is None:
            raise RuntimeError("Model not fitted. Call `fit()` first.")
        
        mu, sigma = self.calculate_mean_scale(x_vec)
        means = self.mean(x_vec)
        lower, upper = self.ci(x_vec)
        return means, lower, upper

    def nll(self, params=None, x=None, y=None):
        if x is None: x = self.x
        if y is None: y = self.y
        if params is None:
            params = self.params

        mu, sigma = self.calculate_mean_scale(x)

        if self.model_type == 'kinetic_gaussian':
            return 0.5 * np.sum(np.log(2 * np.pi * sigma**2) + ((y - mu)**2) / (sigma**2))

        elif self.model_type == 'kinetic_studentt':
            from scipy.stats import t
            _, _, _, _, nu = params
            return -np.sum(t.logpdf(y, df=nu, loc=mu, scale=sigma))

        elif self.model_type == 'kinetic_skewnorm':
            _, _, _, _, alpha = params
            return -np.sum(skewnorm.logpdf(y, alpha, loc=mu, scale=sigma))

        elif self.model_type == 'kinetic_jfskewt':
            _, _, _, _, ta, tb = params
            return -np.sum(jf_skew_t.logpdf(y, ta, tb, loc=mu, scale=sigma))

        elif self.model_type == 'kinetic_skewt':
            _, _, _, _, a, df = params
            return -np.sum(skewt.logpdf(y, a, df, loc=mu, scale=sigma))

        else:
            raise ValueError("Unknown model type.")

    def cdf(self, x_eval, y_eval):
        mu, sigma = self.calculate_mean_scale(x_eval)

        if self.model_type == 'kinetic_gaussian':
            from scipy.stats import norm
            return norm.cdf(y_eval, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_studentt':
            from scipy.stats import t
            _, _, _, _, nu = self.params
            return t.cdf(y_eval, df=nu, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewnorm':
            _, _, _, _, alpha = self.params
            return skewnorm.cdf(y_eval, alpha, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_jfskewt':
            _, _, _, _, ta, tb = self.params
            return jf_skew_t.cdf(y_eval, ta, tb, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewt':
            _, _, _, _, a, df = self.params
            return skewt.cdf(y_eval, a, df, loc=mu, scale=sigma)

        else:
            raise ValueError("Unknown model type.")

    def mean(self, x_eval):
        """
        Calculate the mean value at x_eval for the fitted model.
        
        Parameters
        ----------
        x_eval : array-like or float
            The x value(s) to evaluate.
        
        Returns
        -------
        mu : array-like or float
            The mean value(s) at x_eval.
        """
        mu, sigma = self.calculate_mean_scale(x_eval)
        
        if self.model_type == 'kinetic_gaussian':
            from scipy.stats import norm
            result = norm.mean(loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_studentt':
            _, _, _, _, nu = self.params
            from scipy.stats import t
            result = t.mean(df=nu, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewnorm':
            _, _, _, _, alpha_param = self.params
            from scipy.stats import skewnorm
            result = skewnorm.mean(alpha_param, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_jfskewt':
            _, _, _, _, ta, tb = self.params
            result = jf_skew_t.mean(ta, tb, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewt':
            _, _, _, _, a, df = self.params
            result = skewt.stats(a, df, loc=mu, scale=sigma, moments='m')

        else:
            raise ValueError("Unknown model type.")
        
        return result

    def ci(self, x_eval, alpha=0.05):
        mu, sigma = self.calculate_mean_scale(x_eval)
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        if self.model_type == 'kinetic_gaussian':
            from scipy.stats import norm
            lower = norm.ppf(lower_q, loc=mu, scale=sigma)
            upper = norm.ppf(upper_q, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_studentt':
            _, _, _, _, nu = self.params
            from scipy.stats import t
            lower = t.ppf(lower_q, df=nu, loc=mu, scale=sigma)
            upper = t.ppf(upper_q, df=nu, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewnorm':
            _, _, _, _, alpha_param = self.params
            lower = skewnorm.ppf(lower_q, alpha_param, loc=mu, scale=sigma)
            upper = skewnorm.ppf(upper_q, alpha_param, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_jfskewt':
            _, _, _, _, ta, tb = self.params
            lower = jf_skew_t.ppf(lower_q, ta, tb, loc=mu, scale=sigma)
            upper = jf_skew_t.ppf(upper_q, ta, tb, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewt':
            _, _, _, _, a, df = self.params
            lower = skewt.ppf(lower_q, a, df, loc=mu, scale=sigma)
            upper = skewt.ppf(upper_q, a, df, loc=mu, scale=sigma)

        else:
            raise ValueError("Unknown model type.")

        return lower, upper

    def ppf(self, x_eval, q_eval):
        mu, sigma = self.calculate_mean_scale(x_eval)

        if self.model_type == 'kinetic_gaussian':
            from scipy.stats import norm
            return norm.ppf(q_eval, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_studentt':
            from scipy.stats import t
            _, _, _, _, nu = self.params
            return t.ppf(q_eval, df=nu, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewnorm':
            _, _, _, _, alpha = self.params
            from scipy.stats import skewnorm
            return skewnorm.ppf(q_eval, alpha, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_jfskewt':
            _, _, _, _, ta, tb = self.params
            # Assuming jf_skew_t is your custom skew-t implementation with ppf method
            return jf_skew_t.ppf(q_eval, ta, tb, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewt':
            _, _, _, _, a, df = self.params
            # Assuming jf_skew_t is your custom skew-t implementation with ppf method
            return jf_skew_t.ppf(q_eval, a, df, loc=mu, scale=sigma)

        else:
            raise ValueError("Unknown model type.")


    def model_equation(self):
        if self.params is None or self.model_type is None:
            return "Model not fitted or unknown model type."

        p = self.params
        
        if self.model_type == 'kinetic_jfskewt':
            a, c, tau, measurement, ta, tb = self.params
            return (
                f"$\mu(t) = {a:.3f} \cdot t$\n"
                f"$\sigma(t) = \sqrt{{({c:.3f} \cdot t / ({tau:.3f} + t))^2 + {measurement:.3f}^2}}$\n"
                f"$a ={ta:.3f}$, b={tb:.3f}$"
            )

        if self.model_type == 'kinetic_skewt':
            a, c, tau, measurement, a, df = self.params
            return (
                f"$\mu(t) = {a:.3f} \cdot t$\n"
                f"$\sigma(t) = \sqrt{{({c:.3f} \cdot t / ({tau:.3f} + t))^2 + {measurement:.3f}^2}}$\n"
                f"$a ={a:.3f}, df={df:.3f}$"
            )

        elif self.model_type == 'kinetic_gaussian':
            a, c, tau, measurement = p
            return (
                f"$\mu(t) = {a:.3f} \cdot t$\n"
                f"$\sigma(t) = \sqrt{{({c:.3f} \\cdot t / ({tau:.3f} + t))^2 + {measurement:.3f}^2}}$"
            )

        elif self.model_type == 'kinetic_studentt':
            a, c, tau, measurement, nu = p
            return (
                f"$\mu(t) = {a:.3f} \cdot t$\n"
                f"$\sigma(t) = \sqrt{{({c:.3f} \cdot t / ({tau:.3f} + t))^2 + {measurement:.3f}^2}}$\n"
                f"$\\nu = {nu:.3f}$ (degrees of freedom)"
            )

        elif self.model_type == 'kinetic_skewnorm':
            a, c, tau, measurement, alpha = p
            return (
                f"$\mu(t) = {a:.3f} \\cdot t$\n"
                f"$\sigma(t) = \sqrt{{({c:.3f} \cdot t / ({tau:.3f} + t))^2 + {measurement:.3f}^2}}$\n"
                f"$\\alpha = {alpha:.3f}$ (shape/skew parameter)"
            )
        else:
            return "Model not fitted or unknown model type."

    def x_for_confidence_exceeds_abs(self, threshold, x_bounds=(1e-6, 1e3)):
        """
        Finds x where either the lower or upper bound of the 95% CI
        first exceeds the given threshold in absolute value.

        Returns the smaller x satisfying either condition, or None if not found.
        """
        if self.params is None or self.model_type is None:
            raise RuntimeError("Model not fitted. Call `fit()` first.")

        from scipy.optimize import root_scalar

        def func_lower(x):
            x = np.array([x])
            try:
                lower, _ = self.ci(x, alpha=0.05)
                val = np.abs(lower[0]) - threshold
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception as e:
                print(f"Exception in func_lower at x={x}: {e}")
                return 1e10

        def func_upper(x):
            x = np.array([x])
            try:
                _, upper = self.ci(x, alpha=0.05)
                val = np.abs(upper[0]) - threshold
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception as e:
                print(f"Exception in func_upper at x={x}: {e}")
                return 1e10

        # Solve for lower CI bound
        x_lower = None
        if func_lower(x_bounds[0]) >= 0:
            x_lower = x_bounds[0]
        elif func_lower(x_bounds[1]) >= 0:
            sol_lower = root_scalar(func_lower, bracket=x_bounds, method='brentq')
            if sol_lower.converged:
                x_lower = sol_lower.root

        # Solve for upper CI bound
        x_upper = None
        if func_upper(x_bounds[0]) >= 0:
            x_upper = x_bounds[0]
        elif func_upper(x_bounds[1]) >= 0:
            sol_upper = root_scalar(func_upper, bracket=x_bounds, method='brentq')
            if sol_upper.converged:
                x_upper = sol_upper.root

        # Return the smaller of the two valid solutions
        solutions = [x for x in [x_lower, x_upper] if x is not None]

        if not solutions:
            return None
        else:
            return min(solutions)

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

        if model is not None:
            eval_res = evaluate_probability_model(model, xi, yi, plot=True)
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
    latex_table = formatted_df.to_latex(index=False, escape=True, caption="Model Evaluation Results",
                                        label="tab:model_evaluation", column_format="lcccc")

    return latex_table

def evaluate_probability_model(model, x=None, y=None, plot=True):
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
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

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
        plt_to_markdown(plt)

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

def make_fit_and_density_plot(df1, df2, l1='less severe', l2='more severe', title='Some plot', unit='',
               window_starts=[0], window_length=[24], plot_lengths=[8], number_of_bins=12, morning_tolerance = 1.5,
               times=['charttime'], minimal_set_size=10, measurement_std=None,
                 minimal_count_int=3, fit_minimal_count=10, cutoff=None, titles=['Early', 'Intermediate', 'Late'], label=None):
# minimal_count_int is interval for fitting model (preventing fitting if there is not enough


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

        b = df2[(df2['charttime'] >= start)
                & (df2['charttime'] < start + window_length[indx])].copy()
            
        b = recalculate_relative_to_first(b)
        b = b[b['time'] < plot_length]


        if max_range < b['difference'].abs().quantile(0.99):
            max_range = b['difference'].abs().quantile(0.99)


        #axes[indx].scatter(a['time'] + t, a['difference'], c='g', label=l1, alpha=a_alpha, s=2)

        # Making scatter 



        a['color'] = 'g'
        b['color'] = 'r'

        merged = pd.concat([a, b])

        # Mean and deviation plot
        axes[indx].text(0.02, 0.98, titles[indx] + f', $n_1$={len(a)}\n $n_2$={len(b)}' , transform=axes[indx].transAxes,
            ha='left', va='top', fontsize=12)

        sub = merged.sample(min(2000, len(a)))

        # Fit KDE on 2D subsample
        kde = gaussian_kde([sub["time"], sub["difference"]])

        # Evaluate on full data
        densities = kde([merged["time"], merged["difference"]])
        alpha =  np.array((0.3 *(1 -  (densities / densities.max()) )))
        alpha = np.clip(alpha, a_max=0.25, a_min= min(0.1, 0.3 * 2 * 100*100  / len(densities)))

        merged["alpha"] = alpha 
        merged = merged.sample(frac=1)

        #axes[indx].scatter(a['time'], a['difference'], color='g', marker='.', s=25, label=l1, alpha=max(0.05, min(1, 1000/len(a))))
        axes[indx].scatter(merged['time'], merged['difference'], c=merged['color'], marker='.', s=15, alpha=merged['alpha'])


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
            #alpha=0.2, edgecolor='#1E8925', facecolor='#6BD572')
        alpha=0.2, edgecolor='greenyellow', facecolor='#6BD572', label=l1)
        #alpha=0.2, edgecolor='g', facecolor='g')
        #axes[indx].plot(tf_zerod, m_t, c= 'white', ls='-', alpha=0.8, label=model_a.model_equation(), linewidth=2)
        axes[indx].plot(tf_zerod, sigmau, c= 'greenyellow',  alpha=0.5 )
        axes[indx].plot(tf_zerod, sigmal, c= 'greenyellow',  alpha=0.5 )
        axes[indx].plot(tf_zerod, m_t, c= 'greenyellow', alpha=0.5, ls='--', label=model_a.model_equation())
        #axes[indx].plot(tf_zerod, m_t, c= '#1E8925', alpha=0.3, label=model_a.model_equation())

        #m_t, delta_t = model_a.predict(a['time'])
        #outliers = a[(a['difference'] > m_t+delta_t) | (a['difference'] < m_t - delta_t)]
        #plt.scatter(outliers['time'] + start, outliers['difference'], marker='.', s=1, alpha=0.1)
    else:
        model_a = None
    models_a.append(model_a)

    # initialization and fitting
    if len(b[b['time'] < minimal_count_int]) > fit_minimal_count:
        model_b = ProbabilityModel(b['time'], b['difference'])
        model_b.fit(measurement_std=measurement_std, method='kinetic_jfskewt')
        #prediction
        tf = np.linspace(start, start + plot_length, 200)
        tf_zerod = tf - start
        m_t,  sigmal, sigmau = model_b.predict(tf_zerod) 
        #breakpoint()

        #axes[indx].fill_between(tf, m_t  + sigma_t, m_t - sigma_t,
        #alpha=0.2, ls='--', edgecolor='#1E8925', facecolor='#6BD572')

        axes[indx].fill_between(tf_zerod, sigmau, sigmal,
        alpha=0.2, edgecolor='r', facecolor='r', label=l2)
        axes[indx].plot(tf_zerod, m_t, c= 'orangered', ls='--', alpha=0.4, label=model_b.model_equation(), linewidth=2)

    else:
        model_b = None
    models_b.append(model_b)

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
    plt_to_markdown(plt, label=label)
    plt.close()
    return models_a, models_b

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

def plot_quartils_density(df0, x_col='time', y_col='difference', window_start=0, window_length=24, plot_lengt=8, measurement_std=None, 
            number_of_windows=10, minimal_set_size=10, cutoff=None, label=None):
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
        #ax.set_xlabel(x_col)
        #ax.set_ylabel(y_col)
        ax.grid(linestyle='--', color='lightgrey')


        if cutoff != None:
            ax.axhline(cutoff, c='grey', ls='--')
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
            alpha=0.2, edgecolor='orangered', facecolor='#6BD572')
            #axes[indx].plot(tf_zerod, m_t, c= 'white', ls='-', alpha=0.8, label=model_a.model_equation(), linewidth=2)
            ax.plot(tf_zerod, m_t, c= 'orangered', ls='--', alpha=0.7, label=model.model_equation(), linewidth=3)
            
            ax.legend(fontsize='small', loc="lower left")
        else:
            model = None

        
        models.append(model)


    plt.setp(axes, ylim=(-max_range*1.05, max_range*1.05))
    plt.suptitle("Quartiles of first value", fontsize=16)
    plt_to_markdown(plt, label=label)
    plt.close()

    return models, labels


def print_time(event):
    #print(event + (f'\n time of running:{time.time() - starttime}\n'))
    return 0

from google.cloud import bigquery




# %%
#| echo: false
#| output: asis


cash_file = "cash_lab_stability.parquet"
cash_file_path = os.path.join(DATA_DIR, cash_file)


#print(f"\n### Loading data from database\n")
# Loading Data
if os.path.exists(cash_file_path) and True:
    #print(f"FIltered data loaded from file {cash_file}.")
    filtered_data = pd.read_parquet(cash_file_path)
else:

    raise Exception('Missing data file')
    print(f" Saved data to {cash_file_path}\n")

with open('sdc.pickle', 'rb') as f:
    sdcs = pickle.load(f)



#print(f"LaTeX table saved to {output_tex}")


# %%
#| echo: false
#| output: asis

n_tests = len(list_of_tests)
n_cols = 3
n_rows = int(np.ceil(n_tests / n_cols))

# Create a custom diverging grayscale colormap
#colors = ["black", "white", "black"]  # left -> middle -> right
#cmap = LinearSegmentedColormap.from_list("custom_grey_diverging", colors)

rename_mapping = {'first_value': 'first value', 'blood_cells_volume': 'blood', 'platelets_volume': 'platelets',
                  'plasma_volume': 'plasma', 'fluid_volume_within_24h': 'fluids', 'max_crp': 'max CRP', 'albumin_dose_within_24h': 'albumin'}

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))#, layout='constrained')
axs = axs.flatten()
reg_data_list={}
for idx, test in enumerate(list_of_tests):
    #print(f'\n## {test_names[test]}\n ') 
    data_to_process = filtered_data[filtered_data['itemid'] == test][:max_data]
    data_to_process['first_value'] = data_to_process.groupby('stayid')['value'].transform('first')

    # Multilinear regression
    reg_data = data_to_process.copy()
    reg_data = reg_data[~reg_data['stayid'].str.contains('hirid')] # Removing HiRID data for regression because missing Albumin data
    reg_data = reg_data[~(reg_data['max_crp'] < 0)] # Removing records without CRP
    reg_data.sort_values(['stayid', 'charttime'])
    reg_data['first_time'] = reg_data.groupby('stayid')['charttime'].transform('first')


    reg_data['charttime'] = reg_data['charttime'] - reg_data['first_time']
    reg_data['difference'] = reg_data['value'] - reg_data['first_value']
    reg_data = reg_data[(reg_data['charttime'] != 0) & (reg_data['charttime'] < 24) ] # 24 h
    reg_data['rate'] = reg_data['difference'] / reg_data['charttime']
    reg_data['absrate'] = reg_data['rate']

    reg_data['mean rate'] = reg_data.groupby('stayid')['rate'].transform('mean')
    reg_data['rate dev.'] = reg_data.groupby('stayid')['absrate'].transform(lambda x: np.sqrt((x**2).mean()))
    reg_data['crrt'] = reg_data['crrt'].astype(int)

    reg_data = reg_data[['stayid', 'mean rate', 'rate dev.', 'first_value', 'blood_cells_volume', 'platelets_volume',
                         'plasma_volume', 'fluid_volume_within_24h', 'max_crp', 'albumin_dose_within_24h', 'crrt']]
    reg_data = reg_data.drop_duplicates()
    reg_data_list.update({test: reg_data})

    corr_data = reg_data[['mean rate', 'rate dev.', 'first_value', 'blood_cells_volume', 'platelets_volume',
                      'plasma_volume', 'fluid_volume_within_24h', 'max_crp', 'albumin_dose_within_24h', 'crrt']]

    corr_data = corr_data.rename(columns=rename_mapping)

    # Compute correlation matrix
    corr_matrix = corr_data.corr(method='spearman')

 # Identify top 3 features by max absolute correlation with mean rate or rate dev.
    candidate_features = corr_matrix.columns.difference(['mean rate', 'rate dev.'])
    corr_with_targets = corr_matrix.loc[candidate_features, ['mean rate', 'rate dev.']].abs()
    corr_with_targets['max_corr'] = corr_with_targets.max(axis=1)
    top3_features = corr_with_targets['max_corr'].sort_values(ascending=False).head(3).index.tolist()

    # Final 3x3 matrix: rows = top3_features, columns = mean rate, rate dev., first_value
    final_corr_matrix = corr_matrix.loc[top3_features, ['mean rate', 'rate dev.', 'first value']]

    # Plot gray-scale heatmap in subplot
    sns.heatmap(final_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axs[idx], cbar=False,
               annot_kws={"size": 18})
    # Increase font sizes for this Axes
    axs[idx].set_title(f"{test_names[test]}", fontsize=20, pad=5)
    axs[idx].tick_params(axis='both', labelsize=18)
    # Move x-axis ticks to the top
    #axs[idx].xaxis.set_ticks_position('top')
    #axs[idx].xaxis.set_label_position('top')

    # Rotate x-axis tick labels if needed
    axs[idx].tick_params(axis='x', rotation=45)  # or rotation=0 for no rotation

    # Rotate y-axis tick labels to horizontal
    axs[idx].tick_params(axis='y', rotation=0)

# Remove empty subplots if any
for j in range(idx + 1, len(axs)):
    fig.delaxes(axs[j])

norm = colors.Normalize(vmin=-1, vmax=1)
sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])

# Adjust layout to leave room for colorbar on the right
plt.tight_layout(rect=[0, 0, 0.92, 1])  # reserve 8% of space on right

# Create a new axis for the vertical colorbar
cbar_ax = fig.add_axes([0.94, 0.25, 0.02, 0.5])  # [left, bottom, width, height]

# Create the colorbar
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

# Style the colorbar
cbar.ax.tick_params(labelsize=14, length=8, width=1.2)  # bigger ticks and labels
cbar.outline.set_linewidth(1)
cbar.set_label('Correlation', labelpad=10, fontsize=16)  # label on sid
#plt.tight_layout(pad=3.0)  # increase pad for more spacing overall
#plt.tight_layout()
#plt.show()
plt.savefig(os.path.join('correlation_heatmaps.pdf'))
plt.close()











# [markdown]
# # Confounders
# In this part we present each individual test dependency on multiple factor.  


# %%
#| echo: false
#| output: asis




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
    label="tab:minimal_measurement_error"
)


#display(Markdown(latex_table))


time_to_cid = []
time_to_routine = []

#print(f'\n## Analyte plots\n ') 
for test in list_of_tests:
#test_names = {3024561: 'Albumin', 3003458: 'Phosphate', 3006140: 'Bilirubin', 3012095: 'Magnesium', 3032080: 'INR',
#              43534077: 'Urea', 40762351: 'Hemoglobin', 3020564: 'Creatinine', 3010813: 'WBC', 3007461: 'Platelets', 3035995: 'ALP', 3015377: 'Calcium', 3000285: 'Sodium', 3020460: 'CRP', 42869588: 'Hematocrit'}
#for test in [3024561, 40762351]:

    print(f'\n## {test_names[test]}\n ') 
    #print('\n### multilinear regression\n ')
    reg_data = reg_data_list[test]

    X = reg_data[['first_value', 'blood_cells_volume', 'platelets_volume',
                          'plasma_volume', 'fluid_volume_within_24h', 'max_crp', 'albumin_dose_within_24h', 'crrt']]
    y = reg_data['mean rate']
    y_abs = reg_data['rate dev.']

    #print('\n#### Corelation\n')
    corr_data = reg_data[['mean rate', 'rate dev.', 'first_value', 'blood_cells_volume', 'platelets_volume',
                      'plasma_volume', 'fluid_volume_within_24h', 'max_crp', 'albumin_dose_within_24h', 'crrt']]

    corr_data = corr_data.rename(columns=rename_mapping)

    # Compute correlation matrix
    corr_matrix = corr_data.corr(method='spearman')

    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt_to_markdown(plt, pformat='pdf', label=f'{test_names[test]}: correlation matrix of mean rate, rate deviation and confounders')
    plt.close()

    print('\n')


    """
    # for table to compate 1. time window, admission to routine and quartiles
    individual_test_comp = []

    print('\n#### Plots\n')
    #print("\n")


    
    # Row with 
    new_row = {'Data': 'All'}
    if models[0] is not None:
        new_row['quartile range'] = ''
        for h in hours_to_table:
            new_row[f'{h}h'] = models[0].ci_max(h)
        new_row['Time to SDC'] = models[0].x_for_confidence_exceeds_abs(test_cutoffs[test])
    individual_test_comp.append(new_row)
    
    new_row = {'Data': 'First to routine'}
    if model_routine[0] is not None:
        new_row['quartile range'] = ''
        for h in hours_to_table:
            new_row[f'{h}h'] = model_routine[0].ci_max(h)
        new_row['Time to SDC'] = model_routine[0].x_for_confidence_exceeds_abs(test_cutoffs[test])
    individual_test_comp.append(new_row)


    
    for i, qmodel in enumerate(quartile_models):
        new_row = {'Data': f'{i+1}. quartile'}
        new_row['quartile range'] = quantiles[i]
        if qmodel is not None:
            for h in hours_to_table:
                new_row[f'{h}h'] = qmodel.ci_max(h)
            new_row['Time to SDC'] = qmodel.x_for_confidence_exceeds_abs(test_cutoffs[test])
        individual_test_comp.append(new_row)


    times_df = pd.DataFrame(individual_test_comp)

    display(Markdown(times_df.to_latex(index=False, float_format="%.2f")))
    """


    data_to_process = filtered_data[filtered_data['itemid'] == test][:max_data]
    
    data_to_process['first_value'] = data_to_process.groupby('stayid')['value'].transform('first')

    #value_median = data_to_process['first_value'].median()
    #make_fit_and_plot(data_to_process[data_to_process['first_value'] <= value_median],
    #               df2=data_to_process[data_to_process['first_value'] > value_median],
    #              l1=f"=<{value_median}", l2=f'>{value_median}',
    #              title='Sepratation by median of value',
    #                 measurement_std=maximal_measurement_error[test],
    #                 cutoff=test_cutoffs[test])

    print("\n")

    make_fit_and_density_plot(data_to_process[data_to_process['fluid_volume_within_24h'] <= vol_treshold],
                   data_to_process[data_to_process['fluid_volume_within_24h'] > vol_treshold],
                  l1=f"=<{vol_treshold}ml of fluids", l2=f'>{vol_treshold}ml of fluids',
                  title='Division by volumne in first 24 hours',
                     measurement_std=minimal_measurement_error[test],
                     cutoff=test_cutoffs[test],
                     label=f'{test_names[test]}: division by fluids given in first 24 hours')

    print("\n")


    make_fit_and_density_plot(data_to_process[data_to_process['crrt'] == False],
                   df2=data_to_process[data_to_process['crrt'] == True],
                  l1=f"CRRT not used", l2=f'CRRT used',
                  title='by CRRT use',
                     measurement_std=minimal_measurement_error[test],
                     cutoff=test_cutoffs[test],
                     label=f'{test_names[test]}: division by CRRT use')

    print("\n")
    make_fit_and_density_plot(data_to_process[data_to_process['max_crp'] <= crp_treshold],
                   df2 = data_to_process[data_to_process['max_crp'] > crp_treshold],
                  l1=f"CRP =<{crp_treshold}", l2=f'CRP >{crp_treshold}',
                  title='Division by CRP in first 24 hours',
                     measurement_std=minimal_measurement_error[test],
                     cutoff=test_cutoffs[test],
                     label=f'{test_names[test]}: division by CRP in first 24 hours')
    print("\n")
    make_fit_and_density_plot(data_to_process[data_to_process['albumin_dose_within_24h'] <= alb_treshold],
                   df2=data_to_process[data_to_process['albumin_dose_within_24h'] > alb_treshold],
                  l1=f"Albumin given =<{alb_treshold}g", l2=f'Albumin given >{alb_treshold}g',
                  title='Division by Alb in first 24 hours',
                     measurement_std=minimal_measurement_error[test],
                     cutoff=test_cutoffs[test],
                     label=f'{test_names[test]}: division by Albumin given in first 24 hours')
    print("\n")
    make_fit_and_density_plot(data_to_process[data_to_process['blood_cells_volume'] <= alb_treshold],
                   df2=data_to_process[data_to_process['blood_cells_volume'] > alb_treshold],
                  l1=f"Blood given =<{alb_treshold}ml", l2=f'Blood given >{alb_treshold}ml',
                  title='Division by blood given in first 24 hours',
                     measurement_std=minimal_measurement_error[test],
                     cutoff=test_cutoffs[test],
                     label=f'{test_names[test]}: division by blood given in first 24 hours')
#print_time("End")
#pdf.close()


                   
            






