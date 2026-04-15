
import psycopg2
import pandas as pd
import os
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



class probabilitymodel:
    def __init__(self, x, y, params=None, model_type=None):
        self.x = x
        self.y = y
        self.params = params
        self.model_type  = model_type
        
    def calculate_mean_scale(self, t):
        if self.params is None or self.model_type is None:
            raise RuntimeError("Model not fitted. Call `fit()` first.")

        a = self.params[0]
        c = self.params[1]
        tau = self.params[2]
        measurement = self.params[3]
        if self.model_type == 'kinetic_jfskewt':
            d = self.params[6]
        else:
            d = 0

        mu = a * t + d
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
        a, c, tau, measurement, ta, tb, d = params
        mu_t = a * x + d
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

    def fit(self, x=None, y=None, bounds=None, measurement_std=None, method='best', init_params=None):
        if x is None: x = self.x
        if y is None: y = self.y

        # Initial guesses
        if init_params is None:
            init_gaussian = [0, 1.0, 1.0, measurement_std / 2]
            init_studentt = [0, 1.0, 1.0, measurement_std / 2, 5.0]
            init_skewnorm = [0, 1.0, 1.0, measurement_std / 2, 0.0]  # alpha init at 0
            init_jfskewt = [0, 1.0, 5.0, measurement_std / 10, 2, 2.0, 0]  # df=5.0, alpha=0.0
            init_skewt = [0, 1.0, 1.0, measurement_std / 2, 0, 3.0]  #a=1 df=5.0
        else:
            init_gaussian = init_param
            init_studentt = init_param
            init_skewnorm = init_param  # alpha init at 0
            init_jfskewt = init_param  # df=5.0, alpha=0.0
            init_skewt = init_param  #a=1 df=5.0            
        

        
        # Bounds        
        bounds_jfskewt = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 168.0),
            (measurement_std, None), (0.6, 50), (0.6, 50), (None, None) # under 0.6 (or smthg like 0.55) returns NaNs 
        ]
        bounds_skewt = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0),
            (measurement_std, None), (-50, 50), (0, 30)
        ]
        bounds_gaussian = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0), (measurement_std, None)
        ]
        
        bounds_studentt = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0), (measurement_std, None), (0, 30)
        ]
        bounds_skewnorm = bounds if bounds else [
            (None, None), (1e-3, None), (1e-2, 24.0), (measurement_std, None), (-20, 20)
        ]

        if method == 'best':
            res_gaussian = minimize(
                self.neg_log_likelihood_kinetics, init_gaussian, args=(x, y),
                bounds=bounds_gaussian, method='L-BFGS-B', options={'maxiter': 10000}
            )
            #print('gauss optimized')
            res_studentt = minimize(
                self.neg_log_likelihood_kinetics_t, init_studentt, args=(x, y),
                bounds=bounds_studentt, method='L-BFGS-B', options={'maxiter': 10000}

            )

            res_skewnorm = minimize(
                self.neg_log_likelihood_kinetics_skewnorm, init_skewnorm, args=(x, y),
                bounds=bounds_skewnorm, method='L-BFGS-B', options={'maxiter': 10000}

            )
            #print('skewnorm optimized')
            
            res_jfskewt = minimize(
                self.neg_log_likelihood_kinetics_jfskewt, init_jfskewt, args=(x, y),
                bounds=bounds_jfskewt, method='L-BFGS-B', options={'maxiter': 10000}

            )
            #print('jfskewt optimized')
            #res_skewt = minimize(
            #    self.neg_log_likelihood_kinetics_skewt, init_skewt, args=(x, y),
            #    bounds=bounds_skewt, method='L-BFGS-B', options={'maxiter': 100}

            #)
            #print('skewt optimized')

            nlls = {
            'kinetic_gaussian': res_gaussian.fun,
            'kinetic_studentt': res_studentt.fun,
            'kinetic_skewnorm': res_skewnorm.fun,
            'kinetic_jfskewt': res_jfskewt.fun,
            #'kinetic_skewt': res_skewt.fun
            }
            best_model = min(nlls, key=nlls.get) if method == 'best' else f"kinetic_{method}"
            self.model_type = best_model

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

        else:            
            if method == 'kinetic_gaussian':
                res_gaussian = minimize(
                    self.neg_log_likelihood_kinetics, init_gaussian, args=(x, y),
                    bounds=bounds_gaussian, method='L-BFGS-B', options={'maxiter': 10000}
                )
                #print('gauss optimized')

                
                self.params = res_gaussian.x
            elif method == 'kinetic_studentt':
                res_studentt = minimize(
                    self.neg_log_likelihood_kinetics_t, init_studentt, args=(x, y),
                    bounds=bounds_studentt, method='L-BFGS-B', options={'maxiter': 10000}

                )
                self.params = res_studentt.x
            elif method == 'kinetic_skewnorm':
                res_skewnorm = minimize(
                    self.neg_log_likelihood_kinetics_skewnorm, init_skewnorm, args=(x, y),
                    bounds=bounds_skewnorm, method='L-BFGS-B', options={'maxiter': 10000}

                )
                #print('skewnorm optimized')
                self.params = res_skewnorm.x
            elif method == 'kinetic_jfskewt':
                res_jfskewt = minimize(
                    self.neg_log_likelihood_kinetics_jfskewt, init_jfskewt, args=(x, y),
                    bounds=bounds_jfskewt, method='L-BFGS-B', options={'maxiter': 10000}

                )
                self.params = res_jfskewt.x
            elif method == 'kinetic_skewt':
                #print('jfskewt optimized')
                #res_skewt = minimize(
                #    self.neg_log_likelihood_kinetics_skewt, init_skewt, args=(x, y),
                #    bounds=bounds_skewt, method='L-BFGS-B', options={'maxiter': 100}

                #)
                #print('skewt optimized')
                raise Exception('Not implemented')
                self.params = res_skewt.x
            else:
                raise Exception('Unkown method/model')
            self.model_type = method



        # Choose model
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
            _, _, _, _, ta, tb, d = params
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
            _, _, _, _, ta, tb, d = self.params
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
            _, _, _, _, ta, tb, _ = self.params
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
            _, _, _, _, ta, tb, _ = self.params
            lower = jf_skew_t.ppf(lower_q, ta, tb, loc=mu, scale=sigma)
            upper = jf_skew_t.ppf(upper_q, ta, tb, loc=mu, scale=sigma)

        elif self.model_type == 'kinetic_skewt':
            _, _, _, _, a, df = self.params
            lower = skewt.ppf(lower_q, a, df, loc=mu, scale=sigma)
            upper = skewt.ppf(upper_q, a, df, loc=mu, scale=sigma)

        else:
            raise ValueError("Unknown model type.")

        return lower, upper
        
    def ci_max(self, x_eval, alpha=0.05):
    	a, b = self.ci(x_eval,alpha=alpha)
    	return max(abs(a), abs(b))

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
            _, _, _, _, ta, tb, _ = self.params
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
            a, c, tau, measurement, ta, tb, d = self.params
            return (
                f"$\mu(t) = {a:.3f} \cdot t {'' if d < 0 else '+'} {d:.3f}$\n"
                f"$\sigma(t) = \sqrt{{({c:.3f} \cdot t / ({tau:.3f} + t))^2 + {measurement:.3f}^2}}$\n"
                f"$a ={ta:.3f}$, $b={tb:.3f}$"
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

    def x_for_confidence_exceeds_abs(self, threshold, x_bounds=(0.0, 48.0), n_points=3000):
        """
        Finds the x where either the lower or upper bound of the 95% CI
        first exceeds the given threshold in absolute value.

        Instead of a numerical solver, evaluates CI on a dense grid
        and finds the first crossing using linear interpolation.

        Returns the smaller x satisfying either condition, or NaN if not found.
        """
        if self.params is None or self.model_type is None:
            raise RuntimeError("Model not fitted. Call `fit()` first.")

        if threshold is None:
            return np.nan

        # Grid of x values
        xs = np.linspace(x_bounds[0], x_bounds[1], n_points)

        # Compute CI for all grid points
        lowers, uppers = self.ci(xs, alpha=0.05)

        # Function to find first crossing for one branch
        def find_crossing(vals):
            diffs = np.abs(vals) - threshold
            # Find indices where it changes from negative -> positive
            mask = (diffs[:-1] < 0) & (diffs[1:] >= 0)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                if diffs[0] > 0:
                    return 0
                else:
                    return None
            i = idxs[0]
            # Linear interpolation between xs[i], xs[i+1]
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = diffs[i], diffs[i+1]
            if y1 == y0:  # flat edge case
                return x1
            frac = -y0 / (y1 - y0)
            return x0 + frac * (x1 - x0)

        x_lower = find_crossing(lowers)
        x_upper = find_crossing(uppers)

        candidates = [x for x in (x_lower, x_upper) if x is not None]
        return min(candidates) if candidates else np.nan

    @staticmethod
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

            axs[0].set_title(f"PP plot of PIT values ({model.model_type})")
            axs[0].set_xlabel('Theoretical cumulative probability (PIT value)')
            axs[0].set_ylabel('Empirical cumulative probability')
            axs[0].legend()
            # 2. Histogram of standardized residuals
            axs[1].hist(standardized_residuals, bins=80, edgecolor='k', density=True)
            axs[1].set_title('Histogram of standardized residuals')
            axs[1].set_xlabel('Standardized residual')
            axs[1].set_ylabel('Density')

            # 3. PIT histogram
            axs[2].hist(pit_values, bins=80, edgecolor='k', density=True)
            axs[2].set_title('PIT histogram (should be ~ uniform)')
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

