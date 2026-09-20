'''
Ridge Regression instead of PCA

'''


import numpy as np
from core_functions_v7 import compute_rsa_matrix_corr, compute_lag_correlation, zscore
from sklearn.decomposition import PCA

from itertools import repeat
from dask.distributed import Future, progress

from threading import Lock
log_lock = Lock()

def log_message(message, log_path="progress_log.txt"):
    with log_lock:
        with open(log_path, "a") as f:
            f.write(message + "\n")

def regression_border(model_rdm_series_list, averaging_diagonal_time_tps, correlation_threshold=0.1):
    '''
    this is to decide what timepoints to exclude of the same model, from regressing out.

    correlation_threshold: the threshold for the lag correlation to be considered significant
    '''

    exclusion_window_for_models = []
    for model in model_rdm_series_list:
        drsa_corr = compute_rsa_matrix_corr(model, model)
        lag, lag_corr = compute_lag_correlation(drsa_corr, averaging_diagonal_time_tps)

        # max of the lag where the lag_corr is above correlation_threshold

        if np.any(lag_corr > correlation_threshold):
            max_lag = int(np.max(lag[lag_corr > correlation_threshold]))
        else:
            max_lag = 0

        max_lag = min(max_lag, averaging_diagonal_time_tps) # ensure it does not exceed the averaging diagonal time window
        exclusion_window_for_models.append(max_lag)

    return exclusion_window_for_models


def compute_single_t_neural_task(i, t_neural,
                                 neural_rdm_series,
                                 model_rdm_series_list,
                                 exclusion_windows,
                                 averaging_diagonal_time_tps,
                                 variance_threshold):
    iterations, subsample_tps, rdm_len = neural_rdm_series.shape
    n_models = len(model_rdm_series_list)
    num_other_models = n_models - 1
    max_predictors = 1 + n_models * (2 * averaging_diagonal_time_tps + 1)

    # allocate once per task
    X = np.empty((rdm_len, max_predictors), dtype=neural_rdm_series.dtype)
    y = neural_rdm_series[i, t_neural]
    y = zscore(y)
    result = np.zeros((n_models, subsample_tps), dtype=neural_rdm_series.dtype)
    

    for t_model in range(subsample_tps):
        start = max(0, t_model - averaging_diagonal_time_tps)
        end = min(subsample_tps, t_model + averaging_diagonal_time_tps + 1)
        window_len = end - start

        for m_interest in range(n_models):
            main = model_rdm_series_list[m_interest][i, t_model]  # (rdm_len,)
            # build mask
            mask = np.zeros(subsample_tps, dtype=bool)
            mask[start:end] = True
            exc = exclusion_windows[m_interest]
            mask[max(0, t_model - exc):min(subsample_tps, t_model + exc + 1)] = False

            X[:, 0] = main

            # other models block
            col = 1
            for m in range(n_models):
                if m == m_interest: 
                    continue
                block = model_rdm_series_list[m][i, start:end]   # (window_len, rdm_len)
                X[:, col:col+window_len] = block.T
                col += window_len

            # same-model block
            sm = model_rdm_series_list[m_interest][i, mask]      # (k, rdm_len)
            k = sm.shape[0]
            X[:, col:col+k] = sm.T
            n_predictors = 1 + (num_other_models * window_len) + k

            # zscore scale
            Z = zscore(X[:, :n_predictors])

            # loadings = pca.components_  # (n_components, n_predictors)

            alpha = 10.0  # regularization strength
            XtX = Z.T @ Z
            ridge_betas = np.linalg.solve(XtX + alpha * np.eye(Z.shape[1]),
                                        Z.T @ y)
            betas = ridge_betas

            result[m_interest, t_model] = betas[0]

    return (i, t_neural, result)


def compute_rsa_matrix(neural_rdm_series, model_rdm_series_list,
                           averaging_diagonal_time_tps=300,
                           variance_threshold=0.99, client=None):

    if client is None:
        raise ValueError("A Dask client must be provided for parallel computation.")

    iterations, subsample_tps, _ = neural_rdm_series.shape
    n_models = len(model_rdm_series_list)

    # compute once on client
    exclusion_windows = regression_border(model_rdm_series_list, averaging_diagonal_time_tps)
    print("\u2713 compute regression borders", flush=True)

    # scatter the data for dask
    neural_f = client.scatter(neural_rdm_series)
    models_f = client.scatter(model_rdm_series_list)
    print("\u2713 scatter arrays on workers", flush=True)

    # task args
    i_list = []
    t_list = []
    for i in range(iterations):
        for t in range(subsample_tps):
            i_list.append(i)
            t_list.append(t)

    # map the top-level task, passing Futures
    results = client.map(
        compute_single_t_neural_task,
        i_list,
        t_list,
        [neural_f] * len(i_list),
        [models_f] * len(i_list),
        [exclusion_windows] * len(i_list),
        [averaging_diagonal_time_tps] * len(i_list),
        [variance_threshold] * len(i_list),
    )
    print("\u2713 submit tasks to workers", flush=True)
    progress(results, interval=5*60) # Show progress every 5 minutes
    results = client.gather(results)

    # assemble outputs
    rsa_holder = np.zeros((n_models, iterations, subsample_tps, subsample_tps), dtype=neural_rdm_series.dtype)
    for i, t_neural, out in results:
        rsa_holder[:, i, t_neural, :] = out
    rsa_matrices = [rsa_holder[m].mean(axis=0) for m in range(n_models)]
    return rsa_matrices