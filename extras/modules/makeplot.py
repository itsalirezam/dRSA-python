'''

takes .npy for each subject with shape (n_models, subsample_tps, subsample_tps)
takes a .json for the model labels
run this script in the folder containing the .npy and .json files (or change the path)

retures plots that are averaged across subjects or plots single subjects
can be set to do stats

make sure to set resolution and adtw_in_sec in settings

---

edit: if you run this with the first argument as np file path, it will only take that file instead of all in the folder

- doesn't overwrite of an output with that name already exists


could be helpful for running makeplot.py separately each time for single npy files:
```
for file in *.npy; do
    python makeplot.py "$file"
done

```

'''

# === Settings ===

time_lock = True
tps_from_center = 30 if time_lock else None

adtw_in_sec = 1.5 # window for diagonal averaging
resolution = 100 # for plot in seconds

statistical_testing = True
if statistical_testing:
    cluster_p_value_threshold = 0.05

save_subs_lags = False # saves a 2d array shape: (n_subs, models, n_lags)

merge_model_plots = False # set to True for all models in one plot

# ========


import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

if statistical_testing:
    from mne.stats import permutation_cluster_1samp_test 
    from scipy import stats

results_path = os.getcwd()

def compute_lag_correlation(rsa_matrix, averaging_diagonal_time_window, tps_from_center=None):
    """
    Computes lag correlation (diagonal average) from RSA matrix,
    restricted to ± averaging_diagonal_time_window (in timepoints).
    """
    max_lag = averaging_diagonal_time_window

    lags = np.arange(-max_lag, max_lag + 1)
    lag_corr = np.zeros_like(lags, dtype=float)

    if tps_from_center == None:
        for i, lag in enumerate(lags):
            diag_vals = np.diagonal(rsa_matrix.T, offset=lag)
            lag_corr[i] = np.nanmean(diag_vals)
    else: # time lock
        for i, lag in enumerate(lags):
            diag_vals = np.diagonal(rsa_matrix.T, offset=lag)
            center = len(diag_vals) // 2
            window_vals = diag_vals[center : center + tps_from_center]
            lag_corr[i] = np.nanmean(window_vals)

    return lags, lag_corr


# the directory should contain, same result for different subjects

file_path = sys.argv[1] if len(sys.argv) > 1 else None
if file_path == None:
    files = sorted(f for f in os.listdir(results_path) if f.endswith(".npy"))
    loaded_data = [np.load(os.path.join(results_path, f), allow_pickle=True) for f in files]
else:
    loaded_data = [np.load(file_path, allow_pickle=True)]

# stack all subs lags (averaged diagonals)
rsa_subs_stacked = np.stack(loaded_data) # shape: (n_subjects, n_models, subsample_tps, subsample_tps)
rsa_subs_averaged = np.mean(rsa_subs_stacked, axis=0) # for the plot, shape: (n_models, subsample_tps, subsample_tps)

n_subs = rsa_subs_stacked.shape[0]
n_models = rsa_subs_stacked.shape[1]
adtw_in_tps = int(adtw_in_sec * resolution)
n_lags = adtw_in_tps * 2 + 1

# compute lags for each subject and each model
subs_lags = np.empty((n_subs, n_models, n_lags)) # shape (n_subjects, n_models, time points e.g. -300 to + 300)
for sub in range(n_subs):
    for model in range(n_models):
        lags, subs_lags[sub, model] = compute_lag_correlation(rsa_subs_stacked[sub, model], adtw_in_tps, tps_from_center=tps_from_center)

# calculate Standard Error of Mean
if len(sys.argv) == 1:
    std = np.std(subs_lags, ddof=1, axis=0) # standard deviation across subjects, for each model
    sem = std / np.sqrt(n_subs) # shape: (n_models, adtw_in_tps)

# mean lag across subjects
lags_subs_averaged = np.mean(subs_lags, axis=0)

if save_subs_lags:
    np.save('subs_lags.npy', subs_lags)


# plot
try:
    with open("selected_model_labels.json", "r") as f:
        selected_models_labels = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error type: {type(e).__name__}")
    print(f"you probably didn't provide the json for model names ...")
    selected_models_labels = [f"Model {i+1}" for i in range(n_models)]


lags_in_seconds = lags/resolution

if merge_model_plots:

    hex_colors = ["#19669d","#a08b23", "#1e631e", "#831313", "#42167e", "#8d4d16", "#2f3c3e", "#5e3c99", "#b15928", "#3182bd", "#31a354", "#e6550d", "#756bb1", "#636363"]
    colors = hex_colors[:n_models]

    plt.figure(figsize=(10, 5))
    for i in range(n_models):
        label = selected_models_labels[i]
        plt.plot(lags_in_seconds, lags_subs_averaged[i], label=label, color=colors[i])
        if len(sys.argv) == 1:
            plt.fill_between(lags_in_seconds,
                            lags_subs_averaged[i] - sem[i],
                            lags_subs_averaged[i] + sem[i],
                            alpha=0.3,
                            color=colors[i])
        
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Lag (neural time - model time) [s]')
    plt.ylabel('Correlation')
    # plt.title('')
    plt.legend()
    plt.tight_layout()

    # save without overwriting the previous output
    filename = 'merge-into-one.png'
    full_path = os.path.join(results_path, filename)
    # increment number if file exists
    i = 1
    base, ext = os.path.splitext(full_path)
    while os.path.exists(full_path):
        full_path = f"{base}_{i}{ext}"
        i += 1
    plt.savefig(full_path, dpi=300)
    # plt.show()

else:
        
    fig, axs = plt.subplots(n_models, 2, figsize=(11, 2.7 * n_models))

    for i, rsa_matrix in enumerate(rsa_subs_averaged):
        model_name = selected_models_labels[i]

        # RSA matrix plot
        ax_matrix = axs[i, 0] if n_models > 1 else axs[0]
        im = ax_matrix.imshow(rsa_matrix, cmap='viridis', aspect='auto', origin='lower')
        ax_matrix.set_title(f'dRSA Matrix     Model: {model_name}', fontsize=8)
        ax_matrix.set_xlabel('Model time', fontsize=7)
        ax_matrix.set_ylabel('Neural time', fontsize=7)
        fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        # Lag correlation plot
        ax_lag = axs[i, 1] if n_models > 1 else axs[1]
        ax_lag.plot(lags_in_seconds, lags_subs_averaged[i])
        if len(sys.argv) == 1:
            ax_lag.fill_between(lags_in_seconds,
                            lags_subs_averaged[i] - sem[i],
                            lags_subs_averaged[i] + sem[i],
                            alpha=0.3)
        ax_lag.axvline(0, color='gray', linestyle='--')
        ax_lag.axhline(0, color='gray', linestyle='--')
        # ax_lag.set_title(f'Lagged Correlation', fontsize=8)
        ax_lag.set_xlabel('Lag (neural time - model time) [s]', fontsize=7)
        ax_lag.set_ylabel('Correlation', fontsize=7)

        # Mark the peak
        peak_idx = np.argmax(lags_subs_averaged[i])            # index of maximum correlation
        peak_lag = lags_in_seconds[peak_idx]    # lag at maximum
        peak_val = lags_subs_averaged[i][peak_idx]             # maximum correlation value

        ax_lag.plot(peak_lag, peak_val, 'ro', markersize=2)    # red dot at the peak
        ax_lag.annotate(f'Lag: {peak_lag:.2f}s',
                        xy=(peak_lag, peak_val),
                        xytext=(peak_lag + 0.2, peak_val),  # offset text a bit
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=7)
        
        # statistical testing
        if statistical_testing:

            selected_model = subs_lags[:, i, :]

            # do cluster permutation
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                selected_model, 
                n_permutations=1000,
                threshold=None,
                tail=1,   
                n_jobs=1
            )

            significant_clusters = np.where(cluster_p_values < cluster_p_value_threshold)[0]

            # Highlight significant clusters
            for cl_idx in significant_clusters:
                cluster = clusters[cl_idx][0]
                start, end = cluster[0], cluster[-1]

                # turn the indices to time:
                t_start, t_end = lags_in_seconds[start], lags_in_seconds[end]

                # Shade cluster region
                ax_lag.axvspan(t_start, t_end, color='green', alpha=0.2)

        if time_lock:

            # timelock patch
            matrix_center_x, matrix_len_y = rsa_matrix.shape[1]//2,  rsa_matrix.shape[0]
            ## Define the region you want to highlight
            x_start, x_end = matrix_center_x, matrix_center_x + tps_from_center

            ax_matrix.axvline(x_start, color='white', linestyle=':', linewidth=1.5)
            ax_matrix.axvline(x_end,   color='white', linestyle=':', linewidth=1.5)

    # Clean up layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # save without overwriting the previous output
    filename = 'multiple-plots.png'
    full_path = os.path.join(results_path, filename)
    # increment number if file exists
    i = 1
    base, ext = os.path.splitext(full_path)
    while os.path.exists(full_path):
        full_path = f"{base}_{i}{ext}"
        i += 1
    plt.savefig(full_path, dpi=300)
    # plt.show()

