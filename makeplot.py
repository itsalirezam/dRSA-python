# python makeplot.py makeplot-cli-config.yaml {dir or numpy path}

"""
makeplot.py — dRSA plotting module

Two ways to use it:

1. Standalone CLI (averages a folder of subjects, or plots a single subject):

     python makeplot.py <path_to_npy_or_folder> <config.yaml> \
         [--labels path/to/labels.json] [--output-dir path/to/dir]

   - path_to_npy_or_folder: a single subject's .npy with shape (n_models, tps, tps),
     or a folder of such .npy files (one per subject), which get stacked and averaged.
   - plot-config.yaml:  resolution and averaging_diagonal_time_window_sec must match with the original main.py config.yaml run
    example yaml config below

   - the --labels and --output-dir flags can be ignored, it goes to default (uses the same input folder)

2. As a module, called directly from main.py right after run_drsa_pipeline, using the
   array already in memory (no need to save/reload from disk for a single run):

     from makeplot import make_plots

     if config['save_plots']:
         make_plots(
             rsa_data=rsa_matrices_arr,                 # (n_models, tps, tps)
             model_labels=selected_models_labels,
             resolution=resolution,
             averaging_diagonal_time_window_sec=averaging_diagonal_time_window_sec,
             TL_frame=TL_frame,
             output_dir='_results',
             plotting_config=config
         )

config.yaml:
    Reuses main.py's config.yaml rather than a separate one, since resolution /
    averaging_diagonal_time_window_sec must stay identical between the
    run that produced the data and the run that plots it.

        TL_frame: false
        tps_from_center: 15
        merge_model_plots: false
        save_subs_lags: false

plot-config.yaml:

        averaging_diagonal_time_window_sec: 1.5
        resolution: 100
        merge_model_plots: false
        statistical_testing: false
        cluster_p_value_threshold: 0.05
        TL_frame: false
        tps_from_center: 30
        save_subs_lags: false
    

    All of these have sane defaults if the section (or individual keys) is missing.
"""

from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import yaml


# ======= helpers 

def load_rsa_data(input_path: Path) -> np.ndarray:
    """
    Loads one subject's .npy or a folder of subjects' .npy files.
    Always returns a stacked array of shape (n_subs, n_models, tps, tps).
    """
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in: {input_path}")
        data = [np.load(f, allow_pickle=True) for f in files]
        file_name_stem = None
    elif input_path.is_file():
        data = [np.load(input_path, allow_pickle=True)]
        file_name_stem = input_path.stem
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    return np.stack(data), file_name_stem  # (n_subs, n_models, tps, tps)


def load_model_labels(labels_path: Path, n_models: int) -> list:
    try:
        with open(labels_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: could not read model labels from {labels_path}, "
              f"falling back to generic names.")
        return [f"Model {i + 1}" for i in range(n_models)]


def get_unique_path(directory: Path, filename: str) -> Path:
    """Return a path that won't overwrite an existing file."""
    directory = Path(directory)
    full_path = directory / filename

    base, ext = full_path.stem, full_path.suffix
    i = 1
    
    while full_path.exists():
        full_path = directory / f"{base}_{i}{ext}"
        i += 1

    return full_path


# ======= computation

def compute_lag_correlation(rsa_matrix, adtw_in_tps, tps_from_center=None):
    """Diagonal-averaged lag correlation from a single (tps, tps) RSA matrix."""
    lags = np.arange(-adtw_in_tps, adtw_in_tps + 1)
    lag_corr = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        diag_vals = np.diagonal(rsa_matrix.T, offset=lag)
        if tps_from_center is not None:
            center = len(diag_vals) // 2
            diag_vals = diag_vals[center: center + tps_from_center]
        lag_corr[i] = np.nanmean(diag_vals)

    return lags, lag_corr

def compute_all_lags(rsa_stack, adtw_in_tps, tps_from_center=None):
    """
    rsa_stack: (n_subs, n_models, tps, tps)
    Returns lags (n_lags,) and subs_lags (n_subs, n_models, n_lags)
    """
    n_subs, n_models = rsa_stack.shape[:2]
    n_lags = adtw_in_tps * 2 + 1
    subs_lags = np.empty((n_subs, n_models, n_lags))

    lags = np.arange(-adtw_in_tps, adtw_in_tps + 1)
    for sub in range(n_subs):
        for model in range(n_models):
            lags, subs_lags[sub, model] = compute_lag_correlation(
                rsa_stack[sub, model], adtw_in_tps, tps_from_center
            )
    return lags, subs_lags

def run_cluster_stats(selected_model_lags, cluster_p_value_threshold, n_permutations=1000):
    """
    selected_model_lags: (n_subs, n_lags) for one model. Requires n_subs > 1.
    Returns a list of (start_idx, end_idx) for significant clusters.
    """
    from mne.stats import permutation_cluster_1samp_test  # imported lazily: only needed here

    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        selected_model_lags,
        n_permutations=n_permutations,
        threshold=None,
        tail=1,
        n_jobs=1,
    )
    sig = np.where(cluster_p_values < cluster_p_value_threshold)[0]
    return [(clusters[i][0][0], clusters[i][0][-1]) for i in sig]

# ======= plotting functions

_HEX_COLORS = ["#19669d", "#a08b23", "#1e631e", "#831313", "#42167e", "#8d4d16",
               "#2f3c3e", "#5e3c99", "#b15928", "#3182bd", "#31a354", "#e6550d",
               "#756bb1", "#636363"]


def plot_merged(lags_in_seconds, lags_avg, sem, model_labels, output_dir):
    n_models = len(model_labels)
    colors = [_HEX_COLORS[i % len(_HEX_COLORS)] for i in range(n_models)]

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(model_labels):
        plt.plot(lags_in_seconds, lags_avg[i], label=label, color=colors[i])
        if sem is not None:
            plt.fill_between(lags_in_seconds, lags_avg[i] - sem[i], lags_avg[i] + sem[i],
                              alpha=0.3, color=colors[i])

    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Lag (neural time - model time) [s]')
    plt.ylabel('Correlation')
    plt.legend()
    plt.tight_layout()

    out_path = get_unique_path(output_dir, 'merge-into-one.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_grid(rsa_avg, lags_in_seconds, lags_avg, sem, subs_lags, model_labels,
              output_dir, TL_frame=False, tps_from_center=None,
              statistical_testing=True, cluster_p_value_threshold=0.05,
              custom_name=None, fix_heatmap_scales=None, fix_lag_scales=None, show_diagonal_reference=True):
    n_models = len(model_labels)
    fig, axs = plt.subplots(n_models, 2, figsize=(11, 2.7 * n_models))
    if n_models == 1:
        axs = axs.reshape(1, 2)

    if fix_heatmap_scales:
        vmin, vmax = fix_heatmap_scales
    else:
        vmin = vmax = None

    for i, model_name in enumerate(model_labels):
        rsa_matrix = rsa_avg[i]

        ax_matrix = axs[i, 0]
        im = ax_matrix.imshow(rsa_matrix, cmap='viridis', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        if show_diagonal_reference:
            ax_matrix.plot(
                [0, rsa_matrix.shape[1] - 1],
                [0, rsa_matrix.shape[0] - 1],
                color='#cc8d85',
                linestyle='--',
                linewidth=1
            )
        ax_matrix.set_title(f'dRSA Matrix     Model: {model_name}', fontsize=8)
        ax_matrix.set_xlabel('Model time', fontsize=7)
        ax_matrix.set_ylabel('Neural time', fontsize=7)
        fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        ax_lag = axs[i, 1]
        ax_lag.plot(lags_in_seconds, lags_avg[i])

        if fix_lag_scales:
            ax_lag.set_ylim(fix_lag_scales)

        if sem is not None:
            ax_lag.fill_between(lags_in_seconds, lags_avg[i] - sem[i], lags_avg[i] + sem[i], alpha=0.3)
        ax_lag.axvline(0, color='gray', linestyle='--')
        ax_lag.axhline(0, color='gray', linestyle='--')
        ax_lag.set_xlabel('Lag (neural time - model time) [s]', fontsize=7)
        ax_lag.set_ylabel('Correlation', fontsize=7)

        peak_idx = np.argmax(lags_avg[i])
        peak_lag, peak_val = lags_in_seconds[peak_idx], lags_avg[i][peak_idx]
        ax_lag.plot(peak_lag, peak_val, 'ro', markersize=2)
        ax_lag.annotate(f'Lag: {peak_lag:.2f}s', xy=(peak_lag, peak_val),
                         xytext=(peak_lag + 0.2, peak_val),
                         arrowprops=dict(arrowstyle='->', color='red'), fontsize=7)

        if statistical_testing:
            if subs_lags.shape[0] > 1:
                clusters = run_cluster_stats(subs_lags[:, i, :], cluster_p_value_threshold)
                for start, end in clusters:
                    ax_lag.axvspan(lags_in_seconds[start], lags_in_seconds[end], color='green', alpha=0.2)
            else:
                print(f"Skipping cluster stats for '{model_name}': need more than 1 subject.")

        if TL_frame:
            center_x = rsa_matrix.shape[1] // 2
            ax_matrix.axvline(center_x, color='white', linestyle=':', linewidth=1.5)
            ax_matrix.axvline(center_x + tps_from_center, color='white', linestyle=':', linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    if custom_name:
        out_path = get_unique_path(output_dir, custom_name)
    else:
        out_path = get_unique_path(output_dir, 'output.png')
        

    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


# ===== orchestrator

def make_plots(rsa_data, model_labels, resolution, averaging_diagonal_time_window_sec,
                output_dir='.', plotting_config=None, custom_name=None):
    """
    Core entry point — usable both from the CLI and directly from main.py.

    rsa_data: either
        - (n_models, tps, tps)            a single subject / an already-averaged array, or
        - (n_subs, n_models, tps, tps)     multiple subjects to average across
    model_labels: list[str], length n_models.
    resolution, averaging_diagonal_time_window_sec: same meaning as in
        config.yaml, and MUST match the run that produced rsa_data.
    TL_frame: mark diagonal average location within the "tps_from_center" frame
    plotting_config: dict — the section of config.yaml. Missing keys
        fall back to defaults.
    """
    plotting_config = plotting_config or {}
    statistical_testing = plotting_config.get('statistical_testing', False)
    cluster_p_value_threshold = plotting_config.get('cluster_p_value_threshold', 0.05)
    merge_model_plots = plotting_config.get('merge_model_plots', False)
    save_subs_lags = plotting_config.get('save_subs_lags', False)
    TL_frame = plotting_config.get('TL_frame', False)
    tps_from_center = plotting_config.get('tps_from_center', None) if TL_frame else None
    selected_models_idx = plotting_config.get('selected_models_idx', None)
    fix_heatmap_scales = plotting_config.get('fix_heatmap_scales', None)
    show_diagonal_reference = plotting_config.get('show_diagonal_reference', True)
    fix_lag_scales = plotting_config.get('fix_lag_scales', None)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rsa_data = np.asarray(rsa_data)
    if rsa_data.ndim == 3:  # (n_models, tps, tps) -> treat as a single "subject"
        rsa_data = rsa_data[np.newaxis, ...]
    elif rsa_data.ndim != 4:
        raise ValueError(f"Expected rsa_data with 3 or 4 dims, got shape {rsa_data.shape}")

    if selected_models_idx:
        rsa_data = rsa_data[:, selected_models_idx, ...]
        model_labels = [model_labels[model_idx] for model_idx in selected_models_idx]

    n_subs, n_models = rsa_data.shape[:2]
    if len(model_labels) != n_models:
        raise ValueError(f"Got {len(model_labels)} model labels for {n_models} models.")

    adtw_in_tps = int(averaging_diagonal_time_window_sec * resolution)

    rsa_avg = np.mean(rsa_data, axis=0)  # (n_models, tps, tps)

    lags, subs_lags = compute_all_lags(rsa_data, adtw_in_tps, tps_from_center)
    lags_avg = np.mean(subs_lags, axis=0)
    lags_in_seconds = lags / resolution

    sem = None
    if n_subs > 1:
        std = np.std(subs_lags, ddof=1, axis=0)
        sem = std / np.sqrt(n_subs)

    if save_subs_lags:
        np.save(output_dir / 'subs_lags.npy', subs_lags)

    if merge_model_plots:
        out_path = plot_merged(lags_in_seconds, lags_avg, sem, model_labels, output_dir)
    else:
        out_path = plot_grid(rsa_avg, lags_in_seconds, lags_avg, sem, subs_lags, model_labels,
                              output_dir, TL_frame=TL_frame, tps_from_center=tps_from_center,
                              statistical_testing=statistical_testing,
                              cluster_p_value_threshold=cluster_p_value_threshold,
                              custom_name=custom_name, fix_heatmap_scales=fix_heatmap_scales, fix_lag_scales= fix_lag_scales,
                              show_diagonal_reference = show_diagonal_reference)

    print(f"Saved plot to: {out_path}")
    return out_path


# ===== CLI

def _parse_args():
    parser = argparse.ArgumentParser(description="Plot dRSA lag correlations and matrices.")
    parser.add_argument("config_path", type=Path,
                             help="Path to the same config.yaml used to produce the .npy file(s).")
    parser.add_argument("input_path", type=Path,
                         help="A single subject's .npy file, or a folder of .npy files to average.")
    parser.add_argument("--labels", type=Path, default=None,
                         help="Path to a JSON list of model labels. "
                              "Defaults to '<results_dir>/selected_model_labels.json'.")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to save the plot. Defaults to the input's folder.")
    return parser.parse_args()


def main():
    args = _parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    rsa_data, file_name_stem = load_rsa_data(args.input_path)
    n_models = rsa_data.shape[1]

    results_dir = args.input_path if args.input_path.is_dir() else args.input_path.parent
    labels_path = args.labels or (results_dir / "selected_model_labels.json")
    model_labels = load_model_labels(labels_path, n_models)

    output_dir = args.output_dir or results_dir

    make_plots(
        rsa_data=rsa_data,
        model_labels=model_labels,
        resolution=config['resolution'],
        averaging_diagonal_time_window_sec=config['averaging_diagonal_time_window_sec'],
        output_dir=output_dir,
        plotting_config=config,
        custom_name=file_name_stem
    )


if __name__ == "__main__":
    main()


# looks like this: python makeplot.py {config path} {dir or npy path} --labels --output-dir