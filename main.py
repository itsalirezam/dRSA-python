'''
run: python main.py {config_path} {subject}
dask run: python main.py config.yaml 3 --dask --scheduler tcp://10.40.40.17:8786

'''


import matplotlib.pyplot as plt
import numpy as np
import json, os, sys, time, re
import yaml
from pathlib import Path
import argparse, importlib

from core_functions import *
from makeplot import make_plots
from core_functions import CUSTOM_MODULE_DIR


print(f'\n\nStarted on {time.ctime()} Local Time')

# ===============

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('subject', type=int)
parser.add_argument('--dask', action='store_true', help='Run on dask cluster')
parser.add_argument('--scheduler', default=None, help='Dask scheduler address, e.g. tcp://10.40.40.17:8786 (required with --dask)')
args = parser.parse_args()

# ===============

subject = args.subject

config_path = args.config_path
with open(config_path) as f:
    config = yaml.safe_load(f)


rsa_computation_method = config['rsa_computation_method']
custom_rsa_computation_module = config.get('custom_rsa_computation_module', None)

# ===== dask cluster only =====

dask_client = None

if args.dask:
    if not args.scheduler:
        parser.error("--scheduler is required when --dask is set")
    if not re.match(r'^tcp://[\w.\-]+:\d+$', args.scheduler):
        parser.error(f"--scheduler must look like tcp://<host>:<port>, got: {args.scheduler!r}")

    from dask.distributed import Client
    try:
        dask_client = Client(args.scheduler, timeout='10s')
    except OSError as e:
        sys.exit(f"Could not connect to dask scheduler at {args.scheduler}: {e}")
    print(dask_client)

    dask_client.upload_file('core_functions.py')
    if rsa_computation_method != 'correlation' and custom_rsa_computation_module:
        mod_path = CUSTOM_MODULE_DIR / f"{custom_rsa_computation_module}.py"
        dask_client.upload_file(str(mod_path))

elif args.scheduler:
    parser.error("--scheduler was given but --dask was not set — did you forget --dask?")

# ===== configuration =====    

## load models

selected_models = []
selected_models_labels = []
model_rdm_metrics = []

for model_config in config['models']:

    model = np.load(model_config['path'])

    selected_models.append(model)
    selected_models_labels.append(model_config['name'])
    model_rdm_metrics.append(model_config['rdm_metric'])

## other

frequency_band = config['frequency_band']
region = config['region']
neural_data_folder_path = config['neural_data_folder_path']
simulation = config['simulation'] # boolean

if simulation:
    selected_neural_data = np.load(config['simulation_neural_data_path'])
else:
    selected_neural_data = np.load(f'{neural_data_folder_path}{region}-{frequency_band}/sub{subject}-{frequency_band}-{region}.npy')

neural_rdm_metric = config['neural_rdm_metric']
n_subsamples = config['n_subsamples']
subsampling_iterations = config['subsampling_iterations']
subsample_duration_sec = config['subsample_duration_sec']
averaging_diagonal_time_window_sec = config['averaging_diagonal_time_window_sec']
resolution = config['resolution']
time_lock = config['time_lock']
double_precision = config['double_precision']
save_rsa_matrices = config['save_rsa_matrices']
save_plots = config['save_plots']


## derived pameters and variables

tps = selected_neural_data.shape[1]
adtw_in_tps = int(averaging_diagonal_time_window_sec * resolution)
subsample_tps = int(subsample_duration_sec * resolution)

## mask

if config['mask']:
    mask = np.squeeze(np.load(config['mask_path']))
else:
    mask = None

## time lock

if time_lock:
    candidate_path = config['candidate_path']
    manual_subsample_start_candidates = (np.squeeze(np.load(candidate_path)) - subsample_tps // 2)
else:
    manual_subsample_start_candidates = None


## check assumptions - do lengths match

### check model counts
if not (len(selected_models) == len(selected_models_labels) == len(model_rdm_metrics)):
    raise ValueError(
        f"Lengths do not match: "
        f"models={len(selected_models)}, "
        f"labels={len(selected_models_labels)}, "
        f"metrics={len(model_rdm_metrics)}"
    )

### check model time points
mismatches = []
for model, label in zip(selected_models, selected_models_labels):
    if model.shape[1] != tps:
        mismatches.append(f"{label}: {model.shape}")

if mismatches:
    raise ValueError(
        f"Model and neural data shapes don't match:\n" +
        "\n".join(mismatches)
    )

if mask is not None and mask.shape[0] != tps:
    raise ValueError('mask time points do not match neural data')

# ===== run dRSA =====

rsa_matrices = run_drsa_pipeline(
    neural_data=selected_neural_data,
    models=selected_models,
    model_labels=selected_models_labels,
    model_rdm_metrics=model_rdm_metrics,
    neural_rdm_metric=neural_rdm_metric,
    rsa_computation_method=rsa_computation_method,
    n_subsamples=n_subsamples,
    subsampling_iterations=subsampling_iterations,
    subsample_tps=subsample_tps,
    adtw_in_tps=adtw_in_tps,
    time_lock=time_lock,
    manual_subsample_start_candidates=manual_subsample_start_candidates,
    mask=mask,
    double_precision=double_precision,
    custom_rsa_computation_module=custom_rsa_computation_module,
    dask_client=dask_client,
)


file_name = f'sub{subject}-{frequency_band}_dRSA_matrices-{rsa_computation_method}-TL{time_lock}'

## save dRSA matrices
if save_rsa_matrices:
    os.makedirs('_results', exist_ok=True)
    output_path = Path('_results')
    np.save(output_path / file_name, rsa_matrices)

    # save labels for plotting later
    with open("_results/selected_model_labels.json", "w") as f:
        json.dump(selected_models_labels, f)

## save plot
if save_plots:
    make_plots(
        rsa_data=rsa_matrices,
        model_labels=selected_models_labels,
        resolution=resolution,
        averaging_diagonal_time_window_sec=averaging_diagonal_time_window_sec,
        output_dir=Path('_results'),
        plotting_config=config,
        custom_name=file_name
    )