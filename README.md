# overview

version 8

can be used locally and on the cluster
more modular
separate config.yaml file

## how to use

- local run
```bash
python main.py {config_path} {subject_path}
```

- dask run
```bash
python main.py {config_path} {subject_path} --dask --scheduler {dask scheduler tcp ip}
```

- do multiple subjects:

```bash
    for i in {1..9}; do
        python main.py {config_path} $i
    done
```

## makeplot.py CLI

Two ways to use it as a separate module:

- path to .npy file: a single subject's .npy with shape (n_models, tps, tps) plot the single subject.
- path to directory: a folder of such .npy files (one per subject), which get stacked and averaged.

plot-config.yaml:  resolution and averaging_diagonal_time_window_sec must match with the original main.py config.yaml run

```
python makeplot.py {config path} {dir or npy path} --labels --output-dir
```

- do multiple .npy files in a folder one by one
```
for file in /path/to/folder/*.npy; do
    python makeplot.py makeplot-cli-config.yaml "$file"
done
```

the two flags are optional, and will resort to default (same input folder) if not specified
