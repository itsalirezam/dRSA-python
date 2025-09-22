# overview


## how to use

first load the neural data and the models in the main script
then set the settings and right parameters for analysis

to run the analysis:

for a single subject:
```zsh
python main.py {subject-number}
```

or to for every subject:
```zsh
for i in {1..9}; do
    python main.py $i
done
```