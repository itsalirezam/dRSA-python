# overview


## how to use

- load neural data and models in the main script
- set the settings and right parameters for analysis

- run:    
    for a single subject:
    ```zsh
    python streamlined.py {subject-number}
    ```
    
    or to for every subject:
    ```zsh
    for i in {1..9}; do
        python streamlined.py $i
    done
    ```
