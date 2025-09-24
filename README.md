# overview


## how to use

- load neural data and models in the main script
- set the settings and right parameters for analysis

- run:    
    for a single subject:
    ```zsh
    python main.py {subject-number}
    ```
    
    for multiple subjects:
    ```zsh
    for i in {1..9}; do
        python main.py $i
    done
    ```
