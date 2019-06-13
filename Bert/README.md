## Bert
1. Train Bert
    ```
    cd Bert/pytorch-pretrained-BERT/examples
    mkdir train_output
    python3.7 run_classifier.py
    ```
2. Predict
    ```
    python3.7 predict.py --data_dir PATH_TO_TEST_CSV --output_csv_path OUTPUT_PATH
    ```
