```mermaid
graph LR;
    download_data --> |"data/raw/train.txt"| split_train_data
    download_data --> |"data/raw/test.txt"| split_test_data
    download_data --> |"data/raw/val.txt"| split_val_data
    
    split_train_data --> |"data/interim/train_features.txt"| tokenize_features
    split_train_data --> |"data/interim/train_labels.txt"| encode_labels
    
    split_test_data --> |"data/interim/test_features.txt"| tokenize_features
    split_test_data --> |"data/interim/test_labels.txt"| encode_labels
    
    split_val_data --> |"data/interim/val_features.txt"| tokenize_features
    split_val_data --> |"data/interim/val_labels.txt"| encode_labels
    
    tokenize_features --> |"data/interim/tokenized_train.txt"| train_model
    tokenize_features --> |"data/interim/tokenized_val.txt"| train_model
    tokenize_features --> |"data/interim/tokenized_test.txt"| test_model
    
    encode_labels --> |"data/interim/encoded_train_labels.txt"| train_model
    encode_labels --> |"data/interim/encoded_val_labels.txt"| train_model
    encode_labels --> |"data/interim/encoded_test_labels.txt"| test_model
    
    train_model --> |"models/model.h5"| test_model
    train_model --> |"reports/metrics.json"| test_model
    train_model --> |"reports/loss_plot.svg"| test_model
```

---

There are a total of 8 stages.

You can all run stages sequentially by running the following command:
```bash
dvc repro
```

You can also run a specific stage by running the following command:
```bash
dvc repro <stage_name>
```