# Project Data Pipeline

The following Mermaid diagram illustrates the data pipeline managed by DVC in this project:

```mermaid
graph LR;
    download_data --> |"Download raw data files"| split_train_data
    download_data --> |"Download raw data files"| split_test_data
    download_data --> |"Download raw data files"| split_val_data
    
    split_train_data --> |"Split raw training data into features and labels"| tokenize_features
    split_train_data --> |"Split raw training data into features and labels"| encode_labels
    
    split_test_data --> |"Split raw test data into features and labels"| tokenize_features
    split_test_data --> |"Split raw test data into features and labels"| encode_labels
    
    split_val_data --> |"Split raw validation data into features and labels"| tokenize_features
    split_val_data --> |"Split raw validation data into features and labels"| encode_labels
    
    tokenize_features --> |"Tokenize feature text"| train_model
    tokenize_features --> |"Tokenize feature text"| test_model
    
    encode_labels --> |"Encode label data"| train_model
    encode_labels --> |"Encode label data"| test_model
    
    train_model --> |"Trained model file"| test_model
    train_model --> |"Training metrics and plots"| test_model
