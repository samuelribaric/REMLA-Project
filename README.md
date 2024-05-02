# Pipeline

There are 4 stages, each defined in their own file in the `src` directory:
1) Load data (`data_loader.py`)
2) Tokenize data (`preprocess.py`)
3) Encode labels (`encode_labels.py`)
4) Train model (`train.py`)

These stages are defined in the `dvc.yaml` file.

---

To run all stages sequentially:

```
dvc repro
```

---

To run individual stages, just prepend the stage name:

```
dvc repro load_data
```

```
dvc repro tokenize_data
```

```
dvc repro encode_labels
```

```
dvc repro train_model
```