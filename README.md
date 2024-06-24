# Model Training Pipeline

This repository contains the ML training pipeline for our course project. It includes all necessary steps to train a model and stores the resulting model in an accessible location with a public link for integration into `model-service`. The preprocessing has been factored into a separate library, `lib-ml`, which is managed through the package manager PyPi. Additionally, this repository includes a GitHub workflow that runs linters to ensure code quality.

## Setup

### 1. Clone the repository
```bash
git clone git@github.com:remla2024-team9/model-training.git
```

### 2. Navigate to the project directory
```bash
cd REMLA-Project
```

### 3. Install Poetry
To manage dependencies efficiently, we use Poetry. Install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation). Ensure that Poetry is not installed within the project's virtual environment but globally or per user.

### 4. Setting Up the Environment with Poetry
Once Poetry is installed, you can set up the project environment:

#### Note! Use Python 3.10, it might be best if you remove your existing virtual environment before running the commands below. Please do not use `pip install package`. Poetry does not track dependencies when `pip install` is used. Instead use `poetry add package` to add additional packages to poetry.

```bash
poetry env use python3.10
```

#### Alternative approach:
```bash
where python (for Windows) - Copy the path of Python 3.10
where python3 (for Mac)
poetry env use PYTHON_PATH - This should start installing packages already. If it doesn't, run: 'poetry install'
```

#### Install necessary dependencies
```bash
poetry install
```

##### Possible Problems During Depency Installation
If you run into issues when installing TensorFlow, check your system path length limit (on Windows). Follow this [link](https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/) to disable the path length limit.

#### Activate the virtual environment
```bash
poetry shell
```

### 5. Install Google Cloud CLI
To manage `dvc` configurations and to store the trained model we use Google Cloud Storage. To able to run the DVC pipeline, `gcloud` CLI must be installed globally. Follow this [link](https://cloud.google.com/sdk/docs/install) to install. After installation, run `gcloud auth login` and follow the first link to complete authentication.

### 6. Bind to the DVC Remote Storage
To manage and version control large data files and models, we use Data Version Control (DVC):

#### Initialize DVC (if not already done)
```bash
poetry run dvc init
```

#### Pull data from the DVC remote storage
```bash
poetry run dvc pull
```

### 7. Running the Project
To run the project and reproduce all stages defined in `dvc.yaml`:

```bash
poetry run dvc repro
```

To run a specific stage:

```bash
poetry run dvc repro <stage_name>
```

### 8. Running the Tests
To run all tests:

```bash
poetry run dvc repro test_long_urls test_memory_usage test_repeatability test_tokenization_latency test_implicit_bias test_integration test_parity
```

To run a specific test:

```bash
poetry run dvc repro <test_name>
```

### 9. Running the Linters
To run the linters, execute the following command:

```bash
poetry run pylint <directoryname_or_filename>
```

To maintain high code quality, we use two linting tools: Pylint and DSLinter. More details on design decisions can be found in the report.

### Pylint

Pylint is a well-known tool for static code analysis, helping us enforce coding standards and identify code smells. It is configured with the `.pylintrc` file found in the root folder, based on the [DSLinter configuration](https://github.com/Hynn01/dslinter/blob/main/docs/pylint-configuration-examples/pylintrc-for-ml-projects/.pylintrc).

**Usage:** To run Pylint, navigate to the project root directory and execute:

```sh
pylint --rcfile=.pylintrc .
```

### DSLinter

DSLinter extends Pylint by addressing issues specific to data science workflows, such as data handling and model validation techniques.

### Comparison of Linters

We considered other linters, but chose Pylint and DSLinter for their specific benefits:

- **Flake8**: Good alternative but lacks specialization for ML tools like TensorFlow, NumPy, and pandas.
- **Black**: An uncompromising code formatter, but less relevant and too rigid for our needs.
- **Ruff**: Extremely fast but lacks ML-specific plugins and is unnecessary for our small codebase.
