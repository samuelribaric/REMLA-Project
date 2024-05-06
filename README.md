# REMLA-Project

Welcome to the REMLA-Project repository! This project is designed to utilize an automoated pipeline to deploy a neural network classifier that predicts phising URLs. Here, you'll find all the necessary steps to get the project up and running.

### 1. Clone the repository
```bash
git clone git@github.com:samuelribaric/REMLA-Project.git
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

#### Activate the virtual environment
```bash
poetry shell
```

### 5. Bind to the DVC Remote Storage
To manage and version control large data files and models, we use Data Version Control (DVC):

#### Initialize DVC (if not already done)
```bash
poetry run dvc init
```

#### Pull data from the DVC remote storage
```bash
poetry run dvc pull
```

### 6. Running the Project
To run the project and reproduce all stages defined in `dvc.yaml`:

```bash
poetry run dvc repro
```

You can also run a specific stage with:

```bash
poetry run dvc repro <stage_name>
```


## Project Structure

The project follows a structured pipeline for data processing and model training, as detailed in `pipeline.md` located in the docs directory.


##   Ensuring Code Quality with Pylint and DSLinter

To maintain good code quality in our project, we have incorporated two linting tools into the development workflow: Pylint and DSLinter.

#### Pylint

Pylint is a well known and respected tool for static code analysis. We use it to enforce a coding standard and to find code smells in our codebase. 

##### DSLinter

DSLinter is a special linting tool designed for data science and machine learning projects. It extends pylint by addressing issues that are specific to data science workflows, like the misuse of data handling functions and model validation techniques. 

Pylint is configured with the `.pylintrc` file found in the root folder. Our specifically is based on that recommended by the DSLinter documentation: https://github.com/Hynn01/dslinter/blob/main/docs/pylint-configuration-examples/pylintrc-for-ml-projects/.pylintrc

**Usage:** To run Pylint on the project, navigate to the project root directory and execute"

`pylint <directoryname_or_filename>` 

#### Comparison
3 other linters besides pylint/dslint were considered for the project
##### Flake8
Very likely the next best choice but the team is more experienced with configuring pylint. dslinter is also specialised for tensorflow, NumPy, pandas etc. and the team wanted to experiment with it
##### Black
Black is known as "The uncompromising Python code formatter" which allows for no configuration or customisation. Excellent for ensuring readability across project but not very relevant (and too annoying to use) for ours
##### Ruff
Ruff is known to be an extremely fast linter for python, but this is not relevant for our (small) codebase, and unlike pylint it does not offer ML-specific plugins
