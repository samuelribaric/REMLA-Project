###   Ensuring Code Quality with Pylint and DSLinter

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
