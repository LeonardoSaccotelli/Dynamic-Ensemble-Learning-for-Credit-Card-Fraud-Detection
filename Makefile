#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Dynamic-Ensemble-Learning-for-Credit-Card-Fraud-Detection
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python # Which Python command to use (usually just python, but can be changed for different systems).

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies  (run 'make requirements' to install dependencies)
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files: .pyc, .pyo, __pycache__ (use 'make clean' to clean)
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment (use 'make create_environment' to create a new environments)
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset (use 'make dataset' to generate raw dataset)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py

## Clean the dataset (use 'make cleaning' to generate cleaned interim data)
.PHONY: cleaning
cleaning:
	$(PYTHON_INTERPRETER) src/cleaning.py

## Create the features to be used to model (use 'make features' to generate final features data)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) src/features.py

## Run the training pipeline (use 'make train' to train the models)
.PHONY: train
train:
	$(PYTHON_INTERPRETER) src/modeling/train.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
