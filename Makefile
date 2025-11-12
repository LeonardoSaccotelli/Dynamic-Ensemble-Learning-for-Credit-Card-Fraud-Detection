#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Dynamic-Ensemble-Learning-for-Credit-Card-Fraud-Detection
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python3
VENV_DIR = venv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@echo "Creating Python virtual environment '$(VENV_DIR)'..."
	@$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "---"
	@echo "✅ Virtual environment created successfully."
	@echo "To **ACTIVATE** it, use the command appropriate for your OS:"
	@echo "🐧 Linux/macOS (Bash/Zsh):"
	@echo "    source $(VENV_DIR)/bin/activate"
	@echo "💻 Windows (Command Prompt):"
	@echo "    .\\$(VENV_DIR)\\Scripts\\activate.bat"
	@echo "💻 Windows (PowerShell):"
	@echo "    .\\$(VENV_DIR)\\Scripts\\Activate.ps1"
	@echo "---"

## Remove virtual environment
.PHONY: clean_environment
clean_environment:
	@echo "Removing virtual environment directory '$(VENV_DIR)'..."
	@rm -rf $(VENV_DIR)
	@echo "Virtual environment removed."

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	@echo "Python dependencies installed."

## Freeze Python dependencies
.PHONY: freeze
freeze:
	pip freeze > requirements.txt
	@echo ">> Wrote requirements.txt"

## Delete all compiled Python files
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

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset (use 'make dataset' to download the initial full dataset in data/external from kaggle)
.PHONY: dataset
dataset:
	$(PYTHON_INTERPRETER) fraud_dynamic_ensemble/dataset.py

## Make dataset sampling (use 'make dataset_sampling' to sample a subset of data for modeling)
.PHONY: dataset_sampling
dataset_sampling:
	$(PYTHON_INTERPRETER) fraud_dynamic_ensemble/dataset_sampling.py

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