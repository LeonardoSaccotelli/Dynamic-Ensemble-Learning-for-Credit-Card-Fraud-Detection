# Dynamic-Ensemble-Learning-for-Credit-Card-Fraud-Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 📖 Project Description
This project addresses the critical challenge of identifying fraudulent transactions within highly **unbalanced datasets**. In the context of credit card fraud, legitimate transactions vastly outnumber fraudulent ones, making standard machine learning approaches biased toward the majority class.

The core of this research is a comparative analysis between **Static Ensemble Learning** and **Dynamic Ensemble Selection (DES)**. By evaluating how these models behave when the "cost" of misclassification is high, this project aims to identify the most robust architecture for financial security.

### 🧪 Handling Class Imbalance
The framework implements a multi-tiered strategy to handle the class distribution gap:

* **Cost-Sensitive Learning:** Implementation of the `class_weight` parameter across models to assign a higher penalty to fraudulent misclassifications, forcing the algorithms to prioritize the minority class.
* **Hybrid Resampling Pipeline:** While the codebase is built to be modular, the primary focus is on the combination of:
    * **Random Undersampling:** To reduce majority class noise.
    * **Cluster Centroids:** A sophisticated undersampling technique that replaces clusters of majority samples with their centroids to preserve structural information.
* **Full `imbalanced-learn` Support:** The architecture is designed for flexibility, natively supporting any oversampling or undersampling method provided by the `imblearn` library.

### 📉 Big Data & Subsampling
Recognizing the computational strain of modern financial datasets, the project includes an **Initial Subsampling** layer. This allows for the efficient processing of large-scale data by reducing the initial volume while maintaining the statistical properties required for effective fraud detection.

---

## 🔬 Core Objectives
1.  **Benchmarking:** Comparing the predictive power of static ensembles (fixed at training) against dynamic ensembles (which adaptively select the best model for each specific transaction).
2.  **Strategy Comparison:** Evaluating the trade-offs between "Cost-Sensitive only" approaches vs. "Cost-Sensitive + Resampling" (Undersampling/Cluster Centroids).
3.  **Modular Scalability:** Providing a codebase that can easily swap different balancing techniques to find the optimal configuration for any skewed dataset.

---

## 🛠 Installation & Environment Setup

This project uses a `Makefile` to automate the setup process. Ensure you have **Python 3.10** installed on your system before proceeding.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Dynamic-Ensemble-Learning-for-Credit-Card-Fraud-Detection.git](https://github.com/YourUsername/Dynamic-Ensemble-Learning-for-Credit-Card-Fraud-Detection.git)
cd Dynamic-Ensemble-Learning-for-Credit-Card-Fraud-Detection
```

### 2. Create the Virtual Environment
```bash
make create_environment
```

### 3. Activate the Environment
Based on your operating system, run the activation command:
- Linux/macOS:
```bash
source venv/bin/activate
```

- Windows (Command Prompt):
```bash
.\venv\Scripts\activate.bat
```

- Windows (PowerShell):
```bash
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Once the environment is activated, install the required libraries (including the dynamic ensemble dependencies):
```bash
make requirements
```

---


## 🧹 Maintenance Commands

The Makefile also includes utility commands for project maintenance:

make clean,Remove __pycache__ and compiled Python files.
make clean_environment,Completely remove the venv directory.
make freeze,Update the requirements.txt file with current environment state.


| Command           | Description |
|-------------------| ------------- |
| ```make lint ```  | Check code quality and formatting using Ruff. |
| ```make format ``` | Automatically fix linting issues and format code. |
| ```make clean ``` | Remove __pycache__ and compiled Python files  |
| ```make clean_environment ``` | Completely remove the venv directory. |
| ```make freeze ``` | Update the requirements.txt file with current environment state.|

---

## 📊 Data Workflow

The project manages data through a structured pipeline. Before running any analysis, you must retrieve the raw dataset.

### 1. External Dataset Acquisition
The first step is to download the "Credit Card Fraud Detection" dataset from Kaggle. This is handled automatically by the ```dataset.py``` script.

**Execution:**

```bash
make dataset
```

**What this command does:**
- Checks if the dataset already exists in ```data/external/```. 
- If missing, it uses ```kagglehub``` to download the ```mlg-ulb/creditcardfraud``` dataset. 
- Moves and renames the file to match the project's internal configuration.
 
**Related Configuration** (```fraud_dynamic_ensemble/config.py```): The script relies on these path definitions. If you wish to change where data is stored, modify these variables:

| Variable                      | Default Value                         | Description  |
|-------------------------------|---------------------------------------|--------------|
| ```EXTERNAL_DATA_DIR ```      | ```PROJ_ROOT / "data" / "external"``` |     The directory where external raw data is stored.         |
| ```EXTERNAL_FILENAME ```      | ```creditcardfraud.csv```             |        The final filename used by the project scripts.      |
| ```RANDOM_STATE ```           | ```42```                                 |    Ensures reproducibility across data splits and sampling.          |

--- 
## 📉 Data Sampling & Reduction

The second step in the pipeline is to transition from the **External** (full) data to a **Raw** (subsampled) dataset. This allows for faster experimentation without losing the rare fraudulent signals.

1. **Subsampling Execution**


Use the following command to generate your subsampled dataset based on the current configuration:
```bash
make dataset_sampling
````
**What this command does:**

- Loads the full dataset from `data/external/`.
- Logs the initial class distribution (e.g., showing that Fraud is roughly **0.17%** of the data).
- Applies one of several sampling **policies** (defined in your config).
- Saves the resulting subset to `data/raw/credit_card_fraud_sampling.csv`.

2. **Sampling Configuration**

You can control how the data is reduced by modifying these variables in `fraud_dynamic_ensemble/config.py`.

The script is highly flexible, specifically supporting a "keep all minority" approach which is standard in fraud detection research to ensure no rare fraud cases are lost during data reduction.

| Variable      | Default Value           | Description  |
|---------------|-------------------------|--------------|
| ```POLICY ``` | ```keep_all_minority``` |     Options: `random`, `stratified`, or `keep_all_minority`.         |
| ```RATIO ```  | ```50```                |       Used with `keep_all_minority`. A ratio of `50` means the script will keep 100% of fraud cases and sample enough legitimate cases to reach a 1:50 ratio.     |
| ```N_ROWS ``` | ```None```              |    The absolute number of rows to sample (mutually exclusive with `FRAC`).       |
| ```FRAC ```   | ```None```              |      The percentage of the dataset to keep (e.g., `0.1` for 10%).  |
| ```RANDOM_STATE ```           | ```42```                                 |    Ensures reproducibility across data splits and sampling.          |

3. **Sampling Policies Explained**

- `keep_all_minority`: Specifically designed for fraud. It retains every single fraudulent transaction and samples the majority class based on the `RATIO` you set. This is the recommended setting for this project.

- `stratified`: Reduces the dataset size while maintaining the original percentage of fraud (useful if you want to keep the "needle in the haystack" difficulty exactly the same).

- `random`: A simple random draw from the data without regard for class labels.

---