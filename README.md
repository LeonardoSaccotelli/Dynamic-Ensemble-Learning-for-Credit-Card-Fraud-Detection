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


| Command                       | Description |
|-------------------------------| ------------- |
| ```make help ```              | Display a list of all available commands and their descriptions.|
| ```make lint ```              | Check code quality and formatting using Ruff. |
| ```make format ```            | Automatically fix linting issues and format code. |
| ```make clean ```             | Remove __pycache__ and compiled Python files  |
| ```make clean_environment ``` | Completely remove the venv directory. |
| ```make freeze ```            | Update the requirements.txt file with current environment state.|

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

**1. Subsampling Execution**


Use the following command to generate your subsampled dataset based on the current configuration:
```bash
make dataset_sampling
````
**What this command does:**

- Loads the full dataset from `data/external/`.
- Logs the initial class distribution (e.g., showing that Fraud is roughly **0.17%** of the data).
- Applies one of several sampling **policies** (defined in your config).
- Saves the resulting subset to `data/raw/credit_card_fraud_sampling.csv`.

**2. Sampling Configuration**

You can control how the data is reduced by modifying these variables in `fraud_dynamic_ensemble/config.py`.

The script is highly flexible, specifically supporting a "keep all minority" approach which is standard in fraud detection research to ensure no rare fraud cases are lost during data reduction.

| Variable      | Default Value           | Description  |
|---------------|-------------------------|--------------|
| ```POLICY ``` | ```keep_all_minority``` |     Options: `random`, `stratified`, or `keep_all_minority`.         |
| ```RATIO ```  | ```50```                |       Used with `keep_all_minority`. A ratio of `50` means the script will keep 100% of fraud cases and sample enough legitimate cases to reach a 1:50 ratio.     |
| ```N_ROWS ``` | ```None```              |    The absolute number of rows to sample (mutually exclusive with `FRAC`).       |
| ```FRAC ```   | ```None```              |      The percentage of the dataset to keep (e.g., `0.1` for 10%).  |
| ```RANDOM_STATE ```           | ```42```                                 |    Ensures reproducibility across data splits and sampling.          |

**3. Sampling Policies Explained**

- `keep_all_minority`: Specifically designed for fraud. It retains every single fraudulent transaction and samples the majority class based on the `RATIO` you set. This is the recommended setting for this project.

- `stratified`: Reduces the dataset size while maintaining the original percentage of fraud (useful if you want to keep the "needle in the haystack" difficulty exactly the same).

- `random`: A simple random draw from the data without regard for class labels.

---

## 🧹 Data Cleaning & Deduplication
Once the raw sample is created, the next phase of the pipeline is cleaning. This script focuses on removing redundancy to prepare a high-quality **Interim** dataset.

**1. Cleaning Execution**

To process your raw data into a cleaned format, run:
```bash
make cleaning
````
**What this command does:**

- Loads the subsampled data from `data/raw/`.
- **Deduplication**: Scans all columns and removes exact duplicate rows (keeping the first occurrence).
- **Audit Trail**: Logs the exact number and percentage of duplicates removed.
- **Integrity Check**: Re-calculates and logs the class distribution to ensure the fraud-to-legitimate ratio remains stable after cleaning.
- Saves the result to `data/interim/credit_card_fraud_cleaned.csv`.

**2. Cleaning Configuration**

This script relies on the pathing logic defined in `fraud_dynamic_ensemble/config.py`. While the core logic is automated, the following paths are used:

The script is highly flexible, specifically supporting a "keep all minority" approach which is standard in fraud detection research to ensure no rare fraud cases are lost during data reduction.

| Variable      | Default Value           | Description  |
|---------------|-------------------------|--------------|
| ```RAW_DATA_DIR / RAW_FILENAME``` | ```data/raw/credit_card_fraud_sampling.csv``` |    The input source (generated in the previous step).         |
| ```INTERIM_DATA_DIR / INTERIM_FILENAME ```  | ```data/interim/credit_card_fraud_cleaned.csv```                |       The output destination for cleaned data.  |

---

## 🛠 Feature Engineering & Transformation
The final stage of the data pipeline converts cleaned data into a **Processed** dataset. This script applies mathematical transformations to handle the high variance of transaction amounts and the periodic nature of time.

**1. Feature Engineering Execution**

To generate the final features for your models, run:
```bash
make features
````

**What this command does:**

- **Log Transformation**: Applies `log1p` to the `Amount` column. This compresses the range of transaction values, helping models handle extreme outliers common in financial data.
- **Cyclical Time Encoding**: Converts the linear `Time` column (seconds from the first transaction) into `Time_sin` and `Time_cos` features. This allows the model to understand that the end of one day is temporally close to the start of the next.
- **Column Reordering**: Moves the `Class` target to the final position for standard pipeline compatibility.
- Saves the result to `data/processed/credit_card_fraud_features.csv`.

**2. Configuration & Parameters**

This script relies on the logic defined in `fraud_dynamic_ensemble/config.py`.

| Variable      | Default Value          | Description  |
|---------------|------------------------|--------------|
| ```DAY_SECONDS``` | ```86,400``` |   The period used for time encoding (24 hours in seconds).         |
| ```PROCESSED_FILENAME ``` | ```credit_card_fraud_features.csv```               |       The final output file used for training.  |


---

## ⚙️ ️ Model Training & Evaluation
The training phase evaluates three categories of models: **Static ML models** (e.g., Random Forest, XGBoost), **Static Ensembles** (Voting/Stacking), and **Dynamic Ensemble Selection (DES)** (using `DESlib`).

**1. Training Execution**

To trigger the full training pipeline, use:

```bash 
make train
```

**What this command does:**

- **Data Preparation**: Loads the processed features, shuffles the data, and converts it to NumPy arrays for high-performance computation and `DESlib` compatibility.
- **Outer Evaluation (10x10 CV)**: Executes a **Repeated Stratified K-Fold** (10 splits, 10 repeats) to ensure that performance metrics are statistically robust and not a result of a "lucky" data split.
- **Hyperparameter Tuning**: For every fold, it performs an inner **RandomizedSearchCV** to optimize feature selection (`SelectKBest`) and model parameters.
- **Imbalance Handling**: Simultaneously applies **Cost-Sensitive Learning** (via `class_weight`) and **Resampling** (e.g., Cluster Centroids) as defined in your configuration.
- **Persistence**: Saves detailed CSVs of **Generalization** (test) and **Resubstitution** (train) metrics for every single model in a dedicated experiment folder.

**2. Configuration & Parameters**

This script relies on the logic defined in `fraud_dynamic_ensemble/config.py`. You can modify the nature of the experiment by toggling these specific settings:

| Category        | Parameter         | Description  |
|-----------------|-------------------------|--------------|
| **Imbalance**   | ```USE_COST_SENSITIVE_LEARNING``` |     If `True`, models will use `class_weight` to penalize fraud errors more heavily.        |
| **Resampling**  | ```RESAMPLING_METHOD```                |      Options include `ClusterCentroids`, `RandomUnderSampler`, or `None`.   |
| **Selection**   | ```FS_K_BEST_CANDIDATES```              |   A list of values (e.g., `[20, 24, "all"]`) that the tuner will test to find the best number of features.       |
| **Parallelism** | ```CV_OUTER_PARALLEL_N_JOBS```              |     Number of CPU cores to use for the outer folds. Set to `-1` for all cores.  |
| **DES**         | ```DSEL_SIZE```                |    The percentage of data (e.g., `0.20`) reserved to train the "competence" of Dynamic Ensembles.         |

**3. Understanding the Experiment Logic**

Before running, ensure you uncomment the models you want to test in the `STATIC_MODELS` and `DES_MODELS` lists within `config.py`.

- **Static Ensembles**: These are fixed after training. The script builds a "Pool" (e.g., using `RandomForest` and `XGBoost`) and combines them.
- **Dynamic Ensembles (DES)**: Unlike static models, these "decide" which expert model to trust for each specific transaction based on the local neighborhood of the data point.
- **Metric Focus**: Because accuracy is misleading in fraud, the pipeline focuses on `F1-Score` (defined by `TUNING_SCORING`) to balance catching thieves (Recall) without flagging too many innocent customers (Precision).

---
## 📓 Notebooks & In-depth Analysis
The `notebooks/` directory contains the experimental logs and visual analyses of the project. These are organized chronologically to mirror the research workflow.

**Data Understanding (DU) & Preparation (DP)**
- `1.0-data-quality-inspection`: Initial assessment of the external dataset to identify missing values, data types, and potential inconsistencies.
- `2.0-data-exploration`: Comprehensive Exploratory Data Analysis (EDA) focused on feature distributions and the extreme skewness of the target class.
- `3.0-sampling-checks`: A validation notebook to ensure the **Raw** subsampled dataset remains a statistically representative "mirror" of the original big dataset.
- `4.0-sampling-exploration`: An educational deep-dive into different under-and-over-sampling methods to visualize how they alter the decision boundary.

**Evaluation (EV) & Experimental Results**
- `5.0-feature-selection-analysis`: A post-hoc analysis of the feature selection process, identifying which variables were most frequently selected during hyperparameter tuning across different models.
- `6.0-models-evaluation`: Detailed performance breakdown for individual models within a specific experimental configuration.
- `6.1-models-evaluation-comparison`: Comparison of different models within the same experiment setting, utilizing qualitative plots, quantitative metrics, and **Resampled Corrected t-tests** for statistical significance.
- `6.2-experimental-schemas-comparison`: An "All-vs-All" comparison across different experiment settings (e.g., comparing Cost-Sensitive + Cluster Centroids vs. Pure Cost-Sensitive) to determine the optimal pipeline.

---

## License

This project is released under the [LICENSE](LICENSE). See the LICENSE file for details.