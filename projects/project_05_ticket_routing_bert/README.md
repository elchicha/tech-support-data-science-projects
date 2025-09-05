---

## Running the Code

### 1. From the Command Line

Activate the virtual environment (if not already active):

```sh
source env/bin/activate
```

Run the training script as a module (recommended for correct imports):

```sh
python -m src.train --data_path data/raw/report1757107912186.csv
```

Replace the `--data_path` argument with your CSV file as needed.

### 2. Debugging in VS Code

1. Open the project folder in VS Code.
2. Go to the Run & Debug panel (or press `F5`).
3. Select the configuration named **Debug src.train as module**.
4. Start debugging. You will be prompted for the data CSV path (e.g., `data/raw/report1757107912186.csv`).

This configuration ensures that relative imports work correctly by running `src.train` as a module and setting the `PYTHONPATH`.

---

# Ticket Routing with BERT

**Project goal**  
Automatically assign incoming support tickets to the correct internal queue (e.g., _Login Issues_, _Billing_, _Technical Crash_, _Feature Request_) using a fine‑tuned BERT‑style transformer.  
The model reduces manual triage effort, speeds up response times, and provides consistent routing decisions across languages supported by the underlying multilingual model.

**Why BERT?**

- Context‑aware embeddings capture the nuance of natural‑language tickets better than classic bag‑of‑words or TF‑IDF approaches.
- Pre‑trained models already know general English (or multilingual) semantics, so only a modest amount of labeled ticket data is needed to reach high accuracy.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup & Installation](#setup--installation)
3. [Data Format](#data-format)
4. [Training & Evaluation](#training--evaluation)
5. [Inference Demo](#inference-demo)
6. [Saving & Loading the Model](#saving--loading-the-model)
7. [Future Improvements](#future-improvements)
8. [License](#license)

---

## Repository Structure
