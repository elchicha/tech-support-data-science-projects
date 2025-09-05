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
