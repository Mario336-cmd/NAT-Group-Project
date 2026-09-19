# TrueCheck

**An Explainable Misinformation & Claim-Verification Assistant**

*NAT820S · Natural Language Processing and Applications*  
*Group Project · Semester 2, 2026*

## Table of Contents

- [Project Overview](#project-overview)
- [Project Requirements](#project-requirements)
- [Original FEVER Dataset](#original-fever-dataset)
- [Dataset Preparation and Exploration](#dataset-preparation-and-exploration)
- [Classical Model Training](#classical-model-training)
- [Transformer Model Training](#transformer-model-training)
- [Model Evaluation and Comparison](#model-evaluation-and-comparison)
- [Evidence Retrieval](#evidence-retrieval)

## Project Overview

False or misleading claims spread quickly on social media and messaging apps, and readers often have no easy way to check whether a claim is well supported.

This project builds a prototype that takes a short claim or headline and returns a verdict (e.g., likely true/likely false/unverified) with plain-language reasoning and links to supporting or contradicting evidence, rather than a bare label.

## Project Requirements

1. **Claim classification**  
   Compare and implement at least one classical ML approach and one deep-learning/transformer approach to classify a claim or article by truthfulness label, using the chosen dataset.

2. **Text representation**  
   Justify the chosen embeddings/features for this task.

3. **Evidence retrieval (RAG-style)**  
   For a new claim, retrieve the most relevant supporting or contradicting passages from a small evidence corpus using a similarity index.

4. **Explainable verdict generation**  
   Use an LLM, prompted to reason only over the retrieved evidence, to produce a short plain-language explanation for the verdict, explicitly citing which retrieved passage supports it, and clearly flagging when there is not enough evidence to decide.

5. **Bias & error analysis**  
   Test the classifier across at least two claim categories or topics and report where accuracy or false-positive/negative rates differ meaningfully.

6. **Multilingual consideration**  
   Discuss, or where feasible, test how the pipeline would need to change for a claim written in a language other than English.

7. **Guardrails**  
   The system must never present an unverified LLM guess as a “fact-checked” verdict; if retrieval finds no relevant evidence, it must say so rather than fabricating a verdict.

8. **Evaluation**  
   Report standard classification metrics (accuracy, F1, precision/recall per class) and, for the RAG explanation component, spot-check whether generated explanations are actually grounded in the retrieved evidence.

## Original FEVER Dataset

The project will use the FEVER fact-verification dataset, which contains claims paired with Wikipedia evidence. The original dataset files belong in `Dataset/Original/`.

| Resource | File | Download |
| --- | --- | --- |
| Training dataset | `train.jsonl` | [Download](https://fever.ai/download/fever/train.jsonl) |
| Labelled development dataset | `shared_task_dev.jsonl` | [Download](https://fever.ai/download/fever/shared_task_dev.jsonl) |
| Wikipedia evidence corpus | `wiki-pages.zip` | [Download](https://fever.ai/download/fever/wiki-pages.zip) |

### Required Starting Structure

Before working on the Python scripts, the following structure and dataset files must exist.

Download the files from the [Original FEVER Dataset](#original-fever-dataset) links above and place them in `Dataset/Original/`.

```text
NAT-Group-Project/
├── README.md
├── Dataset/
│   └── Original/
│       ├── train.jsonl
│       ├── shared_task_dev.jsonl
│       └── wiki-pages.zip
└── Python Scripts/
    ├── prepare_dataset.py
    ├── train_classical.py
    ├── train_transformer.py
    ├── evaluate_models.py
    └── evidence_retrieval.py
```

## Dataset Preparation and Exploration

**Python file:** [prepare_dataset.py](Python%20Scripts/prepare_dataset.py)

**Required libraries:** `pandas` and `scikit-learn`. The script also uses Python's built-in `pathlib`, `itertools`, `json`, and `math` modules, which require no separate installation.

The script prepares `Dataset/Original/train.jsonl` for the initial claim-only classification baseline. It inspects the data, applies light cleaning, handles conflicting labels and duplicates, and creates shared train/validation/test splits for both models. These are custom project splits. `shared_task_dev.jsonl` and `wiki-pages.zip` are not used at this stage.

### Inspection and Class Distribution

The original training file contains 145,449 records with the fields `id`, `claim`, `label`, `verifiable`, and `evidence`. The script prints class counts and percentages, data-quality checks, and three sample claims per label.

Inspection found no missing claims, missing labels, empty claims, or unexpected labels. It identified 9,904 repeated claim occurrences beyond the first copy, comparing claim text after trimming surrounding whitespace.

| Label | Original records | Percentage |
| --- | ---: | ---: |
| `SUPPORTS` | 80,035 | 55.03% |
| `NOT ENOUGH INFO` | 35,639 | 24.50% |
| `REFUTES` | 29,775 | 20.47% |
| **Total** | **145,449** | **100.00%** |

### Cleaning and Duplicate Handling

Surrounding whitespace is trimmed from claims and labels. Claim wording, capitalisation, punctuation, numbers, and negation are preserved. Stemming, lemmatisation, and stop-word removal are not applied in this shared preparation step.

Records with missing, blank, or non-text claims/labels are excluded because they do not provide a usable text input and target label. Missing or duplicate IDs and unexpected non-empty text labels stop processing for investigation.

When identical claim text appears with different labels, all records containing that claim are removed because they provide conflicting answers for the same statement. When identical claim text appears multiple times with the same label, only the first record in the original file is kept. The remaining copies are removed before creating the training, validation, and test sets. Claims are matched exactly after trimming leading and trailing whitespace.

| Cleaning result | Records |
| --- | ---: |
| Original records | 145,449 |
| Records with missing or invalid claims/labels removed | 0 |
| Records with conflicting labels removed | 2,557 |
| Duplicate records removed after excluding conflicts | 8,100 |
| **Claims remaining** | **134,792** |

### Final Training Validation and Test Splits

The prepared claims are split using stratified sampling with `random_state=42`, preserving approximately the same label proportions in each split. Both models will use these same saved splits.

| Split | Records | Share | SUPPORTS | NOT ENOUGH INFO | REFUTES |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 94,354 | 70.00% | 51,523 | 23,277 | 19,554 |
| Validation | 20,219 | 15.00% | 11,041 | 4,988 | 4,190 |
| Test | 20,219 | 15.00% | 11,041 | 4,988 | 4,190 |
| **Total** | **134,792** | **100.00%** | **73,605** | **33,253** | **27,934** |

### Columns Retained and Excluded

The processed CSVs retain three columns for the initial claim-only classifier:

| Column | Purpose |
| --- | --- |
| `id` | Links the retained record to the original dataset. It is not a model input. |
| `claim` | The text used as input to the classifier. |
| `label` | The expected classification: `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`. |

Two original fields are excluded from the processed CSVs:

- **`verifiable`:** Describes the annotation's verifiability and overlaps with the target label. It is not needed as an input to the claim-only classifier.
- **`evidence`:** Contains Wikipedia page and sentence references for evidence-based work. These references are not used as inputs at this stage.

### Running the Script and Outputs

With the required libraries installed, run this command from the project root:

```powershell
py "Python Scripts/prepare_dataset.py"
```

From inside `Python Scripts/`, run `py prepare_dataset.py` instead.

The script creates or overwrites these three files:

```text
Dataset/
└── Processed/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

Rerunning preparation overwrites only these three CSVs. It does not clear the `Processed/` directory or modify the original dataset files.

## Classical Model Training

**Planned Python file:** `Python Scripts/train_classical.py`

Train the classical NLP model. One possible approach is TF-IDF with Logistic Regression, the final model choice is still to be decided.

## Transformer Model Training

**Planned Python file:** `Python Scripts/train_transformer.py`

Fine-tune the transformer model.

## Model Evaluation and Comparison

**Planned Python file:** `Python Scripts/evaluate_models.py`

Calculate accuracy, precision, recall, F1, and confusion matrices, and compare both models accross validation and test set.

## Evidence Retrieval

**Planned Python file:** `Python Scripts/evidence_retrieval.py`

Search the FEVER/Wikipedia evidence corpus for passages relevant to a claim.
