# TrueCheck

- [Overview](#overview)
- [Features](#features)
- [Report, Model and Dataset Downloads](#report-model-and-dataset-downloads)
- [Project File Guide](#project-file-guide)
- [Requirements](#requirements)
- [Run the TrueCheck Web Application](#run-the-truecheck-web-application)
- [Recreate Data Preparation, Models and Evaluations](#recreate-data-preparation-models-and-evaluations)
- [Troubleshooting](#troubleshooting)

## Overview

A local web application for checking factual claims against retrieved evidence. TrueCheck combines a fine-tuned BERT classifier with evidence retrieval, cited explanations and support for English, Portuguese and Afrikaans.

Developed for NAT820S, Natural Language Processing and Applications, at the Namibia University of Science and Technology.

Go to [Run the TrueCheck Web Application](#run-the-truecheck-web-application) to use the supplied fine-tuned model. Go to [Recreate Data Preparation, Models and Evaluations](#recreate-data-preparation-models-and-evaluations) to clean the original data, train either model or repeat the available evaluations.

## Features

- **Claim verification:** returns a SUPPORTS or REFUTES prediction, model confidence and an explanation linked to the retrieved evidence.
- **Evidence retrieval:** combines BM25 keyword search and MiniLM semantic search over a local FEVER Wikipedia corpus. The number of passages can be changed from 1 to 10, with 1 selected by default.
- **Multilingual input:** accepts factual claims in English, Portuguese and Afrikaans. Portuguese and Afrikaans claims are translated into English for retrieval and classification, with explanations and evidence translations returned in the input language.
- **Input screening:** rejects greetings, creative requests and other inputs outside the supported factual-claim scope.
- **Image and PDF input:** extracts editable text from PNG, JPEG, WebP and PDF files before submission.
- **Voice input and read aloud:** transcribes recorded speech into editable text and reads responses using available browser voices.
- **History and exports:** saves conversations in the browser and exports responses as PDF or Word documents.
- **Configurable services:** supports separate provider and model settings for explanations, document extraction and speech transcription.

## Report, Model and Dataset Downloads

| Resource | Link |
|---|---|
| Project report, including methodology, experiments, results, bias analysis and limitations | [Read the report](Report/TrueCheck_Report.pdf) |
| Trained BERT checkpoint and tokenizer | [View or download TrueCheck-BERT-Model.zip](https://drive.google.com/file/d/145FlmMlZr_pB1jyh7CYcJEFH1Mi-QgBH/view?usp=drive_link) |
| Original FEVER training claims | [Download train.jsonl](https://fever.ai/download/fever/train.jsonl) |
| Original FEVER Wikipedia evidence archive | [Download wiki-pages.zip](https://fever.ai/download/fever/wiki-pages.zip) |

Place the original FEVER files in `Dataset/Original/` and keep `wiki-pages.zip` zipped. Claim cleaning uses `train.jsonl`. Resolving annotated evidence and building the retrieval corpus also require `wiki-pages.zip`. The web-app guide includes these steps for a fresh clone, and the recreation guide explains them as separate stages.

The fine-tuned `bert-base-uncased` classifier receives a claim and English evidence. It predicts one of two classes, with `0` mapped to SUPPORTS and `1` mapped to REFUTES. Both classes use the same checkpoint files. There is no separate model file for each class.

## Project File Guide

### Main Folders

| Folder | Contents |
|---|---|
| `Backend/` | FastAPI server, local inference, provider integrations and report generation. |
| `Frontend/` | Browser interface, styles, JavaScript and application icons. |
| `Python Scripts/` | Dataset preparation, model training and evaluation scripts. |
| `Retrievers/` | Corpus construction and retrieval implementations. |
| `config/` | Dataset preparation and training configuration. |
| `Dataset/` | Original downloads, processed splits, caches and retrieval indexes. Generated locally and excluded from Git. |
| `Models/` | Saved models, tokenizers, training metadata and run pointers. Excluded from Git. |
| `Evaluation/` | Saved model metrics, predictions, confusion matrices and retrieval diagnostics. |
| `Report/` | Project report PDF. |

### Backend Python Files

The backend modules are loaded by the application. Start the server through Uvicorn using the command in the run guide.

| File | Purpose | How it is used |
|---|---|---|
| [app.py](Backend/app.py) | Defines the FastAPI application, serves the frontend and exposes verification, settings, upload, speech and export routes. | Server entry point through `uvicorn Backend.app:app`. |
| [verification.py](Backend/verification.py) | Coordinates claim preparation, retrieval, classification, explanations and progress updates. | Loaded during application verification. |
| [bert_model.py](Backend/bert_model.py) | Loads BERT, splits long evidence into chunks and combines chunk outputs into one prediction. | Loaded by the verification workflow. |
| [retrieval.py](Backend/retrieval.py) | Loads the hybrid retriever and checks that the corpus and indexes match. | Loaded by the verification workflow. |
| [llm.py](Backend/llm.py) | Screens claims, prepares English translations, generates cited explanations and validates provider responses. | Uses the saved Explanations configuration. |
| [configuration.py](Backend/configuration.py) | Manages provider settings, browser sessions, request authorisation and saved Windows credentials. | Used by API routes and provider clients. |
| [extraction.py](Backend/extraction.py) | Validates image/PDF uploads and extracts visible text through the configured provider. | Used when a document is uploaded. |
| [speech.py](Backend/speech.py) | Validates recorded audio and requests OpenAI speech transcription. | Used for voice input. |
| [reports.py](Backend/reports.py) | Validates saved results and builds PDF and Word reports locally. | Used when exporting a response. |
| [languages.py](Backend/languages.py) | Defines fixed English, Portuguese and Afrikaans application messages. | Imported by backend components. |
| [settings.py](Backend/settings.py) | Defines checkpoint/index paths, device selection, input limits, timeouts and session settings. | Edit asset paths here when selecting another checkpoint. |
| [__init__.py](Backend/__init__.py) | Marks `Backend` as a Python package. | Imported automatically. |

### Data, Training and Evaluation Python Files

| File | Purpose | How it is used |
|---|---|---|
| [prepare_dataset.py](Python%20Scripts/prepare_dataset.py) | Cleans FEVER claims, removes duplicate/conflicting groups, assigns stratified splits and saves binary claim records. | Run first during data preparation. Also provides shared dataset and integrity helpers. |
| [prepare_evidence.py](Python%20Scripts/prepare_evidence.py) | Resolves annotated Wikipedia evidence, updates the processed CSVs and provides evidence chunking helpers. | Run after claim preparation and before training or corpus construction. |
| [train_classical.py](Python%20Scripts/train_classical.py) | Trains TF-IDF with Logistic Regression, selects regularisation using validation results and saves a locked run. | Run for the classical model. Also provides shared metrics and run-management helpers. |
| [train_transformer.py](Python%20Scripts/train_transformer.py) | Fine-tunes BERT, selects the best validation checkpoint and saves training progress and a locked run. | Run for the BERT model. Supports resuming an interrupted run. |
| [evaluate_models.py](Python%20Scripts/evaluate_models.py) | Evaluates completed, locked model runs on the test split and saves metrics, predictions and matrices. | Run after training. Requires complete run metadata and matching prepared data. |
| [evaluate_general_claims.py](Python%20Scripts/evaluate_general_claims.py) | Compares 1 to 10 retrieved passages using an editable 200-claim diagnostic. | Optional local retrieval/BERT evaluation after application assets are ready. |

### Retriever Python Files

| File | Purpose | How it is used |
|---|---|---|
| [common.py](Retrievers/common.py) | Builds the sentence corpus from cited Wikipedia pages and sampled distractor pages. Also provides shared corpus loading helpers. | Run after evidence preparation to create `passages.jsonl`. |
| [bm25_retriever.py](Retrievers/bm25_retriever.py) | Builds and searches the saved BM25 keyword index. | Run with `--build` to create the index. Run without it for sample searches. Imported by the hybrid retriever. |
| [embedding_retriever.py](Retrievers/embedding_retriever.py) | Builds and searches the MiniLM sentence embedding index using FAISS. | Run with `--build` to create the index. Run without it for sample searches. Imported by the hybrid retriever. |
| [hybrid_retriever.py](Retrievers/hybrid_retriever.py) | Combines BM25 and embedding rankings using reciprocal rank fusion. | Imported by the application. It has no standalone command. |
| [tfidf_retriever.py](Retrievers/tfidf_retriever.py) | Builds and searches a TF-IDF retrieval index for comparison. | Optional. Run with `--build` to create the index or without it for sample searches. |

### Frontend Files

FastAPI serves these files directly. There is no separate frontend build step.

| File | Purpose |
|---|---|
| [index.html](Frontend/index.html) | Application layout, navigation, claim input, settings and review dialogs. |
| [style.css](Frontend/style.css) | Colours, typography, responsive layouts and control styles. |
| [app.js](Frontend/app.js) | Main interface behaviour, submissions, result display, readiness checks and cancellation. |
| [history.js](Frontend/history.js) | Validates and saves conversations in browser local storage. |
| [settings.js](Frontend/settings.js) | Handles provider configuration forms and authenticated API requests. |
| [speech.js](Frontend/speech.js) | Handles microphone recording, transcription and browser read aloud. |
| [export.js](Frontend/export.js) | Requests and downloads PDF and Word reports. |
| [favicon.ico](Frontend/favicon.ico) | Browser tab icon. |
| [assets/logo.svg](Frontend/assets/logo.svg) | Vector interface logo and SVG favicon. |
| [assets/logo.png](Frontend/assets/logo.png) | Raster logo used as the Apple touch icon. |

### Configuration and Checkpoint Files

| File | Purpose |
|---|---|
| [Backend/requirements.txt](Backend/requirements.txt) | Python dependencies for the server, local inference and provider integrations. |
| [config/dataset_and_training_settings.json](config/dataset_and_training_settings.json) | Seed, preparation policy, pinned base model revision and training hyperparameters. |
| Checkpoint `config.json` | BERT architecture and the mapping between class identifiers and labels. |
| Checkpoint `model.safetensors` | Trained BERT and classification-layer weights for both classes. |
| Checkpoint `tokenizer.json` | Vocabulary and tokenisation rules for claim and evidence text. |
| Checkpoint `tokenizer_config.json` | Tokenizer settings, including lowercasing, special tokens and the 512-token input limit. |

## Requirements

- Windows 11 and Python 3.12 for the setup below.
- Git to clone the repository.
- Internet access for package installation, asset downloads and external API services.
- For the web application, an API key with access to a supported explanation model. Document extraction and voice input have separate optional configurations.
- Storage for the FEVER downloads, prepared evidence corpus, indexes and BERT checkpoint.

A CPU can run the application. A compatible NVIDIA GPU is optional. BERT uses CUDA automatically when it is available in the installed PyTorch build.

Recreating the project also requires the original FEVER downloads and additional training/evaluation dependencies listed below. A compatible GPU is recommended for BERT fine-tuning.

Dataset preparation, model training, index construction and the local evaluations do not require provider API keys. Their initial package and model downloads require internet access.

The frontend is served by FastAPI. No Node.js installation or separate frontend server is required.

## Run the TrueCheck Web Application

This path uses the supplied trained BERT checkpoint. Model training is not required, but the matching retrieval corpus, indexes and cached MiniLM model must also be available.

This guide contains every setup command needed for a fresh clone. The recreation guide can be skipped when the goal is to run the application with the supplied model.

Run the following commands in PowerShell. All commands after cloning use the repository root as the working directory.

### 1. Clone and Install Dependencies

```powershell
git clone https://github.com/Mario336-cmd/NAT-Group-Project.git
cd NAT-Group-Project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Backend\requirements.txt
```

The commands use the virtual environment's Python directly, so activation is optional.

### 2. Download the BERT Classifier

`Dataset/` and `Models/` are excluded from Git. A fresh clone needs both the classifier checkpoint and the retrieval assets described below.

Download [TrueCheck-BERT-Model.zip](https://drive.google.com/file/d/145FlmMlZr_pB1jyh7CYcJEFH1Mi-QgBH/view?usp=drive_link) and extract these four files directly into this location:

```text
Models/binary_evidence_bert/runs/20260928T123142_b12ec98f/checkpoint/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
```

Avoid an extra nested folder inside `checkpoint/`. The application uses the path defined in [Backend/settings.py](Backend/settings.py) and does not automatically select another training run. Training is not required when using the supplied checkpoint.

### 3. Restore or Build the Retrieval Assets

The BERT download does not include the evidence corpus or retrieval indexes. Choose one of the following options.

#### Option A: Restore Existing Assets

If a matching saved corpus and its indexes are available, copy them to these locations:

```text
Dataset/Index/
  passages.jsonl
  embeddings.faiss
  bm25/
    data.csc.index.npy
    indices.csc.index.npy
    indptr.csc.index.npy
    params.index.json
    vocab.index.json
```

Keep the corpus and its indexes together, then complete the MiniLM cache setup below.

#### Option B: Build Assets from the Original FEVER Files

For a fresh clone without saved retrieval assets, download both files and create `Dataset/Original/`:

| File | Download |
|---|---|
| Original FEVER training claims | [train.jsonl](https://fever.ai/download/fever/train.jsonl) |
| Original FEVER Wikipedia evidence archive | [wiki-pages.zip](https://fever.ai/download/fever/wiki-pages.zip) |

```text
Dataset/Original/
  train.jsonl
  wiki-pages.zip
```

Keep the Wikipedia archive zipped. From the repository root, run these commands in order:

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn==1.8.0
.\.venv\Scripts\python.exe "Python Scripts/prepare_dataset.py"
.\.venv\Scripts\python.exe "Python Scripts/prepare_evidence.py"
.\.venv\Scripts\python.exe Retrievers/common.py
.\.venv\Scripts\python.exe Retrievers/bm25_retriever.py --build
.\.venv\Scripts\python.exe Retrievers/embedding_retriever.py --build
```

These commands prepare the claim records, resolve Wikipedia evidence and build the passage corpus and its indexes. They do not train either classifier. The preparation scripts download the pinned BERT tokenizer, while the embedding build downloads and caches MiniLM.

Initial Wikipedia processing and passage embedding can take time. The commands replace their generated datasets or index files. If the corpus is rebuilt, rebuild both retrieval indexes against it.

#### MiniLM Cache Setup

Option B caches MiniLM automatically. If existing indexes were restored through Option A on a new machine, run this once while online:

```powershell
@'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
'@ | .\.venv\Scripts\python.exe -
```

Run this under the same Windows account and Python environment used to start the application. The application loads the cached model offline.

### 4. Start the Application

From the repository root, start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn Backend.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and wait for the Retriever and Classifier setup to finish. Keep the terminal running while using the application. Press `Ctrl+C` to stop it.

After changing backend code, restart the server and refresh the browser with `Ctrl+F5`.

### 5. Configure API Services

Open **Settings**, choose the provider, enter the exact API model name and API key, then select **Save configuration** for each required service.

| Setting | Supported providers | Model used in the project demonstrations | Required for |
|---|---|---|---|
| Explanations | OpenAI or Anthropic | OpenAI `gpt-6-luna` | Claim screening, language preparation and explanations |
| Image/PDF text extraction | OpenAI or Anthropic | Anthropic `claude-sonnet-5-5` | Image and PDF uploads |
| Voice input | OpenAI | OpenAI `gpt-transcribe` | Recorded speech transcription |

The **Explanations** configuration also handles claim screening and translation, so a working configuration is required for typed English claims. Extraction and voice input can be configured separately when needed. Provider and model choices can be changed in Settings. Choose an accessible model that supports the selected task, including structured output for explanations and image/PDF input for extraction.

Read aloud uses the browser's available voices and does not require an application API key. Saved API credentials use Windows Credential Manager. Conversation history is stored separately in browser local storage.

Retrieval and BERT classification run locally after the assets are available. Claim preparation, explanations, document extraction and speech transcription use the configured external APIs.

### 6. Use TrueCheck

1. Enter a short factual claim or select an example on the home screen.
2. Leave **Evidence passages to use** at 1, or select another value from 1 to 10.
3. Submit the claim and review the prediction, confidence, explanation, citations and evidence.
4. Reopen a saved conversation through **History**, or use the response controls to copy, export or read it aloud.

For document input, use the attachment button and review the extracted text before submitting a claim. Supported files are PNG, JPEG, WebP and unencrypted PDFs up to 10 MiB. PDFs can contain up to 10 pages. Uploaded documents provide claim text and are not added to the evidence corpus.

For voice input, use the microphone button, allow microphone access and review the transcript before submission.

Model confidence is a model score, not a guarantee of accuracy. Verification uses the saved FEVER Wikipedia corpus, so evidence may be missing or outdated. Review the retrieved passages and explanation when interpreting a prediction.

## Recreate Data Preparation, Models and Evaluations

This path provides separate stages for cleaning the original data, preparing evidence, training the two final binary claim-and-evidence models and running the available evaluations. Complete the stages needed for the chosen task. Classical and BERT training are independent options after the shared evidence preparation.

Use a fresh clone to keep new runs separate from existing artefacts. Exact scores can vary with hardware and package versions. All commands use PowerShell and run from the repository root after cloning.

### 1. Set Up the Reproduction Environment

For an existing clone and virtual environment, skip the clone and environment-creation commands.

```powershell
git clone https://github.com/Mario336-cmd/NAT-Group-Project.git
cd NAT-Group-Project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install matplotlib==3.10.8 scikit-learn==1.8.0 tzdata
```

Preparation and training settings are defined in [config/dataset_and_training_settings.json](config/dataset_and_training_settings.json). Review the configuration before preparing data or starting a training run.

### 2. Clean and Split FEVER

**Prerequisite:** the reproduction environment from stage 1.

Download the original files and create `Dataset/Original/`. Only `train.jsonl` is needed for claim cleaning. The Wikipedia archive is needed for evidence preparation in stage 3.

| File | Download | Needed for |
|---|---|---|
| Training claims | [train.jsonl](https://fever.ai/download/fever/train.jsonl) | Claim cleaning and splitting. |
| Wikipedia evidence archive | [wiki-pages.zip](https://fever.ai/download/fever/wiki-pages.zip) | Resolving annotated evidence and constructing the retrieval corpus. |

```text
Dataset/Original/
  train.jsonl
  wiki-pages.zip
```

Run claim preparation:

```powershell
.\.venv\Scripts\python.exe "Python Scripts/prepare_dataset.py"
```

**Outputs:** `Dataset/Processed/train.csv`, `validation.csv` and `test.csv`, containing the cleaned binary claims and their evidence annotations. The pinned BERT tokenizer is downloaded and saved under `Dataset/Cache/`.

The script checks and removes duplicate/conflicting groups, creates stratified 70/15/15 split assignments and excludes NOT ENOUGH INFO from the binary records. Running it again replaces the processed splits. Model training is unnecessary when only the cleaned data is required.

### 3. Prepare Annotated Evidence

**Prerequisites:** stage 2 and `Dataset/Original/wiki-pages.zip`. Keep the archive zipped.

```powershell
.\.venv\Scripts\python.exe "Python Scripts/prepare_evidence.py"
```

**Outputs:** a Wikipedia lookup database at `Dataset/Cache/fever_wiki_all_sets.sqlite3` and updated processed CSVs containing resolved evidence text and source references.

Both training options below use these evidence-prepared CSVs. Classifier training and test evaluation do not require retrieval indexes. Initial Wikipedia processing can take time.

### 4. Train the Classical Model

**Prerequisite:** the evidence-prepared data from stage 3.

```powershell
.\.venv\Scripts\python.exe "Python Scripts/train_classical.py" --experiment binary_evidence_classical
```

This trains TF-IDF with Logistic Regression on the CPU and selects regularisation using validation macro F1 score.

**Outputs:** the fitted model and run metadata under `Models/binary_evidence_classical/runs/<run_id>/`, plus validation results under `Evaluation/binary_evidence_classical/runs/<run_id>/validation/`. Successful training locks the run and updates `Models/binary_evidence_classical/completed.json`.

### 5. Fine-Tune BERT

**Prerequisite:** the evidence-prepared data from stage 3. Classical training is optional.

```powershell
.\.venv\Scripts\python.exe "Python Scripts/train_transformer.py" --experiment binary_evidence_bert --device auto
```

This downloads the pinned pretrained `bert-base-uncased` model, fine-tunes it and selects the checkpoint using validation macro F1 score.

**Outputs:** the checkpoint and run metadata under `Models/binary_evidence_bert/runs/<run_id>/`, plus validation results under `Evaluation/binary_evidence_bert/runs/<run_id>/validation/`. Successful training locks the run and updates `Models/binary_evidence_bert/completed.json`.

For an interrupted BERT run, resume using its saved run identifier:

```powershell
.\.venv\Scripts\python.exe "Python Scripts/train_transformer.py" --experiment binary_evidence_bert --device auto --resume RUN_ID
```

Replace `RUN_ID` with the interrupted run's folder name. Resuming requires the saved training state and matching code, configuration, prepared data, package environment and device identity.

### 6. Evaluate Completed Models

**Prerequisites:** the completed, locked run for each model being evaluated and its matching evidence-prepared data. Keep the complete run directories and metadata. Choose the command for the model or models that were trained.

For the classical model only:

```powershell
.\.venv\Scripts\python.exe "Python Scripts/evaluate_models.py" --experiment binary_evidence_classical
```

For BERT only:

```powershell
.\.venv\Scripts\python.exe "Python Scripts/evaluate_models.py" --experiment binary_evidence_bert --device auto
```

For both models, after completing stages 4 and 5:

```powershell
.\.venv\Scripts\python.exe "Python Scripts/evaluate_models.py" --experiment all --device auto
```

Evaluation uses the test split. Here, `all` refers to the two final binary claim-and-evidence models in this repository. The evaluator uses each model's `completed.json` to select its locked run.

Test outputs are saved under `Evaluation/<experiment>/runs/<run_id>/test/`:

| File | Contents |
|---|---|
| `metrics.json` | Overall and per-class metrics with evaluation identity metadata. |
| `predictions.csv` | Expected labels, predicted labels and class probabilities for individual claims. |
| `confusion_matrix.csv` | Raw confusion matrix counts. |
| `confusion_matrix.png` | Raw-count confusion matrix figure. |
| `confusion_matrix_normalized.png` | Row-normalised confusion matrix figure. |

Training produces corresponding validation outputs. Evaluation checks the prepared data, training scripts and locked model files. Preserve these files after training. Matching existing test results are reused, while mismatched results trigger an error. The four downloadable BERT checkpoint files support application inference but do not include the complete training run needed by this evaluator.

### 7. Build the Corpus and Retrieval Indexes

**Prerequisite:** evidence preparation from stage 3. Model training and test evaluation are optional for this stage.

Build the corpus and both indexes in order:

```powershell
.\.venv\Scripts\python.exe Retrievers/common.py
.\.venv\Scripts\python.exe Retrievers/bm25_retriever.py --build
.\.venv\Scripts\python.exe Retrievers/embedding_retriever.py --build
```

The first command writes `Dataset/Index/passages.jsonl`. The next two create its BM25 and FAISS indexes. The embedding build downloads and caches the pinned MiniLM model. Corpus construction and indexing can be completed independently of classifier training once evidence preparation has finished.

For the optional TF-IDF retrieval comparison, also build its index:

```powershell
.\.venv\Scripts\python.exe Retrievers/tfidf_retriever.py --build
```

Saved retriever comparison artefacts are available under `Evaluation/retrievers/`. The repository includes the retrieval implementations and index builders, but no aggregate comparison runner.

### 8. Optional Evidence-Count Diagnostic

**Prerequisites:** the retrieval assets from stage 7 and an available BERT checkpoint configured in `Backend/settings.py`. Either use the supplied downloadable checkpoint or select a newly trained checkpoint as explained in stage 9.

Compare 1 to 10 retrieved passages:

```powershell
.\.venv\Scripts\python.exe "Python Scripts/evaluate_general_claims.py" --seed 42 --device auto --output-dir Evaluation/general_claims_topk_new
```

This uses the script's 200 labelled general claims and writes `results.json`, `performance.png`, `class_recall.png` and `confusion_matrices.png` to the specified directory. The separate output directory preserves the existing diagnostic results.

The selection split chooses the candidate passage count, while the reserved check split evaluates that choice. The diagnostic uses local retrieval and BERT inference, makes no explanation API calls and does not change the application's default passage count.

### 9. Use a Newly Trained BERT Checkpoint in the Application

**Prerequisites:** a completed BERT run from stage 5 and the retrieval assets from stage 7.

Read `Models/binary_evidence_bert/completed.json` to find the completed BERT run identifier. In [Backend/settings.py](Backend/settings.py), replace the run identifier in `checkpoint_dir` with that value:

```python
checkpoint_dir: Path = (
    ROOT / "Models" / "binary_evidence_bert" / "runs"
    / "NEW_RUN_ID" / "checkpoint"
)
```

Replace `NEW_RUN_ID` with the actual folder name. The application loads BERT for predictions, while the classical model is retained for comparison. Start or restart the server using the [application run instructions](#4-start-the-application), then [configure the required API services](#5-configure-api-services).

## Troubleshooting

| Problem | What to check |
|---|---|
| `py -3.12` is not recognised or Python 3.12 is unavailable | Install Python 3.12 with the Windows Python launcher, then reopen PowerShell. |
| Classifier setup fails | Check that all four checkpoint files are directly inside the configured `checkpoint/` directory. |
| Retriever setup fails | Check that `passages.jsonl`, `embeddings.faiss` and the complete `bm25/` folder exist and were built together. |
| MiniLM cannot be loaded offline | Run the MiniLM cache setup above while online under the account used to start the application. |
| Screening or explanations are unavailable | Check the saved Explanations provider, exact model name, API key and account access. |
| Upload or voice controls are unavailable | Save the separate extraction or voice configuration. Voice recording also needs browser microphone permission. |
| Port 8000 is already in use | Stop the previous server with `Ctrl+C`, or run the command with `--port 8001` and open the matching address. |
| A change is not reflected in the browser | Restart the backend and refresh with `Ctrl+F5`. |
| Evaluation rejects a model run | Use a complete, locked training run with its matching prepared data and unchanged training scripts. Downloaded checkpoint files alone are insufficient. |
| The application still uses the older BERT model | Update `checkpoint_dir` in `Backend/settings.py` to the new run identifier, then restart the server. |
| Training or evaluation reports a missing dependency | Install the additional dependencies in the reproduction environment setup. |
| BERT training was interrupted | Resume the named run with `--resume RUN_ID` using the matching environment, data and saved training state. |

To force CPU inference, set the following in the same PowerShell window before starting the server:

```powershell
$env:TRUECHECK_DEVICE = "cpu"
```
