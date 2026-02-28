# MOF Property Prediction Tool (C++ ML Pipeline)

A **C++-first**, end-to-end machine learning pipeline for predicting **Metal–Organic Framework (MOF)** properties (e.g., **CO₂ uptake**, **surface area**) from tabular structural descriptors.

This repo is intentionally built as a *clean C++ project* (CMake + unit tests + modular pipeline). It is suitable for:

- fast prototyping of MOF screening workflows
- learning how to implement a full ML pipeline in C++
- extending toward CIF parsing / descriptor generation / more advanced models

---

## What this project does

Given a CSV where each row is a MOF and columns contain **numeric descriptors** plus a **target** property, the tool:

1. **Loads** the CSV (header optional, configurable delimiter)
2. **Preprocesses** data (keeps numeric columns, handles missing values)
3. **Builds features** (descriptor selection + derived features)
4. **Splits** train/test
5. **Scales** features (z-score)
6. **Trains** a regression model
7. **Evaluates** with RMSE / MAE / R²
8. **Exports artifacts** (cleaned data, engineered features, predictions, metrics report, optional saved model)

---

## Repository layout

> The repository includes a few convenience folders (e.g., `TEMP/`, `build/`) that you may want to exclude in a “clean” Git repo. The **core project** is `src/`, `include/`, `tests/`, `data/`.

```text
.
├─ include/                    # Public headers (pipeline modules)
│  ├─ preprocessing.h
│  ├─ feature_engineering.h
│  ├─ modeling.h
│  ├─ evaluation.h
│  └─ utils.h
├─ src/                        # Implementation
│  ├─ main.cpp                 # CLI + pipeline orchestration
│  ├─ preprocessing.cpp        # numeric dataset creation + imputation summary
│  ├─ feature_engineering.cpp  # derived features + scaling + export
│  ├─ modeling.cpp             # models + save/load
│  ├─ evaluation.cpp           # RMSE/MAE/R2
│  └─ utils.cpp                # matrix helpers + small utilities
├─ tests/                      # Unit tests + manual pipeline test
├─ data/                       # Example CSV datasets
│  ├─ mof_sample_clean.csv
│  ├─ mof_sample_dirty.csv
│  ├─ mof_realistic_120.csv
│  └─ mof_stress_145.csv
├─ output/                     # Default output root (generated)
├─ CMakeLists.txt
├─ README.md
└─ requirements.txt            # (Currently unused by core C++ build)
```

---

## Build requirements

- **CMake** ≥ 3.16
- A C++ compiler with **C++17** support
  - Linux/macOS: GCC / Clang
  - Windows: MSVC

Optional (nice-to-have):

- `ctest` (comes with CMake) for running tests

---

## Build

From the project root:

```bash
cmake -S . -B build -DCMAKE_CXX_STANDARD=17
cmake --build build --config Release
```

The executable will be:

- Linux/macOS: `./build/mof_prediction_tool`
- Windows (MSVC): `./build/Release/mof_prediction_tool.exe` (or similar)

---

## Run (CLI)

### Minimal example

```bash
./build/mof_prediction_tool \
  --input data/mof_sample_clean.csv \
  --target co2_uptake_mmolg
```

### Choose a model

```bash
./build/mof_prediction_tool \
  --input data/mof_sample_clean.csv \
  --target co2_uptake_mmolg \
  --model linear
```

Supported model names:

- `linear`  ✅ **implemented** (ordinary least squares with optional intercept)
- `rf`      ⚠️ placeholder baseline (currently predicts mean of training targets)
- `svm`     ⚠️ placeholder baseline (currently predicts mean of training targets)
- `nn`      ⚠️ placeholder baseline (currently predicts mean of training targets)

> Notes:
> - Only **LinearRegressionModel** is a real model in the current codebase.
> - The other model types exist to keep the architecture future-ready.

### Common options

```bash
./build/mof_prediction_tool \
  --input data/mof_sample_dirty.csv \
  --target 4 \
  --no-header \
  --delimiter ',' \
  --test-ratio 0.25 \
  --seed 123 \
  --outdir output
```

#### CLI reference

| Flag | Meaning | Default |
|---|---|---|
| `--input <path>` | Path to CSV input | required |
| `--target <name\|index>` | Target column name (if header) or **0-based index** | required |
| `--model <type>` | `linear \| rf \| svm \| nn` | `linear` |
| `--outdir <dir>` | Output directory root | `output` |
| `--test-ratio <float>` | Test split ratio in `(0,1)` | `0.2` |
| `--seed <int>` | RNG seed for splitting/shuffling | `42` |
| `--delimiter <char>` | CSV delimiter | `,` |
| `--no-header` | Treat CSV as headerless (target must be index) | off |
| `--no-square-features` | Disable derived `x²` features | off |
| `--pairwise-product` | Enable pairwise products `xi * xj` | off |
| `--max-pairwise <n>` | Cap number of pairwise features | `64` |
| `--no-save-model` | Do not write `output/models/model.kv` | off |

---

## Input data format

### Expected CSV structure

- Each row = one MOF sample
- Columns = numeric descriptors + one target column

Example (`data/mof_sample_clean.csv`):

```csv
surface_area_m2g,pore_volume_cm3g,density_gcm3,pld_A,co2_uptake_mmolg
850,0.42,1.10,5.1,2.8
920,0.50,1.05,5.8,3.4
```

### Missing values handling

During CSV parsing/preprocessing, the following tokens are treated as missing:

- empty cell
- `nan`, `na`, `null`, `none`, `?` (case-insensitive)

The preprocessing step builds a **numeric-only dataset**, imputes missing numeric features (see report), and drops unusable rows when needed.

---

## Output artifacts

By default, outputs go to `output/` (or `--outdir <dir>`):

```text
output/
├─ data/
│  ├─ processed/
│  │  └─ cleaned.csv
│  └─ features/
│     ├─ engineered_all_unscaled.csv
│     ├─ train_scaled.csv
│     └─ test_scaled.csv
├─ reports/
│  ├─ predictions.csv
│  └─ metrics.txt
└─ models/
   └─ model.kv
```

### What each file is

- `data/processed/cleaned.csv`  
  Numeric-only dataset after preprocessing.

- `data/features/engineered_all_unscaled.csv`  
  Same dataset with derived features (before scaling).

- `data/features/train_scaled.csv`, `test_scaled.csv`  
  Train/test splits after z-score scaling.

- `reports/predictions.csv`  
  Test-set ground truth + predictions, with original row indices.

- `reports/metrics.txt`  
  A readable run report, including:
  - preprocessing stats (kept rows, missing values)
  - feature counts (original vs derived)
  - model name
  - train metrics (optional)
  - test metrics (RMSE, MAE, R²)

- `models/model.kv`  
  Saved model in a simple **key=value** format.
  - For linear regression: coefficients + intercept are saved.
  - For placeholder models: the stored value is the mean target.

---

## Pipeline internals (modules)

### 1) Preprocessing (`preprocessing.*`)

Responsibilities:

- convert CSV → `NumericDataset`
- keep only numeric columns (besides target)
- detect missing values and impute where possible
- produce a `PreprocessSummary` (counts, missingness, etc.)

### 2) Feature engineering (`feature_engineering.*`)

Includes:

- descriptor selection (`DescriptorSelectionMode::All` by default)
- derived feature generation
  - squares (`x²`) enabled by default
  - optional pairwise products (`xi * xj`) with a cap (`--max-pairwise`)
- feature scaling
  - current default: **z-score** scaling (fit on train, applied to test)
- export helpers for saving matrices to CSV

### 3) Modeling (`modeling.*`)

Design:

- `IRegressionModel` interface
- `create_model(ModelType)` factory
- `fit / predict / evaluate` methods

Current implementation status:

- **LinearRegressionModel**: implemented (OLS-style regression)
- RandomForest / SVM / NeuralNet: placeholders (mean regressor baseline)

### 4) Evaluation (`evaluation.*`)

Regression metrics:

- RMSE
- MAE
- R²

### 5) Utilities (`utils.*`)

Small helpers (matrix operations, file IO helpers, checks, etc.).

---

## Running tests

This repo uses plain C++ test executables registered through CTest.

### Build with tests

```bash
cmake -S . -B build \
  -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_TESTING=ON \
  -DMOF_BUILD_TESTS=ON

cmake --build build --config Release
```

### Run

```bash
ctest --test-dir build --output-on-failure
```

Tests live in `tests/` and cover preprocessing, feature engineering, modeling, and evaluation.

---

## Practical tips

### Picking the target column

- If your CSV has a header row: `--target co2_uptake_mmolg`
- If it does **not** have a header row: `--no-header --target 4` (0-based index)

### Start simple

For early experiments, use:

- `data/mof_sample_clean.csv`
- `--model linear`
- defaults for splitting/scaling

### Making the project “clean” for GitHub

If you plan to publish the repo, consider excluding:

- `build/`
- `TEMP/`
- `output/`

via a `.gitignore`.

---

## Roadmap (suggested extensions)

If you want to push this toward a research-grade MOF predictor:

1. **CIF parsing + descriptor generation**
   - parse CIFs and compute common MOF descriptors (PLD, LCD, surface area, density, pore volume)
2. **Better models in C++**
   - implement real Random Forest / SVR / MLP
   - add cross-validation + hyperparameter search
3. **Feature importance + interpretability**
   - permutation importance, SHAP-style approximations
4. **Reproducible experiment tracking**
   - config files (YAML/JSON), run IDs, and artifact folders

---

## License

No license is specified in this repository yet.

If you intend to share publicly, add a `LICENSE` file (e.g., MIT, Apache-2.0) and update this section.

---

## Citation

If you use this codebase in academic work, cite it as software (and cite your MOF dataset sources). A basic placeholder citation:

```bibtex
@software{mof_prediction_tool,
  title  = {MOF Property Prediction Tool (C++ ML Pipeline)},
  year   = {2026},
  note   = {C++ ML pipeline for MOF property prediction from descriptors}
}
```
