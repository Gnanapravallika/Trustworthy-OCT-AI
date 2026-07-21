# TrustOCT Framework TODO List

## Task Board Status
| ID | Task | Status |
| :--- | :--- | :--- |
| **T001** | Repository Setup & Skeleton | ✅ Completed |
| **T002** | Configuration System | ✅ Completed |
| **T003** | Dataset Verification (`verify.py`) | ✅ Completed |
| **T004** | Dataset Statistics (`statistics.py`) | ✅ Completed |
| **T005** | Dataset Class (`kermany.py` / `base_dataset.py`) | ✅ Completed |
| **T006** | DataLoader & Factory (`loader.py` / `factory.py`) | ✅ Completed |
| **T007** | Custom transforms (`transforms.py`) | ✅ Completed |
| **T008** | Unit Tests (`tests/test_pipeline.py`) | ✅ Completed |
| **T009** | Baseline ResNet50 Classifier | ⏳ Pending |

---

## Phase 1: Setup & Initialization (Completed ✅)
- [x] Propose V3.0 codebase architecture
- [x] Create project documents (`PROJECT_SPECIFICATION.md`, `RESEARCH_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `EXPERIMENT_PLAN.md`, `TODO.md`)
- [x] Create repository skeleton (empty modules and directory layouts)
- [x] Write `requirements.txt` and configs templates

## Phase 2: Pipeline Development (Completed ✅)
- [x] Set up data preprocessing (Bilateral filter, CLAHE) in `src/preprocessing/`
- [x] Set up dataset loaders in `src/datasets/`
- [x] Implement unit tests and verify pipeline in `tests/test_pipeline.py`
- [x] Create master Google Colab setup notebook ([TrustOCT_Setup.ipynb](file:///e:/Trustworthy-OCT-AI/notebooks/TrustOCT_Setup.ipynb))

## Phase 3: Model Training & Baselines (Active ⏳)
- [ ] Implement backbone modules and head classes in `src/models/`
- [ ] Implement builder registry in `src/registry/`
- [ ] Write evidential loss function in `src/losses/`
- [ ] Write training and trainer loop in `src/train/`
- [ ] Execute Baseline training (EXP001)
- [ ] Execute Ablation studies (EXP002 - EXP006)
- [ ] Execute Proposed TrustOCT training (EXP007)

## Phase 4: Verification & Comparison
- [ ] Compute ECE, calibration, and selective prediction statistics
- [ ] Produce plots and reliability diagrams in `paper/figures/`
- [ ] Format classification tables in `paper/tables/`
- [ ] Compile explainability outputs (LayerCAM vs Grad-CAM)

