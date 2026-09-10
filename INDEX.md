# SmartSheet — Competition Documentation Package

This package contains an evidence-aware bilingual documentation set for the SmartSheet OMR project.

## Files

- `01_English/SmartSheet_Technical_Documentation_EN.md` — competition-ready English technical documentation.
- `02_Persian/SmartSheet_Technical_Documentation_FA.md` — full Persian reference version.
- `03_Bilingual/README_Bilingual.md` — bilingual submission guidance.
- `04_Project_Evidence/new_benchmark.json` — six-scenario stress-test evidence.
- `04_Project_Evidence/DOCUMENTATION_original.md` — original documentation retained for traceability.

## Important evidence rule

The package distinguishes the historical **95.2% controlled benchmark** from the later **six-scenario stress test (71.7% weighted result)**. They are different test sets and must not be presented as one combined benchmark.

The 95.2% figure is retained as a historical controlled-project result. The newer stress test is included because it reveals specific robustness weaknesses, especially in one-column and severe-perspective scenarios.

## AI terminology

The current core is deterministic computer vision/image processing with OpenCV. Do not claim that the current implementation contains a trained CNN or deep-learning model. Deep-learning-assisted detection is documented as a future development direction.

## Submission rule

Before uploading the final competition ZIP, replace every placeholder repository URL, obsolete product name, or file-structure claim with information verified against the actual project package.
