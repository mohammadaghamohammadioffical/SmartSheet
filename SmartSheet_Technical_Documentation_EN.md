# SmartSheet — Technical & Competition Documentation

**Project type:** Automated OMR (Optical Mark Recognition) grading system  
**Core technology:** Computer Vision + OpenCV + deterministic decision logic  
**Primary application:** Multiple-choice assessment  
**SDG alignment:** SDG 4 — Quality Education

> **Submission integrity note:** The current core implementation is a computer-vision/image-processing system. It is not a trained deep-learning model. For a competition submission, SmartSheet should therefore be presented as an AI-related computer-vision solution, without claiming CNN/deep-learning functionality that is not currently implemented.

---

## 1. Executive Summary

SmartSheet is a software-based OMR grading system that converts a photographed or scanned multiple-choice answer sheet into structured grading results. The system detects the page, corrects geometry when possible, identifies answer-bubble candidates, reconstructs the expected question grid, scores each option using multiple visual features, and compares the detected answers with an answer key.

A central design principle is **conservative automation**: when the visual evidence is weak, conflicting, or structurally incomplete, the system can report `blank`, `ambiguous`, `multi_mark`, or `not_detected` instead of silently inventing an answer or shifting question numbers.

The project is intended to reduce repetitive grading work and make software-based OMR more accessible without requiring dedicated OMR scanning hardware.

---

## 2. Problem Statement

Manual multiple-choice grading is repetitive and becomes increasingly expensive in time as the number of students and questions grows. Conventional OMR workflows can also depend on dedicated scanners or tightly controlled forms.

SmartSheet explores a software-first alternative: use ordinary images of answer sheets and computer vision to reconstruct the answer grid and produce reviewable grading results.

### Intended benefits

- Reduce repetitive manual grading work.
- Provide consistent machine-assisted checking.
- Support photographed and scanned answer sheets.
- Explicitly surface blank, ambiguous, multi-mark, and not-detected cases.
- Export structured results for further processing.
- Reduce dependence on dedicated OMR hardware in suitable scenarios.

---

## 3. What Is Actually Implemented?

The current implementation is based on deterministic computer-vision algorithms and numerical scoring. The main implemented components include:

- OpenCV image processing.
- Page boundary detection and four-point perspective correction when a suitable page contour is found.
- Hough-line fallback for small rotation correction when the page contour is broken.
- CLAHE contrast enhancement.
- Adaptive thresholding.
- Hough Circle candidate detection.
- Contour-based candidate detection.
- Candidate deduplication and geometric filtering.
- Physical grid/lattice reconstruction.
- Multi-feature bubble scoring.
- Conservative answer-status decisions.
- Answer-key comparison.
- JSON and CSV report generation plus an annotated image.

The implementation also contains quality checks for resolution, contrast, sharpness, and brightness and can emit warnings when image quality may reduce reliability.

---

## 4. Core Innovation

The main technical idea is a **hybrid candidate-detection + geometric-grid approach**.

A conventional circle detector may miss a weak or partially visible bubble. If every question number were derived only from the bubbles that happened to be detected, one missing detection could shift the interpretation of later questions.

SmartSheet instead uses detected bubbles as geometric evidence, estimates the physical row/column lattice, and evaluates expected positions on that lattice. When the evidence for an entire row is insufficient, the system reports the row as `not_detected` rather than shifting subsequent question numbers.

### Why this matters

This design makes failures more visible and safer for assessment workflows: a questionable row becomes a review item instead of becoming a silent numbering error.

---

## 5. System Architecture

```text
Input Image
    │
    ▼
Image Loading & Quality Check
    ├── Resolution
    ├── Contrast
    ├── Sharpness
    └── Brightness
    │
    ▼
Page Rectification
    ├── Page contour detection
    ├── Four-point perspective transform
    └── Rotation fallback
    │
    ▼
Preprocessing
    ├── Grayscale
    ├── CLAHE
    ├── Gaussian smoothing
    └── Adaptive thresholding
    │
    ▼
Candidate Detection
    ├── Hough Circles
    ├── Contours
    ├── Radius/size estimation
    └── Deduplication / filtering
    │
    ▼
Grid Reconstruction
    ├── Physical column grouping
    ├── Row clustering
    ├── Pitch estimation
    ├── Row-lattice fitting
    └── Question-number mapping
    │
    ▼
Bubble Scoring
    ├── Dark ratio
    ├── Very-dark ratio
    ├── Contrast
    ├── Fill ratio
    └── Center darkness
    │
    ▼
Decision Layer
    ├── Blank
    ├── Answered
    ├── Ambiguous
    ├── Multi-mark
    └── Not detected
    │
    ▼
Grading & Export
    ├── Answer-key comparison
    ├── Report
    ├── JSON
    ├── CSV
    └── Annotated image
```

---

## 6. Technical Implementation

### 6.1 Image Quality Assessment

Before grading, the implementation estimates image quality using resolution, grayscale statistics, contrast, Laplacian-based sharpness, and brightness. These measurements are used as warnings rather than automatic rejection rules.

### 6.2 Page Rectification

The system first searches for a large quadrilateral contour that can represent the answer-sheet page. When found, the page is ordered and transformed into a normalized rectangular view with a four-point perspective transform.

If the outer page contour is broken, a Hough-line fallback can estimate a small rotation and deskew the image. If neither method is sufficiently reliable, the original image is retained and a warning is produced.

### 6.3 Preprocessing

The current implementation applies:

1. Grayscale conversion.
2. CLAHE contrast enhancement.
3. Gaussian smoothing.
4. Adaptive Gaussian thresholding.
5. Background-intensity estimation from the processed image.

### 6.4 Candidate Detection

Two complementary candidate sources are used:

1. **Hough Circle Transform** for circular answer-bubble candidates.
2. **Contour analysis** for shapes that may not be recovered reliably by Hough detection.

Candidates are filtered and spatially deduplicated. The implementation estimates the real bubble radius before the main detection pass, helping reduce false positives from unrelated page elements.

### 6.5 Grid Reconstruction

The grid reconstruction process uses the spatial geometry of detected candidates rather than treating every detection as an independent question marker.

Main stages:

1. Group candidates into physical answer columns.
2. Estimate horizontal and vertical spacing.
3. Cluster candidate rows.
4. Fit a row lattice using regular spacing and available evidence.
5. Create expected bubble positions for each question.
6. Score the expected positions, including positions where an individual Hough circle was missed.
7. Mark a row `not_detected` when there is insufficient structural evidence.

The implementation supports configurable numbers of answer choices and physical columns within the supported ranges of the code.

### 6.6 Bubble Scoring

Each expected option is evaluated using multiple visual features:

| Feature | Role |
|---|---|
| Dark Ratio | Measures the proportion of dark pixels inside the bubble region. |
| Very-Dark Ratio | Captures stronger ink/pencil evidence. |
| Contrast | Compares local bubble darkness with surrounding/background intensity. |
| Fill Ratio | Estimates how much of the bubble region is filled. |
| Center Darkness | Adds evidence from the center of the bubble. |

The current score combines these features as:

```python
score = (
    0.34 * dark +
    0.22 * very_dark +
    0.22 * contrast +
    0.14 * fill_ratio +
    0.08 * center_dark
)
```

### 6.7 Decision Thresholds

Current implementation parameters include:

```python
MARK_FLOOR = 0.145
STRONG_MARK_FLOOR = 0.205
MULTI_MARGIN = 0.045
MAX_MARKED = 2
```

These are **implementation parameters, not universal OMR standards**. They can require calibration for different sheet designs, cameras, lighting conditions, paper quality, and marking instruments.

### 6.8 Decision States

| Status | Meaning |
|---|---|
| `answered` | A single answer has sufficient confidence for acceptance. |
| `blank` | No option reaches the mark floor. |
| `ambiguous` | The evidence does not sufficiently separate one option from another. |
| `multi_mark` | Two or more options are detected as strongly marked / conflicting. |
| `not_detected` | The question row cannot be reconstructed with sufficient structural evidence. |

After detection, the selected answer is compared with the answer key to produce grading outcomes such as correct or wrong.

### 6.9 Confidence

The current implementation calculates a decision-support confidence score from mark strength and separation from the second-best option:

```python
confidence = 100 * (0.60 * absolute + 0.40 * separation)
```

Where:

- `absolute` measures normalized strength of the best option above the mark floor.
- `separation` measures how clearly the best option exceeds the runner-up.

**Important:** this value is not a calibrated statistical probability. A future labeled validation dataset would be required to calibrate confidence probabilistically.

---

## 7. Evaluation & Evidence

The project currently contains **two different levels of test evidence**. They should not be mixed in the competition submission.

### 7.1 Previously recorded controlled benchmark

The earlier project documentation records 10 controlled exam sheets with a reported average of **95.2%**. This is a historical project result and should be labeled exactly as a **controlled benchmark**, not as a universal real-world accuracy claim.

| Test | Recorded result |
|---|---:|
| Exam 1 | 96% |
| Exam 2 | 100% |
| Exam 3 | 96% |
| Exam 4 | 80% |
| Exam 5 | 100% |
| Exam 6 | 98% |
| Exam 7 | 94% |
| Exam 8 | 100% |
| Exam 9 | 92% |
| Exam 10 | 96% |
| **Recorded mean** | **95.2%** |

> The original record does not provide enough information here to establish a statistically independent, real-world validation study. In particular, the meaning of “accuracy” and the ground-truth protocol should be documented before using 95.2% as a headline scientific performance claim.

### 7.2 New stress-test evidence

A later six-scenario stress test is included in the evidence folder. It contains **205 questions across six generated/controlled scenarios** and records **147 correct**, corresponding to a weighted overall result of approximately **71.7%**.

The most important finding is not the single number but the failure pattern:

| Scenario | Questions | Correct | Recorded result |
|---|---:|---:|---:|
| Clean, 40 questions, 2 columns | 40 | 40 | 100.0% |
| 1 column, 40 questions, rotation/noise | 40 | 5 | 12.5% |
| Blur + noise | 30 | 30 | 100.0% |
| Shadow + decoys | 30 | 30 | 100.0% |
| Perspective + rotation | 30 | 7 | 23.3% |
| Light marks | 35 | 35 | 100.0% |
| **Weighted total** | **205** | **147** | **71.7%** |

These stress tests expose a current robustness gap for certain one-column and severe-perspective conditions. This is valuable engineering evidence and should be presented honestly as a limitation and development target.

### 7.3 What should be claimed?

Recommended competition wording:

> **“In controlled tests, SmartSheet demonstrated strong performance on several clean, noisy, shadowed, and light-mark scenarios, while stress testing also identified weaknesses under severe perspective distortion and some one-column layouts. These results guide the next robustness improvements.”**

Avoid claims such as “100% accurate,” “works on every smartphone,” or “production-ready for all exam sheets” unless supported by a broader independently labeled evaluation.

---

## 8. Outputs

The implementation can generate:

- Human-readable grading report.
- Annotated analysis image.
- JSON report.
- CSV results.

The JSON/CSV outputs are designed to make individual question decisions inspectable and reusable in later systems.

Example CSV schema:

```csv
question,student,correct,status,confidence,detected
1,A,A,Correct,96.5,True
2,B,B,Correct,92.3,True
3,C,C,Correct,89.7,True
```

---

## 9. Educational Impact — SDG 4

SmartSheet is aligned with **SDG 4: Quality Education** because it targets an administrative bottleneck in assessment rather than attempting to replace teachers.

### Potential impact

- Less repetitive grading work.
- Faster access to assessment results.
- More consistent first-pass checking.
- Explicit review queues for uncertain answers.
- Lower hardware requirements than dedicated OMR scanning in suitable settings.

The project should be positioned as **teacher-support technology**. Human review remains important for ambiguous, multi-mark, or structurally uncertain cases.

---

## 10. Responsible Technology Positioning

SmartSheet should be presented with four principles:

1. **Transparency:** explain how decisions are produced.
2. **Uncertainty visibility:** do not silently convert uncertain evidence into confident answers.
3. **Human oversight:** allow educators to review problematic cases.
4. **Evidence-based claims:** distinguish controlled tests from real-world validation.

These principles strengthen the project for an educational-technology competition because they connect technical design with responsible deployment.

---

## 11. Limitations

Current limitations demonstrated by testing include:

- Severe perspective distortion can significantly reduce grid reconstruction reliability.
- Some one-column layouts can cause substantial question-row loss or misclassification.
- Very faint or unusual marks may require calibration.
- Different sheet geometries can require different parameters.
- The current core is deterministic computer vision rather than a trained deep-learning model.
- The current evidence is not yet a large, independently labeled real-world dataset.
- Confidence values are not statistically calibrated probabilities.

---

## 12. Development Roadmap

### Phase 1 — Robustness

- Improve one-column layout handling.
- Strengthen perspective correction for difficult photographs.
- Add more geometry validation before accepting a grid.
- Build targeted regression tests from every discovered failure case.

### Phase 2 — Data-driven calibration

- Collect a diverse real-photo dataset.
- Create question-level ground truth.
- Tune thresholds using training/validation splits.
- Calibrate confidence scores.
- Report precision, recall, confusion matrices, and failure rates.

### Phase 3 — AI-assisted vision

- Evaluate a CNN/deep-learning model for bubble detection or mark classification.
- Compare learned detection against the current deterministic baseline.
- Keep the conservative review layer for uncertain predictions.

### Phase 4 — Educational platform

- Teacher/class dashboards.
- Student analytics.
- LMS integration.
- Mobile capture workflow.
- Multilingual interface.

---

## 13. Installation & Usage

The exact dependency versions and repository URL should be taken from the **actual submission project files**. Do not leave placeholder repository addresses in a competition submission.

Typical local workflow:

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Gradio address printed by the application.

### Basic workflow

1. Upload an answer-sheet image.
2. Enter the answer key.
3. Configure answer-choice count and physical columns as supported by the interface.
4. Start analysis.
5. Review the report and annotated image.
6. Inspect or export JSON/CSV results.

---

## 14. Project Structure

The final submission should document the **actual** files included in the ZIP. A typical structure for the current application is:

```text
SmartSheet/
├── app.py
├── exam_grader.py
├── requirements.txt
├── README.md
├── README_fa.md
├── DOCUMENTATION.md
├── LICENSE                 # only if actually included
└── sample_images/          # if actually included
```

Do not claim that a file, folder, Docker configuration, GitHub repository, or sample dataset exists unless it is actually included in the submitted project.

---

## 15. Competition Positioning

### One-sentence pitch

> **SmartSheet uses computer vision to turn ordinary images of multiple-choice answer sheets into structured, reviewable grading results—reducing repetitive work while keeping uncertain cases visible to educators.**

### Short pitch

> **SmartSheet is a software-first OMR grading system for education. It combines image processing, geometric grid reconstruction, multi-feature mark scoring, and conservative decision logic to automate first-pass grading from photographed or scanned answer sheets. Its design explicitly surfaces uncertainty instead of silently guessing, supporting faster assessment while keeping educators in control.**

### Why the project is meaningful

The strongest story is not “perfect automated grading.” The stronger and more defensible story is:

**accessible assessment automation + transparent computer vision + human review + measurable engineering progress.**

---

## 16. Demo Video Structure — 2:30

| Section | Time |
|---|---:|
| Problem / Hook | 0:00–0:20 |
| Traditional workflow | 0:20–0:40 |
| SmartSheet introduction | 0:40–1:05 |
| Live application demo | 1:05–1:40 |
| Computer-vision pipeline | 1:40–2:05 |
| Results & exports | 2:05–2:20 |
| SDG 4 impact | 2:20–2:25 |
| Final outro | 2:25–2:30 |

The demo should prioritize **real application footage and real test evidence**. Cinematic animation is best used to explain the problem, transition between stages, and visualize the pipeline—not to imply capabilities that the software does not currently have.

---

## 17. Final Submission Checklist

### Technical

- [ ] Final code runs from a clean environment.
- [ ] Dependency file matches the actual code.
- [ ] No placeholder GitHub URLs remain.
- [ ] No obsolete product name such as `ExamVision` remains where the final product is called `SmartSheet`.
- [ ] Sample inputs are included if referenced.
- [ ] Output examples match actual generated fields.

### Evidence

- [ ] Historical 95.2% result is labeled as a controlled benchmark.
- [ ] New stress-test results are included and not hidden.
- [ ] Claims about accuracy, robustness, AI, and readiness match evidence.
- [ ] Real-world validation limitations are disclosed.

### Competition

- [ ] English documentation is complete.
- [ ] Persian documentation is retained for internal/reference use.
- [ ] Demo video is within the requested time limit.
- [ ] Project ZIP has been tested after packaging.
- [ ] Application form uses the same project name, version, and claims as the ZIP.

---

## 18. Final Position

SmartSheet demonstrates how computer vision can automate a repetitive educational workflow using ordinary images rather than specialized OMR hardware. Its most defensible technical contribution is the combination of candidate detection, geometric grid reconstruction, multi-feature bubble scoring, and conservative uncertainty handling.

The project is strongest when its limitations are presented as part of the engineering story: controlled tests show strong performance in several conditions, while stress testing has already revealed specific failure modes that define the next development targets.

**SmartSheet — Smarter Assessment. Transparent Automation. Better Support for Educators.**
