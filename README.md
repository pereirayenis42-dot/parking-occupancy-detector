# Parking Occupancy Detector

**Computer Vision Project | Python | YOLO11 | OpenCV**

## Project Overview

The Parking Occupancy Detector is a computer vision project designed to identify whether individual parking spaces are occupied or vacant using photographs taken from an elevated camera viewpoint.

The project was inspired by a real-world challenge in automotive dealership operations: vehicles are frequently moved between parking locations, making it difficult to maintain an accurate picture of parking availability and vehicle placement.

The system uses YOLO11 for vehicle detection, manually defined parking-space polygons, and image registration to analyze photographs captured at different times of day.

The current prototype monitors four designated parking spaces (A01–A04) and explores how camera alignment and detection accuracy affect occupancy predictions.

## Problem Statement

Parking occupancy detection becomes challenging when photographs are captured from slightly different camera positions or under changing lighting conditions.

Even small changes in camera perspective can cause vehicle detections and parking-space boundaries to become misaligned, resulting in incorrect occupancy classifications.

This project investigates whether image registration and quality-aware prediction methods can improve the reliability of parking occupancy detection.

## Project Objectives

- Detect vehicles in parking-lot photographs using YOLO11.
- Define individual parking spaces using polygon coordinates.
- Classify parking spaces as occupied or vacant.
- Align photographs captured from different viewpoints.
- Evaluate predictions against manually labeled ground-truth data.
- Measure registration errors and investigate their impact on occupancy accuracy.
- Identify uncertain predictions rather than automatically returning potentially incorrect results.

## Current Features

- YOLO11-based vehicle detection
- Polygon-based parking-space mapping
- Automatic occupied/vacant classification
- Feature-based image registration
- Detection-first occupancy experiments
- Registration-quality diagnostics
- Per-space landmark error analysis
- Experimental quality-gated predictions with an `UNKNOWN` state
- CSV and JSON evaluation outputs

## Project Status

**Research prototype — active development**

The core detection and occupancy pipeline has been implemented. Current experiments focus on registration accuracy, viewpoint differences, and determining when the system should abstain from making an occupancy prediction.

## System Architecture

The Parking Occupancy Detector combines object detection, image registration, geometric analysis, and prediction-quality assessment.

### 1. Vehicle Detection

The system uses **YOLO11**, a pretrained object detection model, to identify vehicles in parking-lot photographs.

Detected vehicles are represented by bounding boxes and confidence scores. These detections provide the geometric information used for occupancy classification.

**Main implementation:** `src/detector.py`

### 2. Parking-Space Mapping

Four parking spaces, labeled A01 through A04, are manually defined using polygon coordinates.

These polygons represent the parking-space boundaries in a reference photograph.

Parking-space definitions are stored in:

`data/parking_spaces.json`

### 3. Image Registration

Because photographs are captured at different times and from slightly different viewpoints, the system attempts to align each test photograph with the reference image.

The registration process uses OpenCV feature matching and geometric transformations to estimate the relationship between the images.

This allows parking-space boundaries or vehicle detections to be compared within a common coordinate system.

**Main implementation:** `src/alignment.py`

### 4. Occupancy Classification

The system evaluates the geometric relationship between detected vehicles and predefined parking spaces.

Two approaches have been investigated:

- **Baseline approach:** Align the photograph before evaluating vehicle occupancy.
- **Detect-first approach:** Detect vehicles in the original photograph, transform the detected vehicle regions into reference-image coordinates, and evaluate their relationship with parking-space polygons.

The objective is to determine whether each parking space is occupied or vacant.

**Relevant implementations:**

- `src/occupancy.py`
- `src/occupancy_polygons.py`
- `src/validate.py`
- `src/validate_detect_first.py`

### 5. Registration-Quality Assessment

Image registration can introduce errors when photographs differ in viewpoint, lighting, shadows, or perspective.

The project includes diagnostic experiments that measure registration quality using manually selected landmarks and per-space error measurements.

These diagnostics help determine whether a transformation is sufficiently reliable for occupancy classification.

**Relevant implementations:**

- `src/registration_quality.py`
- `src/per_space_quality.py`
- `src/multi_landmark_check.py`
- `src/assess_registration_quality.py`

### 6. Quality-Gated Predictions

An experimental quality gate evaluates registration reliability before accepting an occupancy prediction.

The intended output states are:

| Prediction | Meaning |
|---|---|
| `OCCUPIED` | A vehicle is predicted to occupy the parking space. |
| `VACANT` | No qualifying vehicle occupancy is predicted. |
| `UNKNOWN` | The prediction is withheld because the quality requirements are not satisfied. |

The `UNKNOWN` state is designed to prevent the system from presenting uncertain predictions as reliable results.

In the current experimental scenario, all four monitored parking spaces were classified as `UNKNOWN` after the registration-quality gate was applied. This remains an active area of investigation.

**Relevant implementations:**

- `src/quality_gated_occupancy.py`
- `src/apply_quality_to_predictions.py`
- `src/evaluate_quality_gate.py`

### 7. Evaluation and Experiment Outputs

The project stores evaluation data in CSV and JSON formats to support reproducible comparisons.

Results are organized into separate directories:

- `results/metrics/baseline/` — Baseline occupancy evaluation
- `results/metrics/detect_first/` — Detect-first, landmark, and quality-gating experiments
- `results/metrics/` — Additional diagnostics and evaluation outputs

Generated photographs and visual debugging outputs are kept locally and excluded from the public repository pending privacy review.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLO11 | Vehicle detection |
| OpenCV | Image processing, feature matching, and image registration |
| NumPy | Numerical calculations and geometric operations |
| Pandas | Evaluation data and CSV processing |
| Git and GitHub | Version control and project documentation |
| Visual Studio Code | Development environment |

## Project Structure

```text
parking-occupancy-detector/
│
├── data/
│   ├── ground_truth.csv
│   └── parking_spaces.json
│
├── src/
│   ├── main.py
│   ├── detector.py
│   ├── alignment.py
│   ├── occupancy.py
│   ├── occupancy_polygons.py
│   ├── validate.py
│   ├── validate_detect_first.py
│   ├── evaluate.py
│   ├── registration_quality.py
│   ├── multi_landmark_check.py
│   ├── quality_gated_occupancy.py
│   └── ... additional experiment scripts
│
├── results/
│   └── metrics/
│       ├── baseline/
│       └── detect_first/
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Original parking photographs, generated visualization images, model weights, and the Python virtual environment are excluded from the public repository.

## Installation

### Prerequisites

- Python 3
- Git
- A Python virtual environment
- Access to the project repository
- Parking-lot photographs for running detection and validation experiments

### Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/parking-occupancy-detector.git
cd parking-occupancy-detector
```

Replace `YOUR-USERNAME` with the repository owner's GitHub username.

### Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Download the YOLO11 Model

The detector uses the pretrained YOLO11 Nano model (`yolo11n.pt`).

Model weights are excluded from GitHub to keep the repository lightweight.

After installing the project dependencies, run the following command from the project's root directory:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

This downloads the official pretrained model weights if they are not already available locally.

The file `yolo11n.pt` must be present in the project root directory because `src/main.py` loads the model from that location.

The current implementation runs inference on the CPU, so a dedicated GPU is not required.

## Dataset and Reproducibility

The original parking-lot photographs are not distributed publicly.

To reproduce the experiments, users must supply their own compatible images and configure the parking-space polygons for their camera viewpoint.

The included ground-truth and metric files document the project's original experiments, but those results cannot be fully reproduced without the corresponding private photographs.

## Usage

### Run the Baseline Parking Occupancy Detector

The baseline program analyzes the reference parking photograph, detects vehicles using YOLO11, and classifies the configured parking spaces.

**1. Prepare the input image**

Place the reference parking photograph inside:

`data/raw/`

The filename must match the `reference_image` value in `data/parking_spaces.json`.

The configuration also contains the crop ratios, expected crop dimensions, and polygon coordinates for the four monitored spaces.

**2. Activate the virtual environment**

On macOS or Linux:

```bash
source .venv/bin/activate
```

**3. Run the detector**

From the project's root directory:

```bash
python src/main.py
```

**4. Review the output**

The program prints the reference image filename, crop dimensions, number of detected vehicles, and occupancy prediction for each configured parking space.

Example output format:

```text
PARKING OCCUPANCY DETECTOR
=============================================
Image: <reference_image>
Crop size: (<width>, <height>)
Vehicles detected: <count>

PARKING OCCUPANCY RESULTS
-----------------------------------
A01: <predicted_status>
A02: <predicted_status>
A03: <predicted_status>
A04: <predicted_status>
```

The program also saves two images:

| Output file | Description |
|---|---|
| `results/images/debug_matching.jpg` | YOLO11 detections, parking-space boundaries, and vehicle matching points |
| `results/images/occupancy_result.jpg` | Parking-space occupancy visualization |

### How the Baseline Works

1. Loads `data/parking_spaces.json`.
2. Opens the configured reference photograph and corrects its EXIF orientation.
3. Crops the photograph using the saved crop ratios.
4. Checks that the cropped image dimensions match the parking-space configuration.
5. Runs YOLO11 vehicle detection with a confidence threshold of 0.25.
6. Uses the detected vehicle positions to classify parking-space occupancy.
7. Saves debugging and occupancy visualization images.

### Running Additional Experiments

The project includes separate scripts for image registration, multi-image validation, detect-first occupancy classification, and registration-quality assessment.

These experiments are maintained independently of the baseline entry point so their results can be compared without modifying the original detector.

Some experimental scripts require locally stored test photographs and additional configuration. Their command-line interfaces should be verified before running them.

### Important Limitations

- The baseline entry point processes one configured reference photograph per execution.
- Parking-space polygons are specific to the reference camera viewpoint.
- Different camera positions may require image registration or remapping.
- The original photographs are not included in the public repository.
- Experimental quality-gated predictions are not part of the baseline `main.py` execution.

## Results and Evaluation

### Baseline Occupancy Classification

The baseline occupancy detector was evaluated against manually labeled ground-truth data.

Evaluation results were saved in `results/metrics/baseline/evaluation_metrics.json`.

| Metric | Result |
|---|---:|
| Accuracy | 87.50% |
| Precision | 100.00% |
| Recall | 80.00% |
| F1 Score | 88.89% |
| Evaluated parking-space observations | 16 |
| Skipped observations | 4 |

### Confusion Matrix

| Actual / Predicted | Occupied | Vacant |
|---|---:|---:|
| Occupied | 8 | 2 |
| Vacant | 0 | 6 |

The baseline correctly classified 14 of the 16 evaluated observations.

Of the 10 actually occupied observations, eight were correctly identified and two were incorrectly classified as vacant.

All six evaluated vacant observations were correctly classified.

### Interpretation

The baseline achieved strong precision in this limited evaluation, with no false-positive occupancy predictions.

However, its recall of 80% indicates that some occupied spaces were missed.

These false negatives are particularly relevant to the project's operational motivation because an occupied parking space incorrectly reported as vacant could produce misleading parking availability information.

The evaluation also excluded four observations, which should be considered when interpreting the reported performance.

### Evaluation Limitations

These results represent a small experimental dataset involving four monitored parking spaces.

They do not establish performance across different parking lots, camera placements, weather conditions, or substantially different viewpoints.

Further testing is required to evaluate how reliably the detector generalizes beyond the current photographs and parking-space configuration.

### Experimental Quality-Gated Predictions

An additional experiment investigated whether registration-quality checks could prevent unreliable occupancy predictions from being accepted.

The experiment used the photograph `parking_2026-10-07_1807.jpg` and evaluated four monitored parking spaces.

Results were saved in:

`results/metrics/detect_first/parking_2026-10-07_1807_gate_metrics.json`

| Metric | Result |
|---|---:|
| Total predictions | 4 |
| Accepted predictions | 0 |
| Unknown predictions | 4 |
| Coverage | 0.00% |
| Abstention rate | 100.00% |
| Selective accuracy | Undefined |

All four predictions were withheld by the experimental quality gate.

The system returned `UNKNOWN` for A01, A02, A03, and A04 rather than accepting the underlying occupancy predictions.

**Interpretation**

The experiment demonstrates that the quality-gating mechanism can abstain from returning occupancy predictions when its acceptance criteria are not satisfied.

However, zero coverage is not a successful operational outcome. A system that withholds every prediction cannot provide useful occupancy information.

This result does not establish that the underlying predictions were incorrect or that the quality gate improved classification accuracy.

Selective accuracy was undefined because no predictions were accepted.

**Next Research Question**

Can the system distinguish genuinely unreliable registration from acceptable alignment more effectively, allowing trustworthy predictions to be retained while uncertain predictions are withheld?

Further experiments will investigate per-space alignment quality, viewpoint-dependent errors, and the relationship between registration error and occupancy classification.

### Image Registration Quality Analysis

To investigate registration accuracy, three manually selected landmarks were compared for each monitored parking space.

The experiment used `parking_2026-10-07_1807.jpg` and measured the pixel displacement between reference landmarks and their corresponding positions in the aligned photograph.

Results are stored in:

`results/metrics/detect_first/parking_2026-10-07_1807_multi_landmark.json`

| Parking Space | Median Error (px) | Maximum Error (px) | MAD (px) |
|---|---:|---:|---:|
| A01 | 31.40 | 32.31 | 0.91 |
| A02 | 20.25 | 21.10 | 0.85 |
| A03 | 18.44 | 20.52 | 1.47 |
| A04 | 22.80 | 26.91 | 4.10 |

Each parking space was evaluated using three manually selected landmarks.

MAD represents the median absolute deviation of the landmark errors within each space.

**Findings**

- A01 exhibited the largest median registration error.
- A03 exhibited the smallest median registration error.
- A04 showed the greatest variability in landmark displacement.
- Registration error differed across parking spaces, suggesting that alignment quality was not uniform across the image.

These results highlight a limitation of relying exclusively on global image-registration measurements when occupancy decisions depend on relatively small, localized regions.

### Global Translation Correction Experiment

A separate diagnostic experiment investigated whether an additional global translation could reduce landmark alignment error.

The correction improved overall landmark alignment in the initial experiment. However, held-out landmark testing revealed that improvements were not uniform across the four parking spaces.

Three spaces improved, while A02 exhibited increased error after correction.

This suggests that a single global translation may not adequately correct localized registration differences.

The translation correction was therefore treated as an experimental diagnostic rather than automatically incorporated into the production occupancy-classification pipeline.

### Implications for Occupancy Classification

Parking occupancy predictions depend on the spatial relationship between detected vehicles and predefined parking-space boundaries.

When image registration is inaccurate, vehicle detections may be displaced relative to their intended parking spaces.

This can produce false vacant or occupied predictions, particularly when vehicles are close to polygon boundaries.

The registration-quality experiments motivate further investigation into local alignment, camera perspective, and quality-aware occupancy classification.

**Reproducibility note:** The translation-correction experiment was conducted during development, but its complete before-and-after measurements were not exported to a dedicated results file. The observations are therefore considered preliminary. A future experiment will save the translation parameters, landmark errors, and held-out validation results in a structured format.

## Known Limitations

Although the prototype demonstrates the feasibility of detecting parking occupancy using computer vision, several limitations remain.

### Camera Viewpoint Sensitivity

The system relies on parking-space polygons defined in a reference photograph. Changes in camera position or perspective can cause alignment errors that affect occupancy classification.

### Registration Accuracy

Global image registration does not always align every parking space equally well. Local displacement can remain even when the overall transformation appears reasonable.

### Small Evaluation Dataset

The current evaluation covers four parking spaces and a limited collection of photographs. Additional data is necessary to assess performance across different environments and conditions.

### False-Negative Predictions

The baseline evaluation produced two false negatives, meaning occupied spaces were incorrectly classified as vacant.

Reducing these errors is important for improving the system's operational usefulness.

### Quality-Gate Coverage

The experimental quality gate withheld all four predictions in one evaluated photograph, producing 0% coverage.

The system requires further calibration to distinguish acceptable registration from genuinely unreliable alignment.

### Dataset Availability

Original parking photographs are excluded from the public repository. This protects potentially identifiable information but limits independent reproduction of the original experiments.

## Future Improvements

The next development phase will focus on:

1. **Local image registration:** Investigate region-specific alignment methods to reduce errors affecting individual parking spaces.
2. **Registration-quality calibration:** Evaluate quality thresholds using a larger set of manually verified landmarks.
3. **Improved occupancy matching:** Compare geometric overlap, vehicle reference points, and transformed detection polygons.
4. **Expanded evaluation:** Collect additional photographs across different times, lighting conditions, and camera viewpoints.
5. **Quality-gate refinement:** Improve prediction coverage while measuring accuracy on accepted predictions.
6. **Reproducible experiments:** Save experimental parameters, landmark measurements, and evaluation results in structured files.
7. **Automated reporting:** Generate consolidated evaluation summaries and visualizations for comparison between approaches.

## Conclusion

The Parking Occupancy Detector demonstrates how object detection, geometric image analysis, and evaluation methods can be combined to investigate a practical parking-management problem.

The baseline system achieved 87.50% accuracy across 16 evaluated parking-space observations, providing an initial demonstration of automated occupancy classification.

Further experiments revealed that image registration introduces significant challenges when photographs differ in viewpoint. Landmark measurements showed that alignment error varies across parking spaces, and the experimental quality gate withheld all predictions in one test photograph.

These findings emphasize that reliable computer vision systems require more than accurate object detection. Spatial alignment, uncertainty assessment, and careful evaluation are also important.

The project remains an active research prototype, with future development focused on improving alignment reliability and producing useful, trustworthy occupancy predictions.

## Project Context

Developed as part of the ITAI 1378 Computer Vision coursework.

The project is motivated by real-world automotive dealership parking and vehicle-location challenges.

**Development environment:** Python, Visual Studio Code, and macOS.

**Project status:** Active development.