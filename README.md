# Plate / Portion

A local web application that describes the layout of food on a plate and explores food portion estimation. Upload a plate photograph to see the detected plate and food regions, hear a clock-position description, and compare visible area shares. A separate RGB-D pathway estimates volume from compatible depth data and can display nutrition predictions when the corresponding trained model or calibration is installed.

This is a research prototype. Food recognition and portion estimates can be wrong. Do not use it to make allergy, dietary, or medical decisions.

## Features

- **Plate understanding:** FoodSeg103-trained SegFormer segmentation, plate ellipse detection, rectification, broad food-region grouping, clock positions, and confidence-aware text.
- **Accessible output:** The same description is available as text and through browser speech synthesis with listen, pause, resume, stop, and speed controls. Keyboard navigation and status messages are included.
- **Relative portions:** Shows the visible share of the detected plate occupied by reported regions. This is an area measurement, not weight.
- **Real RGB-D study:** Accepts a matching 640 × 480 RGB image and raw 16-bit depth PNG from the compatible overhead setup. Depth geometry estimates volume; optional calibration or direct-model weights enable nutrition estimates.
- **Illustrative RGB demo:** With a successful plate segmentation, an opt-in switch constructs synthetic depth using fixed assumptions and shows illustrative volume, mass, and calories. These are demonstration numbers, not measurements inferred from the photo.

Menu recognition is outside the scope of this application. The website runs locally with Flask and plain HTML/CSS/JavaScript; no frontend build step is needed.

## Requirements

- Python 3.11 (the package declares support for Python 3.11 and 3.12).
- A compatible PyTorch and torchvision pair. Apple Silicon can use MPS where available; CPU inference also works.
- The project dependencies in `requirements.txt`.
- For plate analysis: a **trained** FoodSeg103 checkpoint in Hugging Face SegFormer format. Model architecture parameters alone are insufficient.
- For real volume: a matching RGB/depth pair. For real nutrition prediction: the saved direct model checkpoint and/or full calibration from the portion training pipeline.

Model weights and datasets are not bundled. Check dataset and model terms before redistributing them.

## Installation

Clone this repository and run from its root. On macOS with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q tests
```

For other operating systems, install a matching PyTorch/torchvision pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then continue with the requirements and editable install commands. If a pinned pair is unavailable for your Python or OS version, use the selector's matching pair. Use the activated environment for all subsequent commands.

## Configure checkpoints

Place the trained plate model at:

```text
checkpoints/
  foodseg-b0/
    config.json
    preprocessor_config.json
    model.safetensors       # or the corresponding PyTorch weight file
```

The model must be the compatible 104-label SegFormer checkpoint. The directory must contain the actual weights, not just a Kaggle output archive, evaluation figures, or `config.json`. The app loads the model locally.

Optional portion artifacts use these paths:

```text
checkpoints/
  nutrition-direct/
    best.pt                 # trained RGB-D regression model and target scaler
  nutrition-calibration.json # complete training-set geometric calibration
```

The `checkpoints/` directory is ignored by Git. Do not commit large weights or a dataset without checking their provenance and redistribution terms. `Model_Weights_and_Parameters.pdf` describes a model and some fitted constants; it does not contain the tensors in `best.pt` or the complete calibration JSON.

## Run the website

```bash
python web/server.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050). Check component availability at [http://127.0.0.1:5050/api/health](http://127.0.0.1:5050/api/health): `plate`, `direct`, and `calibration` indicate whether the expected local files exist. Availability does not establish that a checkpoint is compatible or accurate; analyze a known example to check inference. The server binds to localhost and processes uploaded images in memory.

1. Upload an image with the full plate rim visible, viewed roughly from above. The diner's side should be at the bottom of the image.
2. Choose a full or short description and select **Analyze plate**. Review the overlay, text, confidence warnings, and visible area shares.
3. Use **Listen** to hear the description. The text remains available for screen readers.
4. To demonstrate the portion UI with an ordinary RGB image, tick **Show illustrative RGB-only portion demo** before analysis. The page labels its results and lets you download a synthetic depth PNG.
5. To use a real depth capture, leave demo mode off and add a matching raw depth PNG. The two inputs must be aligned pixel for pixel, share orientation, and both be 640 × 480.

The RGB demo and real-depth upload are mutually exclusive. A coloured depth preview or a depth image from a different photograph cannot replace the matching raw frame.

## What the numbers mean

| Output | Input | Interpretation |
| --- | --- | --- |
| Clock positions and food groups | RGB + trained plate model | Predicted layout; verify the overlay and text. |
| Visible plate-area percentages | RGB + trained plate model | Image area, not grams or calories. |
| Illustrative volume, mass, calories | RGB + successful plate mask + opt-in demo | Synthetic geometry assumes a 40 cm camera distance and up to 2.5 cm food height; mass uses 0.757 g/cm³ and energy uses 1.07 kcal/g. Not a physical measurement. |
| Estimated volume | Aligned real RGB-D pair | Geometry with nominal 615 px focal lengths and reliability checks. |
| Estimated nutrition | Real RGB-D pair + calibration or trained direct checkpoint | Approximate prediction within the training setup; no general accuracy guarantee for arbitrary photos. |

The illustrative demo does not produce macronutrients because its available fitted constants do not include a complete composition. Do not treat a generated demo PNG as a real sensor frame or feed it into the real-depth input to validate performance.

For command-line creation of the same synthetic 16-bit PNG:

```bash
python scripts/make_demo_depth.py /path/to/plate.jpg results/synthetic_depth_DEMO_ONLY.png
```

This requires the plate checkpoint and a successful plate/food fit. The file name and command output identify the image as synthetic.

## Training and evaluation

The included `src/n5k/` code can download the Nutrition5k overhead RGB-D subset, train the direct regression model, and fit geometric calibration. This is optional for the RGB website and requires substantial disk space and time:

```bash
python -m n5k.data.download --what metadata
python -m n5k.data.download --what imagery --workers 4
python -m n5k.train --epochs 15 --batch-size 8 --out checkpoints/nutrition-direct
python -m n5k.eval.compare --checkpoint checkpoints/nutrition-direct/best.pt --out runs/compare
mkdir -p checkpoints
cp runs/compare/calibration.json checkpoints/nutrition-calibration.json
```

The downloader uses the official depth splits and is resumable. The comparison program fits calibration on training dishes and evaluates against held-out dishes. Train and test dishes must stay separate. The provided `vendor_notes/FOOD_PORTION_ORIGINAL_README.md` has further details about that component's dataset parsing, methodology, and evaluation; it is historical component documentation rather than the setup guide for this app.

Before reporting performance, run the tests and record results from your own checkpoints and test images. Do not present the illustrative numbers or prior summary files as evaluation of the integrated app.

## Project layout

| Path | Purpose |
| --- | --- |
| `web/` | Flask API and browser interface. |
| `src/plateclock/` | Food segmentation, plate detection, geometry, narration, and the integrated service. |
| `src/n5k/` | Nutrition5k RGB-D geometry, training, calibration, and evaluation code. |
| `scripts/` | Command-line utilities, including the synthetic-depth presentation script. |
| `configs/` | Plate and narration configuration. |
| `tests/` | Plate, integration, and synthetic-depth tests. |
| `vendor_notes/` | Historical component README, attribution notice, sample summaries, and supplemental scripts/tests. Not imported at runtime. |
| `checkpoints/`, `data/`, `results/` | Local artifacts excluded by `.gitignore`. |

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `/api/health` reports `plate: false` | Confirm `config.json`, `preprocessor_config.json`, and the complete weight file are directly under `checkpoints/foodseg-b0/`. |
| Model reports a class-count or checkpoint error | Confirm the checkpoint is the trained 104-label model expected by this code. Read the server traceback. |
| Plate failure or no confident food | Try a full, centered plate with visible rim; review the overlay and warnings. A usable image is not guaranteed. |
| Depth rejected | Confirm both images are aligned 640 × 480 and the depth file is single-channel 16-bit raw PNG, not a coloured display. |
| No mass, calories, or macros in real-depth mode | Check `/api/health` for `calibration` or `direct`; supply the actual saved artifact. A depth frame alone yields geometry, not nutrient composition. |
| Speech unavailable | Try a browser with speech synthesis or use a screen reader to access the displayed text. |

## Data, provenance, and licence

The portion component comes from the team's `foodportion4-boop` work and retains its MIT copyright and permission notice in `vendor_notes/FOOD_PORTION_LICENSE`. If this combined project is published in that same repository, preserve the existing root `LICENSE` and its actual copyright holders; consolidate notices only if the resulting file still retains all required notices. The extra component licence file can remain as a clear attribution record. Do not substitute a new holder for someone else's contribution.

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) data is not included and is released under CC BY 4.0. FoodSeg103 data and pretrained checkpoints are not included. This project does not promise accessibility or nutritional accuracy in safety-critical use; test it with intended users and representative photographs before claiming either.
