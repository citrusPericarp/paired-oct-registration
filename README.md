# Paired OCT registration

This repository generates `aligned_B` from the `real_A` and `real_B` images in
the separately downloaded dataset.

Scope: this package contains registration and saved-transform reconstruction.
RetinaUNet training/inference, restoration-model experiments, vessel-shadow
analysis, and the clinical dataset are separate and are not included here.

## Install

```bash
python -m pip install -r requirements-reproduction.txt -e .
```

## Run one sample

```bash
paired-oct-register --dataset ../dataset --output ../output --sample-id 0001
```

## Run the complete dataset

```bash
paired-oct-register --dataset ../dataset --output ../output
```

## Reconstruct saved transforms

For the corrected release (deformation format 2), recreate the five image and
boundary products without re-estimating registration:

```bash
paired-oct-reconstruct --dataset ../dataset --output ../reconstructed
```

The current data-generation environment is Python 3.9.23, NumPy 1.26.4,
SciPy 1.10.1, OpenCV 4.11.0.86 and Pillow 11.1.0. The public package requires
Python >=3.10; the numerical pins also support Python 3.10/3.11. Use a clean environment and the supplied
requirements file. OpenCV 5 changes interpolation results. The reference DTW
uses the same numerical path whether or not optional Numba is installed.
Regeneration (`paired-oct-register`) starts from images and published structural
priors; reconstruction (`paired-oct-reconstruct`) uses the saved transform.
RetinaUNet training and prior inference are a separate upstream workflow.

The corrected archive stores the actual target anchor, node-depth boundaries
and source x-coordinates. It does not recover these from rounded observation
scores. Earlier transform files without these fields must be regenerated before
using the reconstruction command.

The input dataset is located through `manifests/pairs.csv`; no local absolute path is
hard-coded. Output uses the same simple names:

```text
output/
├── aligned_B/
├── aligned_mask_B/
├── aligned_boundary_B/
├── valid_mask/
├── evaluation_mask/
├── deformation/
├── registration_manifest.csv
└── failures.json
```

`mask_A`, `mask_B`, `boundary_A`, and `boundary_B` are model-derived structural
priors. They are not manual annotations or clinical ground truth. `aligned_B`
is a registered pseudo-reference. `aligned_mask_B` is `mask_B` transformed by
the same mapping as `aligned_B`. `aligned_boundary_B` stores the upper and
lower edges extracted from `aligned_mask_B`, together with the original valid
column flags. Use `valid_mask` to identify output pixels
whose source coordinates are inside the finite `real_B` image. The
`evaluation_mask` is the intersection of the in-domain `mask_A` and
`aligned_mask_B` and is intended for paired measurements rather than as a
clinical annotation.

Code repository: https://github.com/citrusPericarp/paired-oct-registration
(MIT license). Clinical images and metadata are maintained separately and are
not included in this repository.


## Registration update 2026-10-01 (software 0.3.0)

D-OCT no longer requires a whole-image common-column fraction. Any nonempty
trusted local domain can enter the local estimator; an empty domain returns zero
residual. Per-column thickness checks, texture DTW, boundary anchoring, compact
support and monotonicity constraints remain. The Python compatibility argument
`minimum_valid_fraction` is retained with default 0.0 and ignored. Earlier 0.70
and 0.75 whole-image thresholds no longer apply. Image-supported fovea selection,
bounded endpoint continuation, and source-shifted local geometric support checks are used.
The reference DTW numerical path is unchanged.

The new data run is `20261001-no-whole-image-gate`. Compact archives retain
`format_version=2` and add `registration_algorithm_version`; the sampling-map
schema and reconstruction semantics are unchanged. Algorithm versions and
software package versions are separate.

### Frozen segmentation of registered images

`aligned_mask_B` remains the transformed native `mask_B`; `evaluation_mask`
remains its intersection with in-domain `mask_A`. New `resegmented_mask_B` and
`resegmented_boundary_B` are frozen v10 predictions on **new aligned_B**. They
are separate products and never feed back into the registration that generated
that image. Their full model output is retained; apply `valid_mask` explicitly
when defining a measurement domain. These masks are model predictions, never
clinical ground truth.

Obtain the independent v10 bundle (not included in this registration package
or the dataset directory; its public download identifier is pending), including `source/`, `checkpoints/best_reviewed_terminal_trajectory.pt`
and `source/configs/decoder_v10_terminal_connected_candidate_20260910.json`.
Install its documented PyTorch/inference dependencies separately. Then run:

```bash
python scripts/infer_registered_B.py --dataset ../dataset --release-root /path/to/v10_bundle --output ../registered_B_inference --device cuda --batch-size 4
```

The wrapper writes an input manifest with role `registered_B`, prediction and
provenance records, and per-scan band/shape QC; it writes the two separate
products into the dataset's all/normal/abnormal views. No prior-mask comparison
or Dice evaluation is performed. Saved transformations reconstruct transformed
products; regeneration of resegmented products additionally requires the frozen
checkpoint and decoder.

The 2026-10-01 data run used Python 3.9.23, NumPy 1.26.4, SciPy 1.10.1,
OpenCV 4.11.0 and Pillow 11.1.0. Its exact runtime is recorded with the data
receipt. The historical Python 3.11.15/NumPy 2.0.1/SciPy 1.17.1 environment belongs to
the previous release and is not the current numerical pin set. Public package installation supports Python >=3.10; use the
current numerical pins in `requirements-reproduction.txt` for reproduction.
