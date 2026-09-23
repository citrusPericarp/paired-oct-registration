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

The reference environment is Python 3.11.15, NumPy 2.0.1, SciPy 1.17.1,
OpenCV 4.11.0.86 and Pillow 11.1.0. Use a clean environment and the supplied
requirements file. OpenCV 5 changes interpolation results. The reference DTW
uses the same numerical path whether or not optional Numba is installed.
Regeneration (`paired-oct-register`) starts from images and published structural
priors; reconstruction (`paired-oct-reconstruct`) uses the saved transform.
RetinaUNet training and prior inference are a separate upstream workflow.

The corrected archive stores the actual target anchor, node-depth boundaries
and source x-coordinates. It does not recover these from rounded observation
scores. Earlier transform files without these fields must be regenerated before
using the reconstruction command.

The input dataset is located through `pairs.csv`; no local absolute path is
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
