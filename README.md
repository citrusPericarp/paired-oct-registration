# Paired OCT Registration

**English** | [简体中文](README_CN.md)

Code for registering postoperative OCT B-scans to paired preoperative scans and
reconstructing registered images from saved transforms.

The pipeline uses retinal masks and boundaries as structural priors:
**A-OCT** estimates global lateral alignment and column-wise axial correction;
**D-OCT** refines local alignment with texture-based dynamic time warping.

## Installation

Use Python 3.10 or 3.11 with the pinned numerical dependencies.

```bash
git clone https://github.com/citrusPericarp/paired-oct-registration.git
cd paired-oct-registration
python -m pip install -r requirements-reproduction.txt -e .
```

## Usage

Estimate registration for one scan:

```bash
paired-oct-register --dataset ../dataset --output ../output --sample-id 0001
```

Process all scans in the manifest:

```bash
paired-oct-register --dataset ../dataset --output ../output
```

Reconstruct registered images, masks and boundaries from saved transforms:

```bash
paired-oct-reconstruct --dataset ../dataset --output ../reconstructed
```

## Inputs and outputs

Place `manifests/pairs.csv` under the dataset root with these columns:

```text
sample_id,real_A,real_B,mask_A,mask_B,boundary_A,boundary_B
```

Paths are relative to the dataset root. `real_A` and `real_B` are the preoperative
and postoperative images; masks and boundaries are model-derived retinal priors.
Boundary `.npz` files contain `upper_y`, `lower_y` and `valid_columns`.
Reconstruction also reads the `deformation` path from the manifest.

| Output | Contents |
|---|---|
| `aligned_B/` | Registered postoperative images |
| `aligned_mask_B/`, `aligned_boundary_B/` | Transformed masks and their boundaries |
| `valid_mask/` | Pixels with valid source sampling coordinates |
| `evaluation_mask/` | Overlap of the preoperative and registered retinal masks |
| `deformation/` | Saved registration parameters (`.npz`, format 2) |
| `registration_manifest.csv`, `failures.json` | Output paths and processing status |

## Code layout

- [`src/paired_oct_registration/`](src/paired_oct_registration/): registration, reconstruction and command-line interfaces.
- [`scripts/`](scripts/): dataset processing utilities.
- [`tests/`](tests/): registration and reconstruction checks.

## License

[MIT](LICENSE).
