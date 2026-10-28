# Paired OCT Registration

**English** | [简体中文](README_CN.md)

Code for registering postoperative OCT B-scans to paired preoperative scans and
reconstructing registered images from saved transforms.

The pipeline uses retinal masks and boundaries as structural priors:
**A-OCT** estimates global lateral alignment and column-wise axial correction;
**D-OCT** estimates axial and bounded lateral residuals using gray-intensity
and axial-gradient dynamic time warping, followed by spatial smoothing.

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
| `deformation/` | Saved registration parameters (`.npz`, format 3) |
| `registration_manifest.csv`, `failures.json` | Output paths and processing status |

## Saved vector transforms

The selected D-OCT method retains 64 depth nodes, intensity/gradient weights
0.72/0.28 and Gaussian smoothing with standard deviations of 2 nodes and 6
columns. Lateral candidates are -4, -2, 0, 2 and 4 target-grid pixels; there is
no patch-based structural matching. Axial residuals are bounded by 12 pixels.
Both components vanish at the processed retinal anchors. A geometric projection
reduces the lateral component when necessary to keep bilinear-cell Jacobians
positive. Empty supported domains retain A-OCT.

Format 3 preserves `residual_nodes` (axial, 64 x width) and adds
`lateral_nodes` (64 x width) and scalar `lateral_scale`. The latter stores the
retained lateral amplitude after geometric projection. `source_x_line` and
`initial_*` fields always describe the frozen A-OCT map. The two residual
components are interpolated along depth, composed in normalized retinal-depth
coordinates, and then composed with A-OCT before one native-B resampling.
The reconstruction command also accepts legacy archives without lateral nodes.

An outer-boundary endpoint frozen after A-OCT is shared by D-OCT by definition;
it is not an independent test that the final field preserves every native model
boundary. Directly transformed native boundaries, frozen stage boundaries and
fresh predictions on registered images are distinct measurements.

## Code layout

- [`src/paired_oct_registration/`](src/paired_oct_registration/): registration, reconstruction and command-line interfaces.
- [`scripts/`](scripts/): dataset processing utilities.
- [`tests/`](tests/): registration and reconstruction checks.

## License

[MIT](LICENSE).
