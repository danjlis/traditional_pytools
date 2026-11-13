# sPHENIX calorimeter clustering and jet tools

Python ports of the sPHENIX calorimeter reconstruction, written to be called by an agent (or
by hand). Each tool reads calorimeter towers for one event and writes JSON describing the
clusters or jets it finds.

| tool | ported from (sPHENIX `coresoftware`) | input layers |
|------|--------------------------------------|--------------|
| `emcal_template_clustering.py` | `CaloReco/RawClusterBuilderTemplate` (BEmcRec / BEmcRecCEMC) | EMCal |
| `topo_clustering.py` | `CaloReco/RawClusterBuilderTopo` | EMCal, inner HCal, outer HCal |
| `jet_clustering.py` | `jetbase/TowerJetInput`, `jetbase/ClusterJetInput`, `jetbase/FastJetAlgo`, `jetbackground/RetowerCEMC` | towers (EMCal, inner HCal, outer HCal), or the cluster JSON of either clustering tool |

`calo_common.py` holds the shared tower readers and calorimeter geometry.

## Requirements

- Python 3.
- For CSV input, the two clustering tools use only the standard library.
- Reading `calo_features.mmap` needs `mmap_ninja` (which brings in numpy), and the jet tool
  needs the scikit-hep `fastjet` package (prebuilt for macOS and Linux). Both are in
  `requirements-jets.txt`:

  ```
  python3 -m venv .venv
  .venv/bin/pip install -r requirements-jets.txt
  ```

  Then use `.venv/bin/python` (or `source .venv/bin/activate`) to run the tools.

## Quick start

No data needed: this builds a toy event with known photons and a hadron, runs all three
tools on it and prints what they found next to the truth.

```
python3 examples/run_example.py
```

The jet step is skipped if no Python with fastjet is found (it also looks in `.venv/`).
[`examples/README.md`](examples/README.md) walks through the example, all the options, the
input and output formats, and how to call the tools from Python.

## Running the tools

On one event of a `calo_features.mmap` directory:

```
.venv/bin/python emcal_template_clustering.py path/to/calo_features.mmap --event 5 --min-cluster-e 0.5
.venv/bin/python topo_clustering.py           path/to/calo_features.mmap --event 5
.venv/bin/python jet_clustering.py            path/to/calo_features.mmap --event 5 --jet-min-pt 3
```

Jets from clusters instead of towers: pass the JSON written by a clustering tool.

```
.venv/bin/python topo_clustering.py path/to/calo_features.mmap --event 5 -o topo.json
.venv/bin/python jet_clustering.py  topo.json --jet-min-pt 3
```

On a CSV file of towers:

```
python3 emcal_template_clustering.py towers.csv -o clusters.json
python3 topo_clustering.py           towers.csv -o topo.json
.venv/bin/python jet_clustering.py   towers.csv -o jets.json
```

Without `-o` the JSON goes to stdout. `--help` on any tool lists its options; the defaults
follow the sPHENIX code (for jets: anti-kt, R = 0.4, retowered EMCal + inner + outer HCal).

From Python:

```python
from calo_common import load_towers
from topo_clustering import run_topo_clustering

towers = load_towers("path/to/calo_features.mmap", event=5)   # or a CSV path, no event
result = run_topo_clustering(towers)
```

## Input formats

**`calo_features.mmap`**: an `mmap_ninja` RaggedMmap directory with one N x 10 array per
event. Columns: E, x, y, z, time, subsystem (4 = EMCal, 5 = inner HCal, 6 = outer HCal; other
subsystems are skipped), ieta, iphi, and two truth columns that are not used. The tower
positions in the file set the geometry: eta bin centers, phi offset and radius of each layer
are measured from it.

**CSV**: columns `eta`, `phi`, `E`, plus `layer` (`cemc`, `hcalin`, `hcalout`; required for
topo and jets) and optionally `ieta`, `iphi`, `time`. Without `ieta`/`iphi`, towers are
binned on a uniform eta/phi grid, which approximates the real geometry.

## Limitations

- Validated against the C++ on 100 run 29 Jet12 events (see ../calotree): Template clusters,
  topoclusters, retowered EMCal and jets agree object by object to float precision, given the
  same towers (all towers, bad ones removed) and the measured geometry. `sphenix_tower_bounds.json`
  holds the CDB tower bounds used for retowering and the Topo bin centers.
- CSV geometry is approximate (uniform bins); mmap geometry is measured from the file.
- Template: no shower-profile probability or chi2 (reported as null), no incidence-angle
  correction, no bad-tower handling.
- Topo: towers absent from the input are treated as non-existent, which matters for
  zero-suppressed data such as `calo_features.mmap` (50 MeV threshold). With mmap input the
  EMCal-inner HCal phi links follow the file's geometry instead of the C++ hard-coded offset.
- Jets: no underlying-event subtraction, no grooming, no bad-tower masking. The event vertex
  is not in the mmap files; pass it with `--vz` (default 0).
