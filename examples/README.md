# Calorimeter clustering tools: worked example

With CSV input, the clustering tools need only Python 3 (standard library). Reading
`calo_features.mmap` needs `mmap_ninja`, and the jet tool needs the scikit-hep `fastjet`
package; see "Setting up" below.

## Running on `calo_features.mmap`

Each tool takes the mmap directory and an event number, and writes the same JSON as for CSV
input (with an added `"event"` field):

```
.venv/bin/python emcal_template_clustering.py ../mmaps/calo_features.mmap --event 5 --min-cluster-e 0.5
.venv/bin/python topo_clustering.py           ../mmaps/calo_features.mmap --event 5
.venv/bin/python jet_clustering.py            ../mmaps/calo_features.mmap --event 5 --jet-min-pt 3
```

Columns of each event's N x 10 array:

| column | content |
|--------|---------|
| 0      | E [GeV] (towers are zero-suppressed at 50 MeV) |
| 1-3    | x, y, z [cm] |
| 4      | time |
| 5      | subsystem: 4 = CEMC, 5 = HCALIN, 6 = HCALOUT; 7 and 8 have no tower indices and are skipped |
| 6, 7   | ieta, iphi |
| 8, 9   | truth information (not used) |

The tower positions in the file set the geometry: the eta bin centers, phi offset and radius
of each layer are measured from the events, replacing the uniform grid used for CSV input.
The Topo EMCal-IHCal phi links also follow this geometry (first EMCal iphi behind IHCal iphi
`i` is `4*i + 2`, where the C++ and CSV input use `4*i + 5`).

From Python, `from calo_common import load_towers`, then
`towers = load_towers("../mmaps/calo_features.mmap", event=5)`. Call it before running a
tool, since it installs the measured geometry.

## 0. Setting up (mmap input and jets)

Homebrew's Python does not allow system-wide `pip install`, so `mmap_ninja` and fastjet live
in a virtual environment inside `project_tools/` (already created in this checkout):

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-jets.txt
```

Then run the jet tool with `.venv/bin/python jet_clustering.py ...` (or `source .venv/bin/activate`
first, after which plain `python` works for all the tools).

## 1. Run the example

From `project_tools/`:

```
python3 examples/run_example.py
```

This makes a toy event, runs both clustering tools and the jet finder on it, and prints what
they found next to the particles that were put in. If no Python with fastjet is found, the
jet step is skipped with a message. The event contains:

| particle                 | E [GeV] | eta    | phi    | what it tests                      |
|--------------------------|---------|--------|--------|------------------------------------|
| isolated photon          | 10      | +0.30  | +1.00  | basic position/energy              |
| close pair, photon A + B | 5 + 3   | -0.50 / -0.455 | 2.50 / 2.55 | splitting one cluster into two |
| photon on the phi=0 seam | 4       | 0.00   | 0.005  | phi wrap-around                    |
| hadron                   | 20      | +0.60  | -2.28  | topo clustering across all 3 layers (50% EMCal, 30% IHCal, 20% OHCal) |

plus Gaussian noise in every tower (5 MeV EMCal, 2.5 MeV IHCal, 6 MeV OHCal).

Each photon should come out as its own cluster within dR ~ 0.003 of where it was put. The
reconstructed energies are a few percent low, because towers below threshold and outside
the toy shower footprint are lost. The template tool only sees the EMCal half of the hadron.
The topo tool collects the hadron from all three layers into one cluster. The jet finder
(anti-kt R = 0.4) makes four jets: the hadron (EM/IH/OH fractions ~0.5/0.3/0.2, as
injected), the isolated photon, the close photon pair as a single jet, and the seam photon.

Files written to `examples/`:

- `example_event_emcal.csv`: EMCal towers, input for the template tool
- `example_event_all.csv`: towers from all three layers, input for the topo tool
- `example_template.json`, `example_topo.json`: the clustering output
- `example_jets.json`: the jet output

## 2. Run the tools yourself

```
python3 emcal_template_clustering.py examples/example_event_emcal.csv -o template.json
python3 topo_clustering.py          examples/example_event_all.csv   -o topo.json
```

Without `-o`, the JSON goes to stdout. Some useful options (`--help` lists all of them):

```
# template: event vertex, thresholds, no splitting, drop the per-tower lists
python3 emcal_template_clustering.py examples/example_event_emcal.csv --vz 10 \
    --tower-threshold 0.03 --peak-threshold 0.3 --no-subcluster-splitting --no-towers

# topo: noise levels and seed/grow/perimeter significances, no splitting
python3 topo_clustering.py examples/example_event_all.csv --noise-emcal 0.03 \
    --sigma-seed 4 --sigma-grow 2 --sigma-peri 1 --no-split --no-towers
```

Jets (sPHENIX default: anti-kt R = 0.4 on retowered EMCal + IHCal + OHCal towers):

```
.venv/bin/python jet_clustering.py examples/example_event_all.csv -o jets.json --jet-min-pt 1

# R = 0.2 kt jets, EMCal only at full granularity, |rapidity| < 0.7, with jet areas, vertex z = 10 cm
.venv/bin/python jet_clustering.py examples/example_event_all.csv --algorithm kt -R 0.2 \
    --layers cemc --no-retower --jet-max-eta 0.7 --area --vz 10 --no-constituents
```

Jets from clusters: give the jet finder the JSON written by either clustering tool. Each
cluster becomes one massless input pointing from the vertex to the cluster position (as
sPHENIX's `ClusterJetInput`). The vertex defaults to the one the clusters were made with
(`--vz` overrides it). Clusters with E < 0 are dropped.

```
.venv/bin/python topo_clustering.py examples/example_event_all.csv -o topo.json
.venv/bin/python jet_clustering.py topo.json --jet-min-pt 1          # jets of topoclusters

.venv/bin/python emcal_template_clustering.py examples/example_event_emcal.csv -o emc.json
.venv/bin/python jet_clustering.py emc.json --cluster-energy ecore   # jets of EMCal clusters
```

`--layers` and `--no-retower` apply only to tower input.

Try `--no-subcluster-splitting` (template) or `--no-split` (topo). The close photon pair
then comes out as a single ~7.3 GeV cluster.

## 3. Input format

CSV with a header row; column names are case-insensitive.

| column        | required                 | meaning                                    |
|---------------|--------------------------|--------------------------------------------|
| `eta`, `phi`  | yes                      | tower center; phi in [0, 2pi) or [-pi, pi) |
| `E`           | yes (alias `energy`)     | tower energy [GeV]                         |
| `layer`       | topo/jets: yes, template: no | `cemc`, `hcalin`, `hcalout` (or 2, 0, 1)   |
| `ieta`,`iphi` | no                       | exact tower indices; used instead of binning eta/phi |
| `time`        | no                       | tower time, for the template's energy-weighted mean time |

If you have the tower indices (e.g. from `TowerInfoContainer::getTowerEtaBin/PhiBin`), pass
them: the eta/phi binning here is an approximation of the real geometry.

For topo, also include the low-energy towers rather than only towers above some cut. With
the default `--sigma-peri 0`, every existing neighbour is added as a perimeter tower.

## 4. Output format

```
{
  "algorithm": "RawClusterBuilderTemplate" | "RawClusterBuilderTopo",
  "parameters": {...},          # the settings used
  "n_clusters": N,
  "clusters": [
    {
      "id", "energy", "eta", "phi", "et", "x", "y", "z", "r", "ntowers",
      "towers": [{"ieta", "iphi", "eta", "phi", "energy"(, "layer")}, ...],
      # template only:
      "ecore", "eta_detector", "peak_tower", "max_tower", "tower_cog",
      "tower_moments", "mean_time", "prob"/"chi2" (null: no shower profile loaded),
      # topo only:
      "abs_energy", "energy_emcal", "energy_ihcal", "energy_ohcal"
    }, ...
  ]
}
```

Jets (`jet_clustering.py`):

```
{
  "algorithm": "FastJet antikt", "parameters": {...}, "n_jet_inputs": N, "n_jets": M,
  "jets": [   # sorted by pT
    {
      "id", "pt", "eta", "phi", "rapidity", "e", "px", "py", "pz", "mass",
      "n_constituents", "emcal_frac", "ihcal_frac", "ohcal_frac", "area" (with --area),
      "constituents": [{"input", "ieta", "iphi", "eta", "phi", "energy", "pt"
                        (, "n_emcal_towers" for retowered EMCal)}, ...]
    }, ...
  ]
}
```

With cluster input, `parameters` also holds `input: "clusters"`, `cluster_algorithm` and
`cluster_energy`, there is `n_input_clusters` instead of `n_input_towers`, and each
constituent is `{"input": "TOPO_CLUSTER" or "CEMC_CLUSTER", "cluster_id", "eta", "phi",
"energy", "pt"}`, where `cluster_id` is the cluster's `id` (or position) in the cluster JSON.
For topocluster jets, the EMCal/IHCal/OHCal fractions split each cluster by its
`energy_emcal/ihcal/ohcal`. EMCal cluster jets have `emcal_frac` = 1.

- Retowered EMCal constituents are indexed on the IHCal grid (24 x 64);
  `n_emcal_towers` says how many EMCal towers were summed into each one.
- Jet inputs are massless towers with eta measured from the vertex (`--vz`). Towers with
  E < 0 are dropped, as in sPHENIX's FastJetAlgo (`--constituent-min-e` changes this).

Clusters:

- Template `eta` is relative to the vertex given with `--vz`; `eta_detector` is relative
  to (0,0,0). Topo `eta` is relative to (0,0,0).
- Tower energies in a cluster are the cluster's share of the tower. For a split cluster,
  that can be less than the tower's full energy.

## 5. Using it from Python (e.g. inside an agent tool)

```python
import sys; sys.path.insert(0, "path/to/project_tools")
from calo_common import read_towers, LAYER_EMCAL
from emcal_template_clustering import run_template_clustering
from topo_clustering import run_topo_clustering
from jet_clustering import run_jet_clustering, run_jet_clustering_on_clusters   # needs fastjet

result = run_template_clustering(read_towers("examples/example_event_emcal.csv",
                                             default_layer=LAYER_EMCAL), vz=0.0)
topo = run_topo_clustering(read_towers("examples/example_event_all.csv"), sigma_peri=0.0)
jets = run_jet_clustering(read_towers("examples/example_event_all.csv"), jet_r=0.4, jet_min_pt=1.0)
topo_jets = run_jet_clustering_on_clusters(topo, jet_r=0.4, jet_min_pt=1.0)   # jets of topoclusters
print(result["n_clusters"], topo["n_clusters"], jets["n_jets"], topo_jets["n_jets"])
```
