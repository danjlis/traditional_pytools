#!/usr/bin/env python3
"""Worked example for the calorimeter clustering tools.

1. Builds a toy event with known particles (photons + one hadron) and Gaussian noise,
   and writes it as tower CSV files:
     example_event_emcal.csv  - EMCal only            (eta, phi, E)
     example_event_all.csv    - EMCal + IHCal + OHCal (layer, eta, phi, E)
2. Runs emcal_template_clustering.py, topo_clustering.py and jet_clustering.py on them,
   exactly as you would from the command line, writing example_template.json,
   example_topo.json and example_jets.json.
3. Prints the reconstructed clusters and jets next to the particles that were put in.

The jet step needs the scikit-hep fastjet package. It uses this Python if fastjet imports,
otherwise project_tools/.venv/bin/python, otherwise it is skipped.

Run:  python3 examples/run_example.py
"""

import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)

from calo_common import GEOMETRY, LAYER_EMCAL, LAYER_IHCAL, LAYER_OHCAL  # noqa: E402
from emcal_template_clustering import BEmcRecCEMC  # noqa: E402

LAYER_CSV_NAME = {LAYER_EMCAL: "cemc", LAYER_IHCAL: "hcalin", LAYER_OHCAL: "hcalout"}

# The particles in the toy event: (label, energy [GeV], eta, phi)
PHOTONS = [
    ("isolated photon", 10.0, 0.30, 1.00),
    ("close pair, photon A", 5.0, -0.50, 2.50),
    ("close pair, photon B", 3.0, -0.455, 2.55),
    ("photon on the phi=0 seam", 4.0, 0.00, 0.005),
]
HADRON = ("hadron (EM+IH+OH)", 20.0, 0.60, 4.00)
NOISE = {LAYER_EMCAL: 0.005, LAYER_IHCAL: 0.0025, LAYER_OHCAL: 0.006}  # GeV per tower


def add_em_shower(towers, energy, eta, phi):
    """Spread energy over EMCal towers with the same transverse profile the template uses."""
    g = GEOMETRY[LAYER_EMCAL]
    yc = (eta + 1.1) / g.deta - 0.5  # shower center in tower units
    xc = (phi % (2 * math.pi)) / g.dphi - 0.5
    for iy in range(int(yc) - 3, int(yc) + 5):
        for ix in range(int(xc) - 3, int(xc) + 5):
            if 0 <= iy < g.neta:
                frac = BEmcRecCEMC.predict_energy_param(energy, abs(ix - xc), iy - yc)
                key = (LAYER_EMCAL, iy, ix % g.nphi)
                towers[key] = towers.get(key, 0.0) + energy * frac


def add_hadron_shower(towers, energy, eta, phi, fractions=(0.3, 0.2, 0.5)):
    """Crude hadronic shower: fractions in (IHCal, OHCal, EMCal)."""
    add_em_shower(towers, energy * fractions[2], eta, phi)
    for layer, frac in ((LAYER_IHCAL, fractions[0]), (LAYER_OHCAL, fractions[1])):
        g = GEOMETRY[layer]
        ieta, iphi = g.get_etabin(eta), g.get_phibin(phi)
        weights = {(de, dp): (0.6 if de == dp == 0 else 0.05) for de in (-1, 0, 1) for dp in (-1, 0, 1)}
        for (de, dp), w in weights.items():
            if 0 <= ieta + de < g.neta:
                key = (layer, ieta + de, (iphi + dp) % g.nphi)
                towers[key] = towers.get(key, 0.0) + energy * frac * w


def make_event(seed=2026):
    random.seed(seed)
    towers = {}
    for _, energy, eta, phi in PHOTONS:
        add_em_shower(towers, energy, eta, phi)
    add_hadron_shower(towers, HADRON[1], HADRON[2], HADRON[3])
    for layer, sigma in NOISE.items():
        g = GEOMETRY[layer]
        for ieta in range(g.neta):
            for iphi in range(g.nphi):
                key = (layer, ieta, iphi)
                towers[key] = towers.get(key, 0.0) + random.gauss(0.0, sigma)
    return towers


def write_csvs(towers, path_emcal, path_all):
    with open(path_emcal, "w") as fe, open(path_all, "w") as fa:
        fe.write("eta,phi,E\n")
        fa.write("layer,eta,phi,E\n")
        for (layer, ieta, iphi), energy in sorted(towers.items()):
            g = GEOMETRY[layer]
            row = f"{g.get_etacenter(ieta):.5f},{g.get_phicenter(iphi):.5f},{energy:.6f}\n"
            fa.write(f"{LAYER_CSV_NAME[layer]},{row}")
            if layer == LAYER_EMCAL:
                fe.write(row)


def run(cmd, python=sys.executable):
    shown = [os.path.relpath(c, TOOLS) if os.path.isabs(c) else c for c in cmd]
    prefix = "" if python == sys.executable else os.path.relpath(python, TOOLS) + " "
    print("$ " + prefix + " ".join(shown))
    subprocess.run([python] + cmd, check=True, cwd=TOOLS)


def find_fastjet_python():
    """A Python interpreter that has the scikit-hep fastjet package, or None."""
    candidates = [sys.executable, os.path.join(TOOLS, ".venv", "bin", "python")]
    for python in candidates:
        if os.path.exists(python) and subprocess.run([python, "-c", "import fastjet"],
                                                     capture_output=True).returncode == 0:
            return python
    return None


def wrap_phi(phi):
    return math.remainder(phi, 2 * math.pi)


def closest_truth(cluster, truths):
    best = None
    for label, energy, eta, phi in truths:
        dr = math.hypot(cluster["eta"] - eta, wrap_phi(cluster["phi"] - phi))
        if best is None or dr < best[1]:
            best = (label, dr, energy)
    return best


def print_clusters(title, clusters, truths, extra=None, min_e=0.5):
    shown = sorted((c for c in clusters if c["energy"] > min_e), key=lambda c: -c["energy"])
    print(f"\n{title}: {len(clusters)} clusters in total, {len(shown)} with E > {min_e} GeV")
    header = f"  {'E [GeV]':>8} {'eta':>8} {'phi':>8} {'ntwr':>5}"
    if extra:
        header += f" {extra[0]:>16}"
    print(header + "   closest particle (dR, E_true)")
    for c in shown:
        label, dr, e_true = closest_truth(c, truths)
        line = f"  {c['energy']:8.3f} {c['eta']:+8.4f} {wrap_phi(c['phi']):+8.4f} {c['ntowers']:5d}"
        if extra:
            line += f" {extra[1](c):>16}"
        print(line + f"   {label} ({dr:.3f}, {e_true:g})")


def main():
    path_emcal = os.path.join(HERE, "example_event_emcal.csv")
    path_all = os.path.join(HERE, "example_event_all.csv")
    out_template = os.path.join(HERE, "example_template.json")
    out_topo = os.path.join(HERE, "example_topo.json")
    out_jets = os.path.join(HERE, "example_jets.json")

    print("== 1. Make a toy event ==")
    towers = make_event()
    write_csvs(towers, path_emcal, path_all)
    print(f"wrote {os.path.relpath(path_emcal, TOOLS)} and {os.path.relpath(path_all, TOOLS)}")
    print("particles in the event:")
    for label, energy, eta, phi in PHOTONS + [HADRON]:
        print(f"  {label:<28} E = {energy:5.1f} GeV  eta = {eta:+.3f}  phi = {wrap_phi(phi):+.3f}")

    print("\n== 2. Run the clustering tools ==")
    run(["emcal_template_clustering.py", path_emcal, "-o", out_template])
    run(["topo_clustering.py", path_all, "-o", out_topo])
    fastjet_python = find_fastjet_python()
    if fastjet_python:
        run(["jet_clustering.py", path_all, "-o", out_jets, "--jet-min-pt", "1"], python=fastjet_python)
    else:
        print("skipping jet_clustering.py: no Python with the fastjet package found "
              "(see examples/README.md, 'Setting up')")

    print("\n== 3. Look at the output ==")
    with open(out_template) as f:
        template = json.load(f)
    with open(out_topo) as f:
        topo = json.load(f)
    # the template sees only the EMCal part of the hadron (half its energy)
    em_truths = PHOTONS + [(HADRON[0] + ", EM part", HADRON[1] * 0.5, HADRON[2], HADRON[3])]
    print_clusters("Template (EMCal)", template["clusters"], em_truths,
                   extra=("ecore", lambda c: f"{c['ecore']:.3f}"))
    print_clusters("Topo (EMCal+IHCal+OHCal)", topo["clusters"], PHOTONS + [HADRON],
                   extra=("EM/IH/OH [GeV]",
                          lambda c: f"{c['energy_emcal']:.1f}/{c['energy_ihcal']:.1f}/{c['energy_ohcal']:.1f}"))

    if fastjet_python:
        with open(out_jets) as f:
            jets = json.load(f)
        # jets are compared in pT: E_true / cosh(eta_true); the photon pair is one jet at R = 0.4
        jet_truths = [(label, e / math.cosh(eta), eta, phi) for label, e, eta, phi in PHOTONS + [HADRON]]
        print(f"\nJets (anti-kt R=0.4, retowered EMCal + IHCal + OHCal): {jets['n_jets']} with pT > 1 GeV")
        print(f"  {'pT [GeV]':>8} {'eta':>8} {'phi':>8} {'nconst':>6} {'EM/IH/OH frac':>16}   closest particle (dR, pT_true)")
        for j in jets["jets"]:
            label, dr, pt_true = closest_truth(j, jet_truths)
            fracs = f"{j['emcal_frac']:.2f}/{j['ihcal_frac']:.2f}/{j['ohcal_frac']:.2f}"
            print(f"  {j['pt']:8.3f} {j['eta']:+8.4f} {j['phi']:+8.4f} {j['n_constituents']:6d} {fracs:>16}"
                  f"   {label} ({dr:.3f}, {pt_true:.2f})")

    print("\nFirst template cluster as stored in the JSON (tower list truncated):")
    c = dict(max(template["clusters"], key=lambda c: c["energy"]))
    c["towers"] = c["towers"][:2] + [f"... {len(c['towers']) - 2} more"]
    print(json.dumps(c, indent=2))
    outputs = [out_template, out_topo] + ([out_jets] if fastjet_python else [])
    print("\nFull output: " + ", ".join(os.path.relpath(o, TOOLS) for o in outputs))


if __name__ == "__main__":
    main()
