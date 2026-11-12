#!/usr/bin/env python3
"""Calorimeter jets with FastJet: a Python port of the sPHENIX tower-jet chain, using scikit-hep fastjet.

Ported from coresoftware/offline/packages:
  jetbackground/RetowerCEMC.cc - EMCal towers summed onto the (coarser) IHCal grid
  jetbase/TowerJetInput.cc     - tower -> massless 4-vector, eta measured from the event vertex
  jetbase/FastJetAlgo.cc       - jet definition (E-scheme), constituent/jet selection,
                                 jet areas, EMCal/IHCal/OHCal energy fractions

Standard sPHENIX tower jets use retowered EMCal + IHCal + OHCal towers, anti-kt R = 0.4.

Differences from the C++:
  * Geometry is the "former" approximate geometry with uniform eta/phi bins (no CDB); the
    retowering maps each EMCal tower to the IHCal bin containing its center.
  * Tower quality flags (isGood) are not available: no masking and no bad-tower rescaling
    in the retowering.
  * Background subtraction (SubtractTowers / constituent subtraction) and SoftDrop are not
    included.

Input: CSV with columns layer, eta, phi, E (+ optional ieta, iphi). Output: JSON.
Needs the scikit-hep fastjet package (pip install fastjet).
"""

import argparse
import contextlib
import json
import math
import os
import sys

from calo_common import (GEOMETRY, LAYER_EMCAL, LAYER_IHCAL, LAYER_NAMES, LAYER_OHCAL,
                         finite_or_none, load_towers, parse_layer)

try:
    import fastjet
except ImportError:
    sys.exit("jet_clustering.py needs the scikit-hep fastjet package: pip install fastjet "
             "(or use project_tools/.venv/bin/python)")

ALGORITHMS = {
    "antikt": fastjet.antikt_algorithm,
    "kt": fastjet.kt_algorithm,
    "cambridge": fastjet.cambridge_algorithm,
}

# component label of each input: the per-layer jet input in sPHENIX
INPUT_NAMES = {
    LAYER_EMCAL: "CEMC_TOWERINFO",
    LAYER_IHCAL: "HCALIN_TOWERINFO",
    LAYER_OHCAL: "HCALOUT_TOWERINFO",
}


@contextlib.contextmanager
def stdout_to_devnull():
    """Silence the FastJet banner, which the C++ library prints to stdout on first use."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


class JetInput:
    """One jet-finding input: a (possibly retowered) tower as a massless 4-vector."""

    __slots__ = ("layer", "ieta", "iphi", "energy", "eta", "phi", "px", "py", "pz", "n_merged")

    def __init__(self, layer, ieta, iphi, energy, eta, phi, n_merged=1):
        self.layer = layer
        self.ieta = ieta
        self.iphi = iphi
        self.energy = energy
        self.eta = eta
        self.phi = phi
        pt = energy / math.cosh(eta)
        self.px = pt * math.cos(phi)
        self.py = pt * math.sin(phi)
        self.pz = pt * math.sinh(eta)
        self.n_merged = n_merged


def retower_cemc(towers):
    """RetowerCEMC::get_fraction - sum EMCal tower energies onto the IHCal grid.

    Returns {(ieta_ihcal, iphi_ihcal): (energy, n_emcal_towers)}.
    """
    gem = GEOMETRY[LAYER_EMCAL]
    gih = GEOMETRY[LAYER_IHCAL]
    retowered = {}
    for t in towers:
        if t.layer != LAYER_EMCAL:
            continue
        key = (gih.get_etabin(gem.get_etacenter(t.ieta)), gih.get_phibin(gem.get_phicenter(t.iphi)))
        e, n = retowered.get(key, (0.0, 0))
        retowered[key] = (e + t.energy, n + 1)
    return retowered


def tower_kinematics(layer, ieta, iphi, vz, radius=None):
    """TowerJetInput: tower direction with eta measured from the vertex at the tower radius."""
    geom = GEOMETRY[layer]
    x, y, z0 = geom.get_center(ieta, iphi)
    r = math.hypot(x, y) if radius is None else radius
    tower_eta = math.asinh(z0 / math.hypot(x, y))
    eta = math.asinh((math.sinh(tower_eta) * r - vz) / r)
    return eta, math.atan2(y, x)


def build_inputs(towers, vz=0.0, layers=(LAYER_EMCAL, LAYER_IHCAL, LAYER_OHCAL), retower=True):
    inputs = []
    if LAYER_EMCAL in layers:
        if retower:
            # retowered EMCal towers sit on the IHCal grid but use the EMCal radius
            r_emcal = GEOMETRY[LAYER_EMCAL].center_radius
            for (ieta, iphi), (energy, n) in sorted(retower_cemc(towers).items()):
                if math.isnan(energy):
                    continue
                eta, phi = tower_kinematics(LAYER_IHCAL, ieta, iphi, vz, radius=r_emcal)
                inputs.append(JetInput(LAYER_EMCAL, ieta, iphi, energy, eta, phi, n_merged=n))
        else:
            for t in towers:
                if t.layer == LAYER_EMCAL and not math.isnan(t.energy):
                    eta, phi = tower_kinematics(LAYER_EMCAL, t.ieta, t.iphi, vz)
                    inputs.append(JetInput(LAYER_EMCAL, t.ieta, t.iphi, t.energy, eta, phi))
    for t in towers:
        if t.layer in (LAYER_IHCAL, LAYER_OHCAL) and t.layer in layers and not math.isnan(t.energy):
            eta, phi = tower_kinematics(t.layer, t.ieta, t.iphi, vz)
            inputs.append(JetInput(t.layer, t.ieta, t.iphi, t.energy, eta, phi))
    return inputs


def run_jet_clustering(towers, jet_r=0.4, algorithm="antikt", vz=0.0,
                       layers=(LAYER_EMCAL, LAYER_IHCAL, LAYER_OHCAL), retower=True,
                       constituent_min_e=0.0, constituent_min_pt=None,
                       jet_min_pt=None, jet_max_eta=None,
                       calc_area=False, ghost_area=0.01, ghost_max_rap=None,
                       save_constituents=True):
    """Find jets in calorimeter towers.

    towers: iterable of calo_common.Tower. layers: which calorimeter layers to use.
    retower: sum EMCal towers onto the HCal grid first (sPHENIX default).
    constituent_min_e: inputs with E below this are dropped (default 0: negative towers removed).
    jet_min_pt / jet_max_eta: jet selection (|rapidity| as in fastjet::SelectorAbsRapMax).
    Returns a JSON-serialisable dict.
    """
    towers = list(towers)
    inputs = build_inputs(towers, vz=vz, layers=layers, retower=retower)

    # FastJetAlgo::jets_to_pseudojets
    pseudojets = []
    for i, inp in enumerate(inputs):
        if inp.energy < constituent_min_e:
            continue
        pj = fastjet.PseudoJet(inp.px, inp.py, inp.pz, inp.energy)
        if constituent_min_pt is not None and pj.perp() < constituent_min_pt:
            continue
        pj.set_user_index(i)
        pseudojets.append(pj)

    # Same as fastjet::JetDefinition(algo, R, E_scheme, Best) in FastJetAlgo. Do not pass Best
    # explicitly: the Python bindings treat enums as ints and resolve that call to the
    # (algo, R, extra_param, recomb_scheme) overload, i.e. pt-scheme with the O(N^3) strategy.
    jetdef = fastjet.JetDefinition(ALGORITHMS[algorithm], jet_r, fastjet.E_scheme)
    if jetdef.recombination_scheme() != fastjet.E_scheme or jetdef.strategy() != fastjet.Best:
        raise RuntimeError(f"unexpected FastJet jet definition: {jetdef.description()}")
    selector = None
    if jet_max_eta is not None and jet_min_pt is not None:
        selector = fastjet.SelectorAbsRapMax(jet_max_eta) & fastjet.SelectorPtMin(jet_min_pt)
    elif jet_max_eta is not None:
        selector = fastjet.SelectorAbsRapMax(jet_max_eta)
    elif jet_min_pt is not None:
        selector = fastjet.SelectorPtMin(jet_min_pt)

    if calc_area and ghost_max_rap is None:
        ghost_max_rap = jet_max_eta + jet_r if jet_max_eta is not None else 5.0

    with stdout_to_devnull():
        if calc_area:
            area_def = fastjet.AreaDefinition(fastjet.active_area_explicit_ghosts,
                                              fastjet.GhostedAreaSpec(ghost_max_rap, 1, ghost_area))
            cluseq = fastjet.ClusterSequenceArea(pseudojets, jetdef, area_def)
            not_ghost = ~fastjet.SelectorIsPureGhost()
            jets = (not_ghost & selector if selector is not None else not_ghost)(cluseq.inclusive_jets())
        else:
            cluseq = fastjet.ClusterSequence(pseudojets, jetdef)
            jets = cluseq.inclusive_jets()
            if selector is not None:
                jets = selector(jets)
        jets = fastjet.sorted_by_pt(jets)

    jets_out = []
    for ijet, jet in enumerate(jets):
        layer_e = {LAYER_EMCAL: 0.0, LAYER_IHCAL: 0.0, LAYER_OHCAL: 0.0}
        constituents = []
        for comp in jet.constituents():
            if calc_area and comp.is_pure_ghost():
                continue
            inp = inputs[comp.user_index()]
            layer_e[inp.layer] += comp.e()
            if save_constituents:
                c = {
                    "input": ("CEMC_TOWERINFO_RETOWER" if (retower and inp.layer == LAYER_EMCAL)
                              else INPUT_NAMES[inp.layer]),
                    "ieta": inp.ieta,
                    "iphi": inp.iphi,
                    "eta": inp.eta,
                    "phi": inp.phi,
                    "energy": inp.energy,
                    "pt": comp.pt(),
                }
                if retower and inp.layer == LAYER_EMCAL:
                    c["n_emcal_towers"] = inp.n_merged
                constituents.append(c)
        e = jet.e()
        out = {
            "id": ijet,
            "pt": jet.pt(),
            "eta": jet.eta(),
            "phi": jet.phi_std(),
            "rapidity": jet.rap(),
            "e": e,
            "px": jet.px(),
            "py": jet.py(),
            "pz": jet.pz(),
            "mass": jet.m(),
            "n_constituents": sum(1 for c in jet.constituents() if not (calc_area and c.is_pure_ghost())),
            "emcal_frac": layer_e[LAYER_EMCAL] / e if e != 0 else None,
            "ihcal_frac": layer_e[LAYER_IHCAL] / e if e != 0 else None,
            "ohcal_frac": layer_e[LAYER_OHCAL] / e if e != 0 else None,
        }
        if calc_area:
            out["area"] = jet.area()
        if save_constituents:
            out["constituents"] = sorted(constituents, key=lambda c: -c["energy"])
        jets_out.append({k: finite_or_none(v) for k, v in out.items()})

    n_by_layer = {LAYER_NAMES[l]: sum(1 for t in towers if t.layer == l) for l in LAYER_NAMES}
    return {
        "algorithm": f"FastJet {algorithm}",
        "parameters": {
            "R": jet_r,
            "recombination": "E_scheme",
            "vertex_z": vz,
            "layers": [LAYER_NAMES[l] for l in layers],
            "retower_emcal": retower,
            "constituent_min_e": constituent_min_e,
            "constituent_min_pt": constituent_min_pt,
            "jet_min_pt": jet_min_pt,
            "jet_max_eta": jet_max_eta,
            "calc_area": calc_area,
            "ghost_area": ghost_area if calc_area else None,
            "ghost_max_rap": ghost_max_rap if calc_area else None,
            "geometry": "former (approximate, uniform eta/phi bins)",
        },
        "n_input_towers": n_by_layer,
        "n_jet_inputs": len(pseudojets),
        "n_jets": len(jets_out),
        "jets": jets_out,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="calo_features.mmap directory (with --event), or CSV file with columns layer, eta, phi, E (optional: ieta, iphi)")
    p.add_argument("--event", type=int, help="event number in the mmap directory")
    p.add_argument("-o", "--output", help="output JSON file (default: stdout)")
    p.add_argument("-R", "--radius", type=float, default=0.4, help="jet radius (default 0.4)")
    p.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="antikt", help="jet algorithm (default antikt)")
    p.add_argument("--vz", type=float, default=0.0, help="event vertex z [cm] (default 0)")
    p.add_argument("--layers", default="cemc,hcalin,hcalout",
                   help="comma-separated layers to use (default cemc,hcalin,hcalout)")
    p.add_argument("--no-retower", action="store_true", help="use EMCal towers at full granularity instead of retowering")
    p.add_argument("--constituent-min-e", type=float, default=0.0,
                   help="drop inputs with E below this [GeV] (default 0: negative towers dropped)")
    p.add_argument("--constituent-min-pt", type=float, help="drop inputs with pT below this [GeV]")
    p.add_argument("--jet-min-pt", type=float, help="minimum jet pT [GeV]")
    p.add_argument("--jet-max-eta", type=float, help="maximum jet |rapidity|")
    p.add_argument("--area", action="store_true", help="calculate jet areas with ghosts")
    p.add_argument("--ghost-area", type=float, default=0.01, help="ghost area (default 0.01)")
    p.add_argument("--no-constituents", action="store_true", help="omit per-jet constituent lists from the output")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (negative for compact)")
    args = p.parse_args(argv)

    layers = tuple(parse_layer(l) for l in args.layers.split(",") if l.strip())

    towers = load_towers(args.input, event=args.event)
    result = run_jet_clustering(
        towers, jet_r=args.radius, algorithm=args.algorithm, vz=args.vz, layers=layers,
        retower=not args.no_retower, constituent_min_e=args.constituent_min_e,
        constituent_min_pt=args.constituent_min_pt, jet_min_pt=args.jet_min_pt,
        jet_max_eta=args.jet_max_eta, calc_area=args.area, ghost_area=args.ghost_area,
        save_constituents=not args.no_constituents,
    )
    if args.event is not None:
        result["event"] = args.event

    text = json.dumps(result, indent=args.indent if args.indent >= 0 else None, allow_nan=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
