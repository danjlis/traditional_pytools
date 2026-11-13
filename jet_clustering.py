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

Input: one event of a calo_features.mmap directory, a CSV of towers (layer, eta, phi, E,
optional ieta, iphi), or a cluster JSON from emcal_template_clustering.py or topo_clustering.py.
With a cluster JSON each cluster becomes one massless input (as jetbase/ClusterJetInput),
pointing from the vertex to the cluster position; topocluster EMCal/IHCal/OHCal energies set
the jet layer fractions. Output: JSON.
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
    """One jet-finding input as a massless 4-vector: a (possibly retowered) tower, or a cluster.

    For a cluster, layer is None, cluster_id is its id in the cluster JSON and fracs gives the
    fraction of its energy in each layer (used for the jet EMCal/IHCal/OHCal fractions)."""

    __slots__ = ("layer", "ieta", "iphi", "energy", "eta", "phi", "px", "py", "pz", "n_merged",
                 "label", "cluster_id", "fracs")

    def __init__(self, layer, ieta, iphi, energy, eta, phi, n_merged=1,
                 label=None, cluster_id=None, fracs=None):
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
        self.label = label
        self.cluster_id = cluster_id
        self.fracs = fracs if fracs is not None else {layer: 1.0}


def _retower_eta_map(gem, gih):
    """RetowerCEMC::get_weighted_fraction: for each IHCal eta bin, the first and last EMCal eta
    bins it covers and the fraction of each of those two that lies inside it."""
    lower, upper = [], []
    ie = 0
    for ih in range(gih.neta):
        ih_lo, ih_hi = gih.get_etabounds(ih)
        found_lo = found_hi = False
        while (not found_lo or not found_hi) and ie < gem.neta:
            em_lo, em_hi = gem.get_etabounds(ie)
            if not found_lo:
                if em_hi > ih_lo and em_lo <= ih_lo:
                    lower.append((ie, (em_hi - ih_lo) / (em_hi - em_lo)))
                    found_lo = True
                if em_hi > ih_lo and em_lo > ih_lo:
                    lower.append((ie, 1.0))
                    found_lo = True
            else:
                if em_hi >= ih_hi and em_lo < ih_hi:
                    upper.append((ie, (ih_hi - em_lo) / (em_hi - em_lo)))
                    found_hi = True
                if em_hi > ih_hi and em_lo > ih_hi:
                    ie -= 1
                    upper.append((ie, 1.0))
                    found_hi = True
            if not (found_lo and found_hi):
                ie += 1
        if not (found_lo and found_hi):
            raise RuntimeError(f"retowering: no EMCal eta bins found for IHCal ieta {ih}")
    return lower, upper


def _retower_first_phi(gem, gih):
    """RetowerCEMC::get_first_phi_index: first EMCal iphi of the four behind IHCal iphi 0."""
    first = next((k for k in range(gem.nphi) if gih.get_phibin(gem.get_phicenter(k)) == 0), None)
    if first is None:
        raise RuntimeError("retowering: no EMCal tower behind IHCal iphi 0")
    if first == 0:
        k = gem.nphi - 1
        while k > 0 and gih.get_phibin(gem.get_phicenter(k)) != gih.nphi - 1:
            k -= 1
        first = 0 if k + 1 == gem.nphi else k + 1
    return first


def retower_cemc(towers):
    """RetowerCEMC (weighted energy distribution, the default): sum EMCal towers onto the
    IHCal grid. Each IHCal tower takes 4 EMCal towers in phi and the EMCal eta bins inside its
    eta bounds, the two boundary bins weighted by their overlap. Bad towers must already be
    removed from `towers` (C++ skips !isGood towers).

    Returns {(ieta_ihcal, iphi_ihcal): (energy, n_emcal_towers)} for every IHCal tower that has
    at least one EMCal input tower.
    """
    gem = GEOMETRY[LAYER_EMCAL]
    gih = GEOMETRY[LAYER_IHCAL]
    lower, upper = _retower_eta_map(gem, gih)
    first_phi = _retower_first_phi(gem, gih)
    emcal = {(t.ieta, t.iphi): t.energy for t in towers if t.layer == LAYER_EMCAL}
    retowered = {}
    for ih in range(gih.neta):
        (ie_lo, f_lo), (ie_hi, f_hi) = lower[ih], upper[ih]
        for ip in range(gih.nphi):
            energy, n = 0.0, 0
            for ie in range(ie_lo, ie_hi + 1):
                frac = f_lo if ie == ie_lo else (f_hi if ie == ie_hi else 1.0)
                for k in range(4):
                    e = emcal.get((ie, (first_phi + 4 * ip + k) % gem.nphi))
                    if e is not None:
                        energy += e * frac
                        n += 1
            if n:
                retowered[(ih, ip)] = (energy, n)
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


# ---------------------------------------------------------------------------------------------
# Clusters as jet inputs (sPHENIX jetbase/ClusterJetInput)
# ---------------------------------------------------------------------------------------------

# cluster JSON "algorithm" -> constituent label
CLUSTER_INPUT_NAMES = {
    "RawClusterBuilderTemplate": "CEMC_CLUSTER",
    "RawClusterBuilderTopo": "TOPO_CLUSTER",
}


def is_cluster_json(path):
    return os.path.isfile(path) and path.lower().endswith(".json")


def read_cluster_json(path, event=None):
    """Read the output of emcal_template_clustering.py or topo_clustering.py.

    The file holds one event (a dict with "clusters"), or a list of such dicts, from which
    `event` picks the one whose "event" field matches (or the event-th entry if none has one).
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if event is None:
            if len(data) != 1:
                raise ValueError(f"{path}: holds {len(data)} events; pick one with --event N")
            data = data[0]
        else:
            matches = [d for d in data if d.get("event") == event]
            if matches:
                data = matches[0]
            elif 0 <= event < len(data) and all("event" not in d for d in data):
                data = data[event]
            else:
                raise ValueError(f"{path}: no entry for event {event}")
    elif event is not None and data.get("event") is not None and data["event"] != event:
        raise ValueError(f"{path}: holds event {data['event']}, not {event}")
    if not isinstance(data, dict) or "clusters" not in data:
        raise ValueError(f"{path}: not a cluster JSON (expected the output of "
                         "emcal_template_clustering.py or topo_clustering.py)")
    return data


def cluster_vertex_z(cluster_result):
    """Vertex the clusters were made with (Template records it; Topo uses 0)."""
    return cluster_result.get("parameters", {}).get("vertex_z", 0.0) or 0.0


def build_cluster_inputs(cluster_result, vz=0.0, energy="energy"):
    """ClusterJetInput: each cluster -> massless 4-vector of its energy pointing from the vertex
    (0, 0, vz) to the cluster position (x, y, z).

    energy: which cluster energy to use, "energy" (default) or "ecore" (Template only).
    """
    algorithm = cluster_result.get("algorithm", "")
    label = CLUSTER_INPUT_NAMES.get(algorithm, "CLUSTER")
    inputs = []
    for i, cl in enumerate(cluster_result["clusters"]):
        e = cl.get(energy)
        if e is None:
            raise ValueError(f"cluster {i} has no '{energy}' (ecore exists only for Template clusters)")
        x, y, z = cl["x"], cl["y"], cl["z"]
        r = math.hypot(x, y)
        if not math.isfinite(e) or r == 0:
            continue
        eta = math.asinh((z - vz) / r)
        phi = math.atan2(y, x)
        if "energy_emcal" in cl:
            # Topo: split by the cluster's energy in each layer
            parts = {LAYER_EMCAL: cl["energy_emcal"], LAYER_IHCAL: cl["energy_ihcal"],
                     LAYER_OHCAL: cl["energy_ohcal"]}
            tot = sum(parts.values())
            fracs = ({l: v / tot for l, v in parts.items()} if tot != 0
                     else {LAYER_EMCAL: 1.0 / 3, LAYER_IHCAL: 1.0 / 3, LAYER_OHCAL: 1.0 / 3})
        else:
            fracs = {LAYER_EMCAL: 1.0}
        inputs.append(JetInput(None, None, None, e, eta, phi, label=label,
                               cluster_id=cl.get("id", i), fracs=fracs))
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
    for inp in inputs:
        if retower and inp.layer == LAYER_EMCAL:
            inp.label = "CEMC_TOWERINFO_RETOWER"
        else:
            inp.label = INPUT_NAMES[inp.layer]
    result = _find_jets(inputs, jet_r=jet_r, algorithm=algorithm,
                        constituent_min_e=constituent_min_e, constituent_min_pt=constituent_min_pt,
                        jet_min_pt=jet_min_pt, jet_max_eta=jet_max_eta, calc_area=calc_area,
                        ghost_area=ghost_area, ghost_max_rap=ghost_max_rap,
                        save_constituents=save_constituents)
    result["algorithm"] = f"FastJet {algorithm}"
    result["parameters"].update({
        "input": "towers",
        "vertex_z": vz,
        "layers": [LAYER_NAMES[l] for l in layers],
        "retower_emcal": retower,
        "geometry": "former (approximate, uniform eta/phi bins)",
    })
    result["n_input_towers"] = {LAYER_NAMES[l]: sum(1 for t in towers if t.layer == l) for l in LAYER_NAMES}
    return _order_result(result)


def run_jet_clustering_on_clusters(cluster_result, jet_r=0.4, algorithm="antikt", vz=None,
                                   energy="energy", constituent_min_e=0.0, constituent_min_pt=None,
                                   jet_min_pt=None, jet_max_eta=None,
                                   calc_area=False, ghost_area=0.01, ghost_max_rap=None,
                                   save_constituents=True):
    """Find jets in clusters: cluster_result is the dict written by emcal_template_clustering.py
    or topo_clustering.py (see read_cluster_json). vz defaults to the vertex the clusters were
    made with. Negative-energy (topo)clusters are dropped by constituent_min_e = 0."""
    if vz is None:
        vz = cluster_vertex_z(cluster_result)
    inputs = build_cluster_inputs(cluster_result, vz=vz, energy=energy)
    result = _find_jets(inputs, jet_r=jet_r, algorithm=algorithm,
                        constituent_min_e=constituent_min_e, constituent_min_pt=constituent_min_pt,
                        jet_min_pt=jet_min_pt, jet_max_eta=jet_max_eta, calc_area=calc_area,
                        ghost_area=ghost_area, ghost_max_rap=ghost_max_rap,
                        save_constituents=save_constituents)
    result["algorithm"] = f"FastJet {algorithm}"
    result["parameters"].update({
        "input": "clusters",
        "cluster_algorithm": cluster_result.get("algorithm"),
        "cluster_energy": energy,
        "vertex_z": vz,
    })
    result["n_input_clusters"] = len(cluster_result["clusters"])
    if "event" in cluster_result:
        result["event"] = cluster_result["event"]
    return _order_result(result)


def _order_result(result):
    order = ["algorithm", "event", "parameters", "n_input_towers", "n_input_clusters",
             "n_jet_inputs", "n_jets", "jets"]
    return {k: result[k] for k in order if k in result}


def _find_jets(inputs, jet_r, algorithm, constituent_min_e, constituent_min_pt, jet_min_pt,
               jet_max_eta, calc_area, ghost_area, ghost_max_rap, save_constituents):
    """FastJetAlgo on a list of JetInput. Returns the result dict without input-specific keys."""

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
            for l, f in inp.fracs.items():
                layer_e[l] += f * comp.e()
            if save_constituents:
                c = {"input": inp.label}
                if inp.cluster_id is not None:
                    c["cluster_id"] = inp.cluster_id
                else:
                    c["ieta"] = inp.ieta
                    c["iphi"] = inp.iphi
                c.update({"eta": inp.eta, "phi": inp.phi, "energy": inp.energy, "pt": comp.pt()})
                if inp.label == "CEMC_TOWERINFO_RETOWER":
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

    return {
        "parameters": {
            "R": jet_r,
            "recombination": "E_scheme",
            "constituent_min_e": constituent_min_e,
            "constituent_min_pt": constituent_min_pt,
            "jet_min_pt": jet_min_pt,
            "jet_max_eta": jet_max_eta,
            "calc_area": calc_area,
            "ghost_area": ghost_area if calc_area else None,
            "ghost_max_rap": ghost_max_rap if calc_area else None,
        },
        "n_jet_inputs": len(pseudojets),
        "n_jets": len(jets_out),
        "jets": jets_out,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="calo_features.mmap directory (with --event), CSV file with columns "
                   "layer, eta, phi, E (optional: ieta, iphi), or a cluster JSON written by "
                   "emcal_template_clustering.py / topo_clustering.py (jets are then made from the clusters)")
    p.add_argument("--event", type=int, help="event number in the mmap directory (or in a multi-event cluster JSON)")
    p.add_argument("-o", "--output", help="output JSON file (default: stdout)")
    p.add_argument("-R", "--radius", type=float, default=0.4, help="jet radius (default 0.4)")
    p.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="antikt", help="jet algorithm (default antikt)")
    p.add_argument("--vz", type=float, help="event vertex z [cm] (default 0; for cluster JSON, the "
                   "vertex the clusters were made with)")
    p.add_argument("--cluster-energy", choices=["energy", "ecore"], default="energy",
                   help="cluster energy used for cluster JSON input (ecore: Template only)")
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

    common = dict(jet_r=args.radius, algorithm=args.algorithm,
                  constituent_min_e=args.constituent_min_e, constituent_min_pt=args.constituent_min_pt,
                  jet_min_pt=args.jet_min_pt, jet_max_eta=args.jet_max_eta, calc_area=args.area,
                  ghost_area=args.ghost_area, save_constituents=not args.no_constituents)

    if is_cluster_json(args.input):
        try:
            clusters = read_cluster_json(args.input, event=args.event)
        except ValueError as e:
            p.error(str(e))
        result = run_jet_clustering_on_clusters(clusters, vz=args.vz, energy=args.cluster_energy, **common)
        if args.event is not None and "event" not in result:
            result = _order_result({**result, "event": args.event})
    else:
        layers = tuple(parse_layer(l) for l in args.layers.split(",") if l.strip())
        towers = load_towers(args.input, event=args.event)
        result = run_jet_clustering(towers, vz=args.vz if args.vz is not None else 0.0, layers=layers,
                                    retower=not args.no_retower, **common)
        if args.event is not None:
            result = _order_result({**result, "event": args.event})

    text = json.dumps(result, indent=args.indent if args.indent >= 0 else None, allow_nan=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
