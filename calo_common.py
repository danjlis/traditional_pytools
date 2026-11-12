"""Shared tower input and geometry helpers for the sPHENIX calorimeter clustering tools.

Geometry follows CaloGeomMapping::BuildFormerGeometry (the "former"/approximate geometry):
towers are projected onto a cylinder of fixed radius at their (eta, phi) bin centers.
The real bin edges live in the CDB (CALO_TOWER_GEOMETRY); here they are approximated by a
uniform grid in eta over [-1.1, 1.1] and in phi over [0, 2pi). If the input CSV carries
explicit ieta/iphi columns, those are used for the tower indices instead of binning eta/phi.

A calo_features.mmap directory (mmap_ninja RaggedMmap, one N x 10 array per event) carries the
real tower positions, so loading one replaces this approximation with the geometry measured
from the file. Columns: E, x, y, z, time, subsystem, ieta, iphi, and two truth columns (unused).
Subsystems: 4 = CEMC, 5 = HCALIN, 6 = HCALOUT; 7 and 8 have no tower indices and are skipped.
"""

import bisect
import csv
import math
import os
import sys

# Layer numbering follows RawClusterBuilderTopo: 0 = IHCal, 1 = OHCal, 2 = EMCal
LAYER_IHCAL = 0
LAYER_OHCAL = 1
LAYER_EMCAL = 2

LAYER_NAMES = {LAYER_IHCAL: "HCALIN", LAYER_OHCAL: "HCALOUT", LAYER_EMCAL: "CEMC"}

_LAYER_ALIASES = {
    "0": LAYER_IHCAL, "hcalin": LAYER_IHCAL, "ihcal": LAYER_IHCAL, "innerhcal": LAYER_IHCAL,
    "1": LAYER_OHCAL, "hcalout": LAYER_OHCAL, "ohcal": LAYER_OHCAL, "outerhcal": LAYER_OHCAL,
    "2": LAYER_EMCAL, "cemc": LAYER_EMCAL, "emcal": LAYER_EMCAL, "emc": LAYER_EMCAL,
}

ETA_MIN = -1.1
ETA_MAX = 1.1


class LayerGeometry:
    """Cylindrical tower grid for one calorimeter layer (RawTowerGeomContainer_Cylinderv1 analogue)."""

    def __init__(self, layer, neta, nphi, radius):
        self.layer = layer
        self.name = LAYER_NAMES[layer]
        self.neta = neta
        self.nphi = nphi
        self.radius = radius
        self.deta = (ETA_MAX - ETA_MIN) / neta
        self.dphi = 2 * math.pi / nphi
        self.set_uniform()

    def set_uniform(self):
        """The approximate uniform grid (used for CSV input)."""
        # BuildFormerGeometry pushes HCal towers out by half a radius
        self.center_radius = self.radius * 1.5 if self.layer in (LAYER_IHCAL, LAYER_OHCAL) else self.radius
        self.phi0 = 0.5 * self.dphi  # center of iphi = 0
        self.eta_centers = None  # measured eta bin centers, if any

    def set_measured(self, eta_centers, phi0, center_radius):
        """Replace the uniform grid with bin centers measured from data."""
        self.eta_centers = list(eta_centers)
        self._eta_edges = [0.5 * (a + b) for a, b in zip(self.eta_centers, self.eta_centers[1:])]
        self.phi0 = phi0
        self.center_radius = center_radius

    def get_etacenter(self, ieta):
        if self.eta_centers is not None:
            return self.eta_centers[ieta]
        return ETA_MIN + (ieta + 0.5) * self.deta

    def get_phicenter(self, iphi):
        return self.phi0 + iphi * self.dphi

    def get_etabin(self, eta):
        # closest bin, as in RawTowerGeomContainer_Cylinderv1::get_etabin
        if self.eta_centers is not None:
            return bisect.bisect_left(self._eta_edges, eta)
        ieta = int(math.floor((eta - ETA_MIN) / self.deta))
        return min(max(ieta, 0), self.neta - 1)

    def get_phibin(self, phi):
        return int(math.floor((phi - self.phi0) / self.dphi + 0.5)) % self.nphi

    def get_center(self, ieta, iphi):
        """Tower center (x, y, z) in cm."""
        r = self.center_radius
        phi = self.get_phicenter(iphi)
        eta = self.get_etacenter(ieta)
        return r * math.cos(phi), r * math.sin(phi), r * math.sinh(eta)

    def get_eta_phi(self, ieta, iphi):
        """RawTowerGeom::get_eta / get_phi computed from the tower center position."""
        x, y, z = self.get_center(ieta, iphi)
        return math.asinh(z / math.hypot(x, y)), math.atan2(y, x)


GEOMETRY = {
    LAYER_EMCAL: LayerGeometry(LAYER_EMCAL, neta=96, nphi=256, radius=93.5),
    LAYER_IHCAL: LayerGeometry(LAYER_IHCAL, neta=24, nphi=64, radius=115.0),
    LAYER_OHCAL: LayerGeometry(LAYER_OHCAL, neta=24, nphi=64, radius=177.423),
}


class Tower:
    __slots__ = ("layer", "ieta", "iphi", "eta", "phi", "energy", "time")

    def __init__(self, layer, ieta, iphi, eta, phi, energy, time):
        self.layer = layer
        self.ieta = ieta
        self.iphi = iphi
        self.eta = eta
        self.phi = phi
        self.energy = energy
        self.time = time


def parse_layer(value):
    key = value.strip().lower().replace("_", "").replace(" ", "")
    if key not in _LAYER_ALIASES:
        raise ValueError(f"unknown calorimeter layer '{value}' (expected one of cemc/emcal, hcalin/ihcal, hcalout/ohcal or 2/0/1)")
    return _LAYER_ALIASES[key]


def _find_column(fieldnames, *candidates):
    lowered = {name.strip().lower(): name for name in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def read_towers(path, default_layer=None):
    """Read towers from a CSV file.

    Required columns: eta, phi, E (aliases: energy, e).
    Optional columns: layer (cemc/hcalin/hcalout or 2/0/1), ieta, iphi, time (aliases: t, tof).
    If layer is absent, default_layer is used (error if that is None too).
    Returns a list of Tower objects; duplicate (layer, ieta, iphi) entries raise ValueError.
    """
    for geom in GEOMETRY.values():
        geom.set_uniform()  # undo any geometry installed by an earlier mmap read
    towers = []
    seen = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV file")
        fields = reader.fieldnames
        c_eta = _find_column(fields, "eta")
        c_phi = _find_column(fields, "phi")
        c_e = _find_column(fields, "e", "energy")
        c_layer = _find_column(fields, "layer", "calo", "detector")
        c_ieta = _find_column(fields, "ieta")
        c_iphi = _find_column(fields, "iphi")
        c_time = _find_column(fields, "time", "t", "tof")
        missing = [n for n, c in (("eta", c_eta), ("phi", c_phi), ("E", c_e)) if c is None]
        if missing:
            raise ValueError(f"{path}: missing required column(s) {missing}; found {fields}")
        if c_layer is None and default_layer is None:
            raise ValueError(f"{path}: a 'layer' column (cemc/hcalin/hcalout) is required")
        if (c_ieta is None) != (c_iphi is None):
            raise ValueError(f"{path}: ieta and iphi columns must be given together")

        for lineno, row in enumerate(reader, start=2):
            layer = parse_layer(row[c_layer]) if c_layer else default_layer
            geom = GEOMETRY[layer]
            eta = float(row[c_eta])
            phi = float(row[c_phi])
            energy = float(row[c_e])
            if c_ieta is not None:
                ieta = int(row[c_ieta])
                iphi = int(row[c_iphi])
                if not (0 <= ieta < geom.neta and 0 <= iphi < geom.nphi):
                    raise ValueError(f"{path}:{lineno}: (ieta, iphi) = ({ieta}, {iphi}) outside {geom.name} grid {geom.neta}x{geom.nphi}")
            else:
                ieta = geom.get_etabin(eta)
                iphi = geom.get_phibin(phi)
            time = 0.0
            if c_time is not None and row[c_time].strip() != "":
                time = float(row[c_time])

            key = (layer, ieta, iphi)
            if key in seen:
                raise ValueError(f"{path}:{lineno}: tower {geom.name} (ieta={ieta}, iphi={iphi}) duplicates line {seen[key]}; "
                                 "check the eta/phi binning or provide ieta/iphi columns")
            seen[key] = lineno
            towers.append(Tower(layer, ieta, iphi, eta, phi, energy, time))
    return towers


_MMAP_SUBSYSTEM_TO_LAYER = {4: LAYER_EMCAL, 5: LAYER_IHCAL, 6: LAYER_OHCAL}
_mmap_cache = {}  # absolute path -> (RaggedMmap, {layer: (eta_centers, phi0, radius)})


def _measure_mmap_geometry(calo, path):
    """Measure each layer's eta centers, phi offset and radius from the events in the file."""
    import numpy as np

    zs = {layer: {} for layer in GEOMETRY}
    first = {}  # layer -> (iphi, x, y)
    for i in range(len(calo)):
        event = np.asarray(calo[i])
        for subsystem, layer in _MMAP_SUBSYSTEM_TO_LAYER.items():
            rows = event[event[:, 5] == subsystem]
            if len(rows):
                zs[layer].update(zip(rows[:, 6].astype(int).tolist(), rows[:, 3].tolist()))
                first.setdefault(layer, (int(rows[0, 7]), float(rows[0, 1]), float(rows[0, 2])))
        if all(len(zs[layer]) == GEOMETRY[layer].neta for layer in GEOMETRY):
            break

    measured = {}
    for layer, geom in GEOMETRY.items():
        missing = sorted(set(range(geom.neta)) - set(zs[layer]))
        if missing:
            raise ValueError(f"{path}: cannot measure {geom.name} geometry: no towers in ieta bins {missing}")
        iphi, x, y = first[layer]
        radius = math.hypot(x, y)
        eta_centers = [math.asinh(zs[layer][ieta] / radius) for ieta in range(geom.neta)]
        phi0 = math.remainder(math.atan2(y, x) - iphi * geom.dphi, 2 * math.pi)
        measured[layer] = (eta_centers, phi0, radius)
    return measured


def read_towers_mmap(path, event):
    """Read one event of a calo_features.mmap directory; applies the geometry measured from it."""
    import numpy as np
    from mmap_ninja.ragged import RaggedMmap

    key = os.path.abspath(path)
    if key not in _mmap_cache:
        calo = RaggedMmap(path)
        _mmap_cache[key] = (calo, _measure_mmap_geometry(calo, path))
    calo, measured = _mmap_cache[key]
    for layer, values in measured.items():
        GEOMETRY[layer].set_measured(*values)

    if not 0 <= event < len(calo):
        raise ValueError(f"{path}: event {event} out of range (file has {len(calo)} events)")
    data = np.asarray(calo[event])
    towers = []
    seen = set()
    for energy, x, y, z, time, subsystem, ieta, iphi, _, _ in data.tolist():
        layer = _MMAP_SUBSYSTEM_TO_LAYER.get(int(subsystem))
        if layer is None:
            continue
        ieta, iphi = int(ieta), int(iphi)
        if (layer, ieta, iphi) in seen:
            raise ValueError(f"{path}: event {event}: duplicate {LAYER_NAMES[layer]} tower (ieta={ieta}, iphi={iphi})")
        seen.add((layer, ieta, iphi))
        towers.append(Tower(layer, ieta, iphi, math.asinh(z / math.hypot(x, y)), math.atan2(y, x), energy, time))
    return towers


def load_towers(path, event=None, default_layer=None):
    """Read towers from a CSV file, or from one event of a calo_features.mmap directory."""
    if os.path.isdir(path):
        if event is None:
            raise ValueError(f"{path}: this is an mmap directory, so an event number is required (--event N)")
        return read_towers_mmap(path, event)
    if event is not None:
        raise ValueError(f"{path}: --event only applies to an mmap directory")
    return read_towers(path, default_layer=default_layer)


def finite_or_none(value):
    """JSON has no NaN/inf; map them to null."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)
