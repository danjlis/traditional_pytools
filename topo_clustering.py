#!/usr/bin/env python3
"""3-D topo-clustering across EMCal, IHCal and OHCal: a Python port of sPHENIX RawClusterBuilderTopo.

Ported from coresoftware/offline/packages/CaloReco/RawClusterBuilderTopo.cc (D. V. Perepelitsa).

Algorithm: seeds with |E| > sigma_seed * noise are grown through neighbours with
|E| > sigma_grow * noise, then one ring of perimeter towers with |E| > sigma_peri * noise is
added. Neighbours are 3x3 in eta/phi within a layer; IHCal and OHCal are adjacent to each
other; an IHCal tower is adjacent to the 4x4 group of EMCal towers in front of it. Clusters
with more than one local maximum are split, with towers on the boundary shared between the
two most energetic adjacent pseudoclusters.

Differences from the C++:
  * Geometry is the "former" approximate geometry with uniform eta/phi bins (no CDB).
  * Tower quality flags (isGood) are not available; every tower in the input is used.
  * Towers missing from the input are treated as non-existent. In the C++ every tower in the
    TowerInfo container exists, so with sigma_peri = 0 the perimeter step also picks up
    zero/low-energy towers; list them in the CSV if you need identical tower lists.
  * Where the C++ indexes a vector with an invalid owner (undefined behaviour), the tower is
    dropped and a warning is printed.
  * Arithmetic is double precision rather than float.

Input: CSV with columns layer, eta, phi, E (+ optional ieta, iphi). Output: JSON.
"""

import argparse
import json
import math
import sys

from calo_common import (GEOMETRY, LAYER_EMCAL, LAYER_IHCAL, LAYER_NAMES, LAYER_OHCAL,
                         finite_or_none, load_towers, warn)

# geometric constants to express IHCal <-> EMCal overlap in eta
EMCAL_ETA_START_GIVEN_IHCAL = [2, 6, 10, 14, 18, 22, 26, 30, 33, 37, 41, 44,
                               48, 52, 55, 59, 63, 66, 70, 74, 78, 82, 86, 90]
EMCAL_ETA_END_GIVEN_IHCAL = [5, 9, 13, 17, 21, 25, 29, 32, 36, 40, 43, 47,
                             51, 54, 58, 62, 65, 69, 73, 77, 81, 85, 89, 93]
IHCAL_ETA_GIVEN_EMCAL = [
    -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2,
    2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
    5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8,
    8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11,
    12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 15,
    15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 18, 18,
    18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21,
    21, 21, 22, 22, 22, 22, 23, 23, 23, 23, -1, -1]

STATUS_MISSING = -2
STATUS_UNKNOWN = -1


def calculate_dR(eta1, eta2, phi1, phi2):
    deta = eta1 - eta2
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    return math.sqrt(deta * deta + dphi * dphi)


class TopoClusterer:
    def __init__(self, noise_ihcal=0.0025, noise_ohcal=0.006, noise_emcal=0.03,
                 sigma_seed=4.0, sigma_grow=2.0, sigma_peri=0.0,
                 local_max_min_e_ihcal=1.0, local_max_min_e_ohcal=1.0, local_max_min_e_emcal=1.0,
                 r_shower=0.025, min_cluster_e=0.0, allow_corner_neighbor=True, use_abs_e=True,
                 enable_hcal=True, enable_emcal=True, do_split=True, emcal_phi_offset=None):
        self.noise = [noise_ihcal, noise_ohcal, noise_emcal]
        self.sigma_seed = sigma_seed
        self.sigma_grow = sigma_grow
        self.sigma_peri = sigma_peri
        self.local_max_min_e = [local_max_min_e_ihcal, local_max_min_e_ohcal, local_max_min_e_emcal]
        self.r_shower = r_shower
        self.min_cluster_e = min_cluster_e
        self.allow_corner_neighbor = allow_corner_neighbor
        self.use_abs_e = use_abs_e
        self.enable_hcal = enable_hcal
        self.enable_emcal = enable_emcal
        self.do_split = do_split

        self.geom = GEOMETRY
        self.emcal_neta = GEOMETRY[LAYER_EMCAL].neta
        self.emcal_nphi = GEOMETRY[LAYER_EMCAL].nphi
        self.hcal_neta = GEOMETRY[LAYER_OHCAL].neta
        self.hcal_nphi = GEOMETRY[LAYER_OHCAL].nphi
        # First EMCal iphi in front of IHCal iphi = 0. The C++ hard-codes 5; with geometry
        # measured from an mmap file, use the EMCal towers actually inside that IHCal tower.
        gem, gih = GEOMETRY[LAYER_EMCAL], GEOMETRY[LAYER_IHCAL]
        # emcal_phi_offset=None picks it automatically; pass 5 to force the C++ value
        if emcal_phi_offset is not None:
            self.emcal_phi_offset = emcal_phi_offset
        elif gem.eta_centers is None:
            self.emcal_phi_offset = 5
        else:
            low_edge = gih.phi0 - 0.5 * gih.dphi
            self.emcal_phi_offset = math.ceil((low_edge - gem.phi0) / gem.dphi) % gem.nphi
        self.n_emcal = self.emcal_neta * self.emcal_nphi
        n_ids = 2 * self.n_emcal
        self.status = [STATUS_MISSING] * n_ids
        self.energy = [0.0] * n_ids
        self.n_dropped = 0
        self.input_towers = {}

    # ------------------------------------------------------------------ tower ID scheme

    def get_id(self, ilayer, ieta, iphi):
        if ilayer < 2:
            return ilayer * self.hcal_neta * self.hcal_nphi + ieta * self.hcal_nphi + iphi
        return self.n_emcal + ieta * self.emcal_nphi + iphi

    def get_ilayer(self, tid):
        if tid < self.n_emcal:
            return tid // (self.hcal_neta * self.hcal_nphi)
        return 2

    def get_ieta(self, tid):
        if tid < self.n_emcal:
            return (tid % (self.hcal_neta * self.hcal_nphi)) // self.hcal_nphi
        return (tid - self.n_emcal) // self.emcal_nphi

    def get_iphi(self, tid):
        if tid < self.n_emcal:
            return tid % self.hcal_nphi
        return (tid - self.n_emcal) % self.emcal_nphi

    def first_matching_emcal_phi_from_ihcal(self, ihcal_phi):
        return (4 * ihcal_phi + self.emcal_phi_offset) % self.emcal_nphi

    def matching_hcal_phi_from_emcal(self, emcal_phi):
        return ((emcal_phi + self.emcal_nphi - self.emcal_phi_offset) // 4) % self.hcal_nphi

    def get_adjacent_towers(self, tid):
        layer = self.get_ilayer(tid)
        ieta = self.get_ieta(tid)
        iphi = self.get_iphi(tid)
        adjacent = []

        # IHCal and OHCal: 3x3 in this layer and the other HCal layer
        if layer in (LAYER_IHCAL, LAYER_OHCAL):
            for dlayer in (0, 1):
                for deta in (-1, 0, 1):
                    for dphi in (-1, 0, 1):
                        if dlayer == 0 and deta == 0 and dphi == 0:
                            continue
                        test_eta = ieta + deta
                        if test_eta < 0 or test_eta >= self.hcal_neta:
                            continue
                        test_layer = (layer + dlayer) % 2
                        test_phi = (iphi + dphi + self.hcal_nphi) % self.hcal_nphi
                        if not self.allow_corner_neighbor and dlayer == 1 and abs(dphi) == 1 and abs(deta) == 1:
                            continue
                        adjacent.append(self.get_id(test_layer, test_eta, test_phi))

        # IHCal: also the 4x4 group of EMCal towers in front of it
        if layer == LAYER_IHCAL and self.enable_emcal:
            phi_start = self.first_matching_emcal_phi_from_ihcal(iphi)
            for new_eta in range(EMCAL_ETA_START_GIVEN_IHCAL[ieta], EMCAL_ETA_END_GIVEN_IHCAL[ieta] + 1):
                for dphi in range(4):
                    new_phi = (phi_start + dphi + self.emcal_nphi) % self.emcal_nphi
                    adjacent.append(self.get_id(LAYER_EMCAL, new_eta, new_phi))

        # EMCal: 3x3 EMCal neighbours and (usually) one IHCal tower
        if layer == LAYER_EMCAL:
            for deta in (-1, 0, 1):
                for dphi in (-1, 0, 1):
                    if deta == 0 and dphi == 0:
                        continue
                    test_eta = ieta + deta
                    if test_eta < 0 or test_eta >= self.emcal_neta:
                        continue
                    test_phi = (iphi + dphi + self.emcal_nphi) % self.emcal_nphi
                    adjacent.append(self.get_id(layer, test_eta, test_phi))
            if self.enable_hcal:
                hcal_eta = IHCAL_ETA_GIVEN_EMCAL[ieta]
                if hcal_eta >= 0:
                    adjacent.append(self.get_id(LAYER_IHCAL, hcal_eta, self.matching_hcal_phi_from_emcal(iphi)))

        return adjacent

    def etacenter(self, tid):
        return self.geom[self.get_ilayer(tid)].get_bin_etacenter(self.get_ieta(tid))

    def phicenter(self, tid):
        return self.geom[self.get_ilayer(tid)].get_bin_phicenter(self.get_iphi(tid))

    # ------------------------------------------------------------------ main algorithm

    def run(self, towers):
        list_of_seeds = []
        layer_order = {LAYER_EMCAL: 0, LAYER_IHCAL: 1, LAYER_OHCAL: 2}
        for t in sorted(towers, key=lambda t: (layer_order[t.layer], t.ieta, t.iphi)):
            if t.layer == LAYER_EMCAL and not self.enable_emcal:
                continue
            if t.layer != LAYER_EMCAL and not self.enable_hcal:
                continue
            # if not using abs E, short circuit all negative towers right here
            if not self.use_abs_e and t.energy < 1.0e-10:
                continue
            tid = self.get_id(t.layer, t.ieta, t.iphi)
            self.status[tid] = STATUS_UNKNOWN
            self.energy[tid] = t.energy
            if abs(t.energy) >= self.sigma_seed * self.noise[t.layer]:
                list_of_seeds.append((tid, t.energy))

        list_of_seeds.sort(key=lambda s: -s[1])

        all_cluster_towers = []
        cluster_index = 0
        for seed_id, _ in list_of_seeds:
            # already claimed by some other seed during its growth
            if self.status[seed_id] > -1:
                continue

            self.status[seed_id] = cluster_index
            cluster_tower_ids = [seed_id]
            grow_tower_ids = [seed_id]

            # growth stage: iteratively add > sigma_grow neighbours
            ig = 0
            while ig < len(grow_tower_ids):
                grow_id = grow_tower_ids[ig]
                ig += 1
                for adj in self.get_adjacent_towers(grow_id):
                    st = self.status[adj]
                    if st == STATUS_MISSING or st == cluster_index:
                        continue
                    if abs(self.energy[adj]) < self.sigma_grow * self.noise[self.get_ilayer(adj)]:
                        continue
                    if st > -1:
                        continue  # owned by somebody else
                    grow_tower_ids.append(adj)
                    cluster_tower_ids.append(adj)
                    self.status[adj] = cluster_index

            # perimeter stage: one ring of > sigma_peri neighbours around the core towers
            n_core = len(cluster_tower_ids)
            for ic in range(n_core):
                for adj in self.get_adjacent_towers(cluster_tower_ids[ic]):
                    st = self.status[adj]
                    if st == STATUS_MISSING or st > -1:
                        continue
                    if abs(self.energy[adj]) < self.sigma_peri * self.noise[self.get_ilayer(adj)]:
                        continue
                    cluster_tower_ids.append(adj)
                    self.status[adj] = cluster_index

            all_cluster_towers.append(cluster_tower_ids)
            cluster_index += 1

        clusters = []
        for cl, original_towers in enumerate(all_cluster_towers):
            if not self.do_split:
                clusters.extend(self.export_single_cluster(original_towers))
                continue
            clusters.extend(self.split_cluster(cl, original_towers))
        return clusters, len(all_cluster_towers)

    def split_cluster(self, cl, original_towers):
        E = self.energy
        local_maxima = []
        for tid in original_towers:
            if E[tid] < self.local_max_min_e[self.get_ilayer(tid)]:
                continue
            neighbors_in_cluster = 0
            has_higher_neighbor = False
            for adj in self.get_adjacent_towers(tid):
                if self.status[adj] != cl:
                    continue
                neighbors_in_cluster += 1
                if E[adj] > E[tid]:
                    has_higher_neighbor = True
                    break
            if has_higher_neighbor or neighbors_in_cluster < 4:
                continue
            local_maxima.append((tid, E[tid]))

        # remove I/OHCal local maxima overlapping an EMCal local maximum
        n = 0
        while n < len(local_maxima):
            lm = local_maxima[n]
            if self.get_ilayer(lm[0]) == LAYER_EMCAL:
                n += 1
                continue
            this_phi = self.phicenter(lm[0])
            if this_phi > math.pi:
                this_phi -= 2 * math.pi
            this_eta = self.etacenter(lm[0])
            has_em_overlap = False
            for n2, lm2 in enumerate(local_maxima):
                if n == n2 or self.get_ilayer(lm2[0]) != LAYER_EMCAL:
                    continue
                # (the C++ shifts this_phi instead of this_phi2 here; dR wraps dphi so it is equivalent)
                this_phi2 = self.phicenter(lm2[0])
                this_eta2 = self.etacenter(lm2[0])
                if calculate_dR(this_eta, this_eta2, this_phi, this_phi2) < 0.15:
                    has_em_overlap = True
                    break
            if has_em_overlap:
                del local_maxima[n]
            else:
                n += 1

        if len(local_maxima) <= 1:
            return self.export_single_cluster(original_towers)

        # ownership: (-1,-1) unseen; (k,-1) owned by pseudocluster k; (-3,-1) shared (pending);
        # (k1,k2) shared between k1 and k2
        ownership = {tid: [-1, -1] for tid in original_towers}
        seed_list = []
        shared_list = []

        local_maxima.sort(key=lambda s: -s[1])
        n_lm = len(local_maxima)
        neighbor_list = []
        for s, (tid, _) in enumerate(local_maxima):
            ownership[tid] = [s, -1]
            neighbor_list.append(tid)

        first_pass = True
        while True:
            new_ownerships = []
            for neighbor_id in neighbor_list:
                if first_pass:
                    new_ownerships.append(ownership[neighbor_id][0])
                    continue
                adjacency = [False] * n_lm
                for adj in self.get_adjacent_towers(neighbor_id):
                    if self.status[adj] != cl:
                        continue
                    owner = ownership[adj][0]
                    if owner > -1:
                        adjacency[owner] = True
                n_adjacent = 0
                last_adjacent = -1
                for s in range(n_lm):
                    if adjacency[s]:
                        last_adjacent = s
                        n_adjacent += 1
                if n_adjacent == 0:
                    warn("topo splitting: neighbor tower is adjacent to no pseudocluster")
                    new_ownerships.append(9999)
                elif n_adjacent == 1:
                    new_ownerships.append(last_adjacent)
                else:
                    new_ownerships.append(-3)

            # transfer neighbor list to seed list or shared list
            for neighbor_id, own in zip(neighbor_list, new_ownerships):
                if own > -1:
                    ownership[neighbor_id] = [own, -1]
                    seed_list.append(neighbor_id)
                if own == -3:
                    ownership[neighbor_id] = [-3, -1]
                    shared_list.append(neighbor_id)

            # new neighbor list from the unseen neighbours of newly owned towers
            new_neighbors = set()
            for neighbor_id, own in zip(neighbor_list, new_ownerships):
                if own > -1:
                    for adj in self.get_adjacent_towers(neighbor_id):
                        if self.status[adj] != cl:
                            continue
                        if ownership[adj][0] == -1:
                            new_neighbors.add(adj)
            neighbor_list = sorted(new_neighbors)
            first_pass = False
            if not neighbor_list:
                break

        # pseudocluster energies and (unweighted) positions
        sum_eta = [0.0] * n_lm
        sum_phi = [0.0] * n_lm
        sum_e = [0.0] * n_lm
        ntower = [0] * n_lm
        for tid in original_towers:
            owner = ownership[tid][0]
            if 0 <= owner < n_lm:
                sum_e[owner] += E[tid]
                sum_eta[owner] += self.etacenter(tid)
                sum_phi[owner] += self.phicenter(tid)
                ntower[owner] += 1
        pc_eta = [sum_eta[i] / ntower[i] if ntower[i] else float("nan") for i in range(n_lm)]
        pc_phi = [sum_phi[i] / ntower[i] if ntower[i] else float("nan") for i in range(n_lm)]

        # resolve shared towers: owned by the two most energetic adjacent pseudoclusters
        ish = 0
        while ish < len(shared_list):
            shared_id = shared_list[ish]
            ish += 1
            adjacency = [False] * n_lm
            for adj in self.get_adjacent_towers(shared_id):
                if self.status[adj] != cl:
                    continue
                first, second = ownership[adj]
                if 0 <= first < n_lm:
                    adjacency[first] = True
                if 0 <= second < n_lm:
                    adjacency[second] = True  # inherit adjacency from shared towers
                if first == -1:
                    shared_list.append(adj)
                    ownership[adj] = [-3, -1]

            highest = -1
            second_highest = -1
            highest_e = -999.0
            second_highest_e = -999.0
            for k in range(n_lm):
                if not adjacency[k]:
                    continue
                if sum_e[k] > highest_e:
                    second_highest_e = highest_e
                    second_highest = highest
                    highest_e = sum_e[k]
                    highest = k
                elif sum_e[k] > second_highest_e:
                    second_highest_e = sum_e[k]
                    second_highest = k
            ownership[shared_id] = [highest, second_highest]

        return self.export_clusters(original_towers, ownership, n_lm, sum_e, pc_eta, pc_phi)

    def export_single_cluster(self, original_towers):
        ownership = {tid: [0, -1] for tid in original_towers}
        return self.export_clusters(original_towers, ownership, 1, [], [], [])

    def export_clusters(self, original_towers, ownership, n_clusters, pc_sum_e, pc_eta, pc_phi):
        E = self.energy
        cl_towers = [[] for _ in range(n_clusters)]
        cl_e = [0.0] * n_clusters
        cl_abs_e = [0.0] * n_clusters
        cl_x = [0.0] * n_clusters
        cl_y = [0.0] * n_clusters
        cl_z = [0.0] * n_clusters
        cl_layer_e = [[0.0, 0.0, 0.0] for _ in range(n_clusters)]

        def add(k, tid, frac, center):
            e = E[tid]
            layer = self.get_ilayer(tid)
            cl_towers[k].append((tid, e * frac))
            cl_e[k] += e * frac
            cl_abs_e[k] += abs(e) * frac
            # position mean uses absolute energy as weights
            cl_x[k] += abs(e) * center[0] * frac
            cl_y[k] += abs(e) * center[1] * frac
            cl_z[k] += abs(e) * center[2] * frac
            cl_layer_e[k][layer] += e * frac

        for tid in original_towers:
            first, second = ownership[tid]
            layer = self.get_ilayer(tid)
            geom = self.geom[layer]
            center = geom.get_center(self.get_ieta(tid), self.get_iphi(tid))
            if not (0 <= first < n_clusters) or second >= n_clusters:
                # undefined behaviour in the C++ (vector indexed with an invalid owner)
                self.n_dropped += 1
                warn(f"topo export: dropping tower {LAYER_NAMES[layer]} (ieta={self.get_ieta(tid)}, "
                     f"iphi={self.get_iphi(tid)}) with invalid ownership ({first}, {second})")
                continue
            if second == -1:
                add(first, tid, 1.0, center)
            else:
                # shared between two clusters: energy-sharing fraction
                t_eta, t_phi = geom.get_eta_phi(self.get_ieta(tid), self.get_iphi(tid))
                dr1 = calculate_dR(t_eta, pc_eta[first], t_phi, pc_phi[first]) / self.r_shower
                dr2 = calculate_dR(t_eta, pc_eta[second], t_phi, pc_phi[second]) / self.r_shower
                r = math.exp(dr1 - dr2)
                frac1 = abs(pc_sum_e[first]) / (abs(pc_sum_e[first]) + r * abs(pc_sum_e[second]))
                add(first, tid, frac1, center)
                add(second, tid, 1.0 - frac1, center)

        out = []
        for k in range(n_clusters):
            if cl_abs_e[k] < self.min_cluster_e:
                continue
            if cl_abs_e[k] > 0:
                mx = cl_x[k] / cl_abs_e[k]
                my = cl_y[k] / cl_abs_e[k]
                mz = cl_z[k] / cl_abs_e[k]
            else:
                mx = my = mz = float("nan")
            r = math.sqrt(mx * mx + my * my)
            eta = math.asinh(mz / r) if r > 0 else float("nan")
            towers_out = []
            for tid, e in cl_towers[k]:
                layer = self.get_ilayer(tid)
                ieta = self.get_ieta(tid)
                iphi = self.get_iphi(tid)
                t = self.input_towers.get((layer, ieta, iphi))
                geom = self.geom[layer]
                towers_out.append({
                    "layer": LAYER_NAMES[layer],
                    "ieta": ieta,
                    "iphi": iphi,
                    "eta": t.eta if t else geom.get_etacenter(ieta),
                    "phi": t.phi if t else geom.get_phicenter(iphi),
                    "energy": e,
                })
            out.append({
                "energy": cl_e[k],
                "abs_energy": cl_abs_e[k],
                "eta": finite_or_none(eta),
                "phi": finite_or_none(math.atan2(my, mx)),
                "et": finite_or_none(cl_e[k] / math.cosh(eta)),
                "x": finite_or_none(mx),
                "y": finite_or_none(my),
                "z": finite_or_none(mz),
                "r": finite_or_none(r),
                "energy_emcal": cl_layer_e[k][LAYER_EMCAL],
                "energy_ihcal": cl_layer_e[k][LAYER_IHCAL],
                "energy_ohcal": cl_layer_e[k][LAYER_OHCAL],
                "ntowers": len(towers_out),
                "towers": towers_out,
            })
        return out


def run_topo_clustering(towers, **params):
    """Topo-cluster towers from all three calorimeter layers.

    towers: iterable of calo_common.Tower. params: keyword arguments of TopoClusterer.
    Returns a JSON-serialisable dict.
    """
    towers = list(towers)
    tc = TopoClusterer(**params)
    tc.input_towers = {(t.layer, t.ieta, t.iphi): t for t in towers}
    clusters, n_initial = tc.run(towers)
    for i, c in enumerate(clusters):
        c["id"] = i
    n_by_layer = {LAYER_NAMES[l]: sum(1 for t in towers if t.layer == l) for l in LAYER_NAMES}
    return {
        "algorithm": "RawClusterBuilderTopo",
        "parameters": {
            "noise_emcal": tc.noise[LAYER_EMCAL],
            "noise_ihcal": tc.noise[LAYER_IHCAL],
            "noise_ohcal": tc.noise[LAYER_OHCAL],
            "sigma_seed": tc.sigma_seed,
            "sigma_grow": tc.sigma_grow,
            "sigma_peri": tc.sigma_peri,
            "emcal_phi_offset": tc.emcal_phi_offset,
            "local_max_min_e_emcal": tc.local_max_min_e[LAYER_EMCAL],
            "local_max_min_e_ihcal": tc.local_max_min_e[LAYER_IHCAL],
            "local_max_min_e_ohcal": tc.local_max_min_e[LAYER_OHCAL],
            "r_shower": tc.r_shower,
            "min_cluster_e": tc.min_cluster_e,
            "allow_corner_neighbor": tc.allow_corner_neighbor,
            "use_abs_e": tc.use_abs_e,
            "enable_emcal": tc.enable_emcal,
            "enable_hcal": tc.enable_hcal,
            "do_split": tc.do_split,
            "geometry": "former (approximate, uniform eta/phi bins)",
        },
        "n_input_towers": n_by_layer,
        "n_topoclusters_before_split": n_initial,
        "n_dropped_towers": tc.n_dropped,  # towers with invalid ownership (C++ UB), see warnings
        "n_clusters": len(clusters),
        "clusters": clusters,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="calo_features.mmap directory (with --event), or CSV file with columns layer, eta, phi, E (optional: ieta, iphi)")
    p.add_argument("--event", type=int, help="event number in the mmap directory")
    p.add_argument("-o", "--output", help="output JSON file (default: stdout)")
    p.add_argument("--noise-emcal", type=float, default=0.03, help="EMCal tower noise [GeV] (default 0.03)")
    p.add_argument("--noise-ihcal", type=float, default=0.0025, help="IHCal tower noise [GeV] (default 0.0025)")
    p.add_argument("--noise-ohcal", type=float, default=0.006, help="OHCal tower noise [GeV] (default 0.006)")
    p.add_argument("--sigma-seed", type=float, default=4.0, help="seed significance (default 4)")
    p.add_argument("--sigma-grow", type=float, default=2.0, help="growth significance (default 2)")
    p.add_argument("--sigma-peri", type=float, default=0.0, help="perimeter significance (default 0)")
    p.add_argument("--local-max-min-e-emcal", type=float, default=1.0, help="min EMCal local-maximum energy [GeV] (default 1)")
    p.add_argument("--local-max-min-e-ihcal", type=float, default=1.0, help="min IHCal local-maximum energy [GeV] (default 1)")
    p.add_argument("--local-max-min-e-ohcal", type=float, default=1.0, help="min OHCal local-maximum energy [GeV] (default 1)")
    p.add_argument("--r-shower", type=float, default=0.025, help="shower radius for shared-tower energy sharing (default 0.025)")
    p.add_argument("--min-cluster-e", type=float, default=0.0, help="minimum cluster |E| saved [GeV] (default 0)")
    p.add_argument("--no-corner-neighbor", action="store_true", help="disallow diagonal neighbours across HCal layers")
    p.add_argument("--no-abs-e", action="store_true", help="ignore negative-energy towers")
    p.add_argument("--no-hcal", action="store_true", help="disable the HCal layers")
    p.add_argument("--no-emcal", action="store_true", help="disable the EMCal layer")
    p.add_argument("--no-split", action="store_true", help="do not split clusters at local maxima")
    p.add_argument("--no-towers", action="store_true", help="omit per-cluster tower lists from the output")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (negative for compact)")
    args = p.parse_args(argv)

    towers = load_towers(args.input, event=args.event)
    result = run_topo_clustering(
        towers,
        noise_ihcal=args.noise_ihcal, noise_ohcal=args.noise_ohcal, noise_emcal=args.noise_emcal,
        sigma_seed=args.sigma_seed, sigma_grow=args.sigma_grow, sigma_peri=args.sigma_peri,
        local_max_min_e_ihcal=args.local_max_min_e_ihcal,
        local_max_min_e_ohcal=args.local_max_min_e_ohcal,
        local_max_min_e_emcal=args.local_max_min_e_emcal,
        r_shower=args.r_shower, min_cluster_e=args.min_cluster_e,
        allow_corner_neighbor=not args.no_corner_neighbor, use_abs_e=not args.no_abs_e,
        enable_hcal=not args.no_hcal, enable_emcal=not args.no_emcal, do_split=not args.no_split,
    )
    if args.event is not None:
        result["event"] = args.event
    if args.no_towers:
        for c in result["clusters"]:
            del c["towers"]

    text = json.dumps(result, indent=args.indent if args.indent >= 0 else None, allow_nan=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
