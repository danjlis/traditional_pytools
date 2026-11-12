#!/usr/bin/env python3
"""EMCal (CEMC) template clustering: a Python port of sPHENIX RawClusterBuilderTemplate.

Ported from coresoftware/offline/packages/CaloReco:
  RawClusterBuilderTemplate.cc  - driver (tower selection, cluster -> RawCluster properties)
  BEmcRec.cc                    - FindClusters (Lednev/GAMS contiguous-tower search), Momenta,
                                  Tower2Global, PredictEnergyParam
  BEmcRecCEMC.cc                - CorrectPosition, CorrectECore, CorrectShowerDepth
  BEmcCluster.cc                - GetSubClusters (local-maximum splitting), GetECore, ...

Differences from the C++ (no CDB / detailed geometry available here):
  * The "former" approximate geometry is used (m_UseDetailedGeometry = false code paths).
  * No shower profile is loaded (LoadProfile), so prob/chi2 are not computed (C++ returns
    prob = -1, chi2 = 0 in that case; reported as null here).
  * Mechanical incidence angles (BEmcRecCEMC::calculateIncidence) need the detailed
    geometry rotations and are not computed.
  * Tower quality flags (isGood) are not available; every tower above threshold is used.
  * Arithmetic is double precision rather than float.

Input: CSV with columns eta, phi, E (+ optional ieta, iphi, layer, time). If a layer column is
present only CEMC towers are used. Output: JSON with the clusters and their properties.
"""

import argparse
import json
import math
import sys

from calo_common import GEOMETRY, LAYER_EMCAL, finite_or_none, load_towers, warn

# EmcCluster static members
MAX_N_PEAKS = 1000
PEAK_ITER = 6
SUBCLUSTER_EMIN = 0.002

# BEmcRecCEMC::angles - average tower angle wrt the transverse plane per eta bin
# (used by CorrectShowerDepth with the approximate geometry)
CEMC_TOWER_ANGLES = [
    2.382132067280038, 2.3708136440088796, 2.355383841518272, 2.3434466029558525, 2.32455480538339, 2.3135158793606005,
    2.2999027706894153, 2.289397309387001, 2.2727994372634956, 2.259380682790801, 2.2392521417328277, 2.226335700118477,
    2.2095791811782735, 2.1935361258633024, 2.1754991989110875, 2.160931303742618, 2.142623616095435, 2.1259546238847804,
    2.1054808114318946, 2.0900705081463244, 2.0670824189635053, 2.050851798834259, 2.026500612990249, 2.005259135338716,
    1.9868346921870874, 1.966226760986188, 1.9455349050274326, 1.924706496569077, 1.9017832210973762, 1.8805537731028699,
    1.8568682630187272, 1.8337068107715466, 1.8112862676694754, 1.7884337950298892, 1.76612732335177, 1.7439109866141163,
    1.7137229887165837, 1.6887545076082588, 1.6601011126718555, 1.633546284539203, 1.6126575096324751, 1.594318479859694,
    1.5699679790263024, 1.5683722241543814, 1.570109507416994, 1.5686069317665148, 1.5706670797900908, 1.5655401209498148,
    1.5760525326399784, 1.5709255737997023, 1.5729857218232786, 1.5714831461727994, 1.573220429435412, 1.5716246745634908,
    1.5472741737300992, 1.528935143957318, 1.50804636905059, 1.4814915409179379, 1.4528381459815345, 1.4278696648732097,
    1.3976816669756769, 1.375465330238023, 1.353158858559904, 1.3303063859203177, 1.3078858428182465, 1.284724390571066,
    1.2610388804869233, 1.2398094324924172, 1.2168861570207163, 1.1960577485623607, 1.1753658926036052, 1.154757961402706,
    1.1363335182510774, 1.1150920405995441, 1.0907408547555344, 1.074510234626288, 1.051522145443469, 1.0361118421578988,
    1.015638029705013, 0.9989690374943581, 0.9806613498471752, 0.9660934546787058, 0.948056527726491, 0.9320134724115198,
    0.9152569534713161, 0.9023405118569655, 0.8822119707989924, 0.8687932163262975, 0.852195344202792, 0.8416898829003777,
    0.8280767742291925, 0.817037848206403, 0.798146050633941, 0.7862088120715213, 0.7707790095809137, 0.7594605863097555,
]


def lowint(x):
    """EmcCluster::lowint"""
    return int(x - 1) if x < 0.0 else int(x)


class EmcModule:
    __slots__ = ("ich", "amp", "tof")

    def __init__(self, ich=0, amp=0.0, tof=0.0):
        self.ich = ich
        self.amp = amp
        self.tof = tof

    def copy(self):
        return EmcModule(self.ich, self.amp, self.tof)


class BEmcRecCEMC:
    """BEmcRec + BEmcRecCEMC with the approximate (former) geometry and no shower profile."""

    def __init__(self, geom, vz=0.0, peak_thresh=0.200,
                 use_correct_position=True, use_correct_shower_depth=True):
        # ix runs along phi (index2), iy along eta (index1); ich = iy * nx + ix
        self.nx = geom.nphi
        self.ny = geom.neta
        self.cyl = True
        self.vz = vz
        self.peak_thresh = peak_thresh
        self.use_correct_position = use_correct_position
        self.use_correct_shower_depth = use_correct_shower_depth
        self.tower_geom = {}
        for iy in range(self.ny):
            for ix in range(self.nx):
                x, y, z = geom.get_center(iy, ix)
                # [Xcenter, Ycenter, Zcenter, dX[0], dY[0], dZ[0], dX[1], dY[1], dZ[1]]
                self.tower_geom[iy * self.nx + ix] = [x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.complete_tower_geometry()

    # ------------------------------------------------------------------ geometry

    def get_tower_geometry(self, ix, iy):
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return None
        return self.tower_geom.get(iy * self.nx + ix)

    def complete_tower_geometry(self):
        """BEmcRec::CompleteTowerGeometry - tower axis vectors from neighbouring centers."""
        idx = [0, 1, 0, -1, -1, 1, 1, -1]
        idy = [-1, 0, 1, 0, -1, -1, 1, 1]
        nb = len(idx)
        completed = {}
        for ich, g0 in self.tower_geom.items():
            ix = ich % self.nx
            iy = ich // self.nx
            inx = 0
            while inx < nb and (idx[inx] == 0 or self.get_tower_geometry(ix + idx[inx], iy + idy[inx]) is None):
                inx += 1
            iny = 0
            while iny < nb and (idy[iny] == 0 or self.get_tower_geometry(ix + idx[iny], iy + idy[iny]) is None):
                iny += 1
            if inx >= nb or iny >= nb:
                raise RuntimeError(f"BEmcRec::CompleteTowerGeometry: no neighbour for (ix,iy)=({ix},{iy})")
            gx = self.get_tower_geometry(ix + idx[inx], iy + idy[inx])
            gy = self.get_tower_geometry(ix + idx[iny], iy + idy[iny])
            g = list(g0)
            g[3] = (gx[0] - g0[0]) / idx[inx]
            g[4] = (gx[1] - g0[1]) / idx[inx]
            g[5] = (gx[2] - g0[2]) / idx[inx]
            g[6] = (gy[0] - g0[0]) / idy[iny]
            g[7] = (gy[1] - g0[1]) / idy[iny]
            g[8] = (gy[2] - g0[2]) / idy[iny]
            completed[ich] = g
        self.tower_geom = completed

    def i_tower_dist(self, ix1, ix2):
        idist = ix2 - ix1
        if self.cyl:
            idistr = self.nx - abs(idist)
            if idistr < abs(idist):
                idist = idistr if idist < 0 else -idistr
        return idist

    def f_tower_dist(self, x1, x2):
        dist = x2 - x1
        if self.cyl:
            distr = self.nx - abs(dist)
            if distr < abs(dist):
                dist = distr if dist < 0 else -distr
        return dist

    # ------------------------------------------------------------------ clustering

    def find_clusters(self, modules):
        """BEmcRec::FindClusters - groups of towers with a common edge (Lednev's GAMS algorithm).

        Returns a list of hit lists.
        """
        nx = self.nx
        nhit = len(modules)
        if nhit <= 0:
            return []
        if nhit == 1:
            return [[m.copy() for m in modules]]

        vhit = sorted((m.copy() for m in modules), key=lambda m: m.ich)
        len_cl = []
        ncl = 0
        nxt = 0
        ia = 0
        for ich in range(1, nhit + 1):
            if ich < nhit:
                ia = vhit[ich].ich

            # new subcluster: gap in channel number, end of list, or start of a new row
            if (ia - vhit[ich - 1].ich > 1) or (ich >= nhit) or (ia % nx == 0):
                ib = nxt
                ie = ich - 1
                nxt = ich
                ncl += 1
                if len(len_cl) < ncl:
                    len_cl.extend([0] * (ncl - len(len_cl)))
                len_cl[ncl - 1] = nxt - ib
                if ncl > 1:
                    # glue the subclusters with a common edge
                    iab = vhit[ib].ich  # the last subcluster begin
                    iae = vhit[ie].ich  # the last subcluster end
                    last = ib - 1       # the pre-last subcluster end
                    for icl in range(ncl - 2, -1, -1):
                        leng = len_cl[icl]
                        if iab - vhit[last].ich > nx:
                            break  # goto new_ich
                        for ichc in range(last, last - leng, -1):
                            if ((vhit[ichc].ich + nx <= iae and vhit[ichc].ich + nx >= iab)
                                    or (self.cyl and (iae % nx == nx - 1) and (iae - vhit[ichc].ich == nx - 1))):
                                # move subcluster icl next to the last subcluster
                                moved = vhit[last + 1 - leng:last + 1]
                                between = vhit[last + 1:ib]
                                vhit[last + 1 - leng:ib] = between + moved
                                for i in range(icl, ncl - 2):
                                    len_cl[i] = len_cl[i + 1]
                                ib -= leng
                                len_cl[ncl - 2] = len_cl[ncl - 1] + leng
                                ncl -= 1
                                break  # goto new_icl
                        last = last - leng

        clusters = []
        ib = 0
        for icl in range(ncl):
            leng = len_cl[icl]
            clusters.append(vhit[ib:ib + leng])
            ib += leng
        return clusters

    def momenta(self, hits, thresh=0.0):
        """BEmcRec::Momenta - returns (e, x, y, xx, yy, yx) in tower units."""
        if not hits:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        emax = 0.0
        ichmax = 0
        for h in hits:
            if h.amp > emax:
                emax = h.amp
                ichmax = h.ich
        if emax <= 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        iymax = ichmax // self.nx
        ixmax = ichmax - iymax * self.nx
        x = y = e = xx = yy = yx = 0.0
        for h in hits:
            a = h.amp
            if a > thresh:
                iy = h.ich // self.nx
                ix = h.ich - iy * self.nx
                idx = self.i_tower_dist(ixmax, ix)
                idy = iy - iymax
                e += a
                x += idx * a
                y += idy * a
                xx += a * idx * idx
                yy += a * idy * idy
                yx += a * idx * idy

        if e <= 0:
            return e, 0.0, 0.0, 0.0, 0.0, 0.0
        x /= e
        y /= e
        xx = xx / e - x * x
        yy = yy / e - y * y
        yx = yx / e - y * x
        x += ixmax
        y += iymax
        while x < -0.5:
            x += self.nx
        while x >= self.nx - 0.5:
            x -= self.nx
        return e, x, y, xx, yy, yx

    def predict_energy(self, en, xcg, ycg, ix, iy):
        """BEmcRec::PredictEnergy (no profile loaded -> parametrised shape)."""
        dx = abs(self.f_tower_dist(float(ix), xcg))
        dy = ycg - iy
        return self.predict_energy_param(en, dx, dy)

    @staticmethod
    def predict_energy_param(en, xc, yc):
        """BEmcRec::PredictEnergyParam - fraction of shower energy in a tower at distance (xc, yc)."""
        p1, p2, p3, p4 = 0.549, 0.304, 0.389, 0.326
        dx = abs(xc)
        dy = abs(yc)
        r2 = dx * dx + dy * dy
        r1 = math.sqrt(r2)
        r3 = r2 * r1
        return p1 * math.exp(-r3 / p2) + p3 * math.exp(-r1 / p4)

    # ------------------------------------------------------------------ corrections

    def correct_ecore(self, ecore):
        """BEmcRecCEMC::CorrectECore"""
        return ecore / 0.950

    def correct_position(self, energy, x, y):
        """BEmcRecCEMC::CorrectPosition (approximate geometry branch)."""
        if not self.use_correct_position:
            return x, y
        if energy < 0.01:
            return x, y

        bx = 0.15
        by = 0.15

        x0 = x
        ix0 = lowint(x0 + 0.5)
        if abs(x0 - ix0) <= 0.5:
            x0 = ix0 + bx * math.asinh(2.0 * (x0 - ix0) * math.sinh(0.5 / bx))
        else:
            x0 = x
            warn(f"BEmcRecCEMC::CorrectPosition: x = {x} dx = {x0 - ix0}")

        # phi bias within a module of 8 towers
        ix8 = int(x + 0.5) // 8
        x8 = x + 0.5 - ix8 * 8 - 4  # from -4 to +4
        dx = 0.10 * x8 / 4.0
        if abs(x8) > 3.3:
            dx = 0.0  # don't correct near the module edge

        xc = x0 - dx
        while xc < -0.5:
            xc += self.nx
        while xc >= self.nx - 0.5:
            xc -= self.nx

        y0 = y
        iy0 = lowint(y0 + 0.5)
        if abs(y0 - iy0) <= 0.5:
            y0 = iy0 + by * math.asinh(2.0 * (y0 - iy0) * math.sinh(0.5 / by))
        else:
            y0 = y
            warn(f"BEmcRecCEMC::CorrectPosition: y = {y} dy = {y0 - iy0}")
        return xc, y0

    def correct_shower_depth(self, ix, iy, energy, xa, ya, za):
        """BEmcRecCEMC::CorrectShowerDepth (approximate geometry branch)."""
        if not self.use_correct_shower_depth:
            return xa, ya, za

        loge = math.log(energy) if energy > 0.1 else math.log(0.1)

        # towers are tilted in phi; tuned from simulation
        phi = 0.0016 - 0.0010 * loge
        xc = xa * math.cos(phi) - ya * math.sin(phi)
        yc = xa * math.sin(phi) + ya * math.cos(phi)

        ra = math.sqrt(xa * xa + ya * ya)
        theta_twr = math.pi / 2 - CEMC_TOWER_ANGLES[iy]
        theta_tr = math.atan2(za - self.vz, ra)

        # shower CG in the longitudinal direction
        length = -1.79968 + 0.837322 * loge
        dz = length * math.sin(theta_tr - theta_twr) / math.cos(theta_twr)
        # baseline correction: the approximate geometry is projected at 93.5 cm,
        # roughly 90% of the actual average EMCal radius
        dz -= self.vz * 0.10

        return xc, yc, za - dz

    def tower2global(self, energy, xc, yc):
        """BEmcRec::Tower2Global - local (tower units) to global (cm) position."""
        ix = int(xc + 0.5)
        if ix < 0 or ix >= self.nx:
            warn(f"BEmcRec::Tower2Global: wrong input x: {ix}")
            return 0.0, 0.0, 0.0
        iy = int(yc + 0.5)
        if iy < 0 or iy >= self.ny:
            warn(f"BEmcRec::Tower2Global: wrong input y: {iy}")
            return 0.0, 0.0, 0.0

        g = self.get_tower_geometry(ix, iy)
        if g is None:
            # cluster CoG outside the EMCal: take geometry from the neighbouring tower
            idx = [1, 0, -1, 0]
            idy = [0, 1, 0, -1]
            ii = 0
            while ii < 4 and self.get_tower_geometry(ix + idx[ii], iy + idy[ii]) is None:
                ii += 1
            if ii >= 4:
                warn(f"BEmcRec::Tower2Global: can not identify neighbour for tower ({ix},{iy})")
                return 0.0, 0.0, 0.0
            g = list(self.get_tower_geometry(ix + idx[ii], iy + idy[ii]))
            g[0] = g[0] - idx[ii] * g[3] - idy[ii] * g[6]
            g[1] = g[1] - idx[ii] * g[4] - idy[ii] * g[7]
            g[2] = g[2] - idx[ii] * g[5] - idy[ii] * g[8]

        xt = g[0] + (xc - ix) * g[3] + (yc - iy) * g[6]
        yt = g[1] + (xc - ix) * g[4] + (yc - iy) * g[7]
        zt = g[2] + (xc - ix) * g[5] + (yc - iy) * g[8]
        return self.correct_shower_depth(ix, iy, energy, xt, yt, zt)

    # ------------------------------------------------------------------ EmcCluster methods

    @staticmethod
    def total_energy(hits):
        return sum(h.amp for h in hits)

    @staticmethod
    def max_tower(hits):
        """EmcCluster::GetMaxTower"""
        emax = 0.0
        ht = EmcModule(-1, 0.0, 0.0)
        for h in hits:
            if h.amp > emax:
                emax = h.amp
                ht = h
        return ht

    def ecore(self, hits):
        """EmcCluster::GetECore - energy in towers predicted to carry > 1% of the shower."""
        thresh = 0.01
        energy, xcg, ycg, _, _, _ = self.momenta(hits)
        es = 0.0
        for h in hits:
            iy = h.ich // self.nx
            ix = h.ich - iy * self.nx
            if self.predict_energy(energy, xcg, ycg, ix, iy) > thresh:
                es += h.amp
        return es

    def ecore_corrected(self, hits):
        """EmcCluster::GetECoreCorrected"""
        return self.correct_ecore(self.ecore(hits))

    def global_pos(self, hits):
        """EmcCluster::GetGlobalPos"""
        e = self.total_energy(hits)
        e_m, x, y, _, _, _ = self.momenta(hits)
        xc, yc = self.correct_position(e_m, x, y)
        return self.tower2global(e, xc, yc)

    def get_subclusters(self, hits, do_splitting=True):
        """EmcCluster::GetSubClusters - split a cluster at its local maxima.

        Returns (list of subcluster hit lists, list of peak EmcModules).
        """
        nx = self.nx
        nhit = len(hits)
        if nhit <= 0:
            return [], []

        hlist = sorted((h.copy() for h in hits), key=lambda m: m.ich)

        # find peaks (towers with a local maximum amplitude in a 3x3 group)
        peak_ch = []
        maxc = 0
        maxamp = 0.0
        for ic in range(nhit):
            amp = hlist[ic].amp
            if amp <= self.peak_thresh:
                continue
            ich = hlist[ic].ich
            # look into three rows only
            ichmax = (ich // nx + 2) * nx - 1
            ichmin = (ich // nx - 1) * nx
            ixc = ich - (ich // nx) * nx
            is_peak = True
            ina = ic + 1
            # check right, and 3 towers above, for another max
            while ina < nhit and hlist[ina].ich <= ichmax:
                ixa = hlist[ina].ich - (hlist[ina].ich // nx) * nx
                if abs(self.i_tower_dist(ixa, ixc)) <= 1 and hlist[ina].amp >= amp:
                    is_peak = False
                    break
                ina += 1
            if is_peak:
                ina = ic - 1
                # check left, and 3 towers below, for another max
                while ina >= 0 and hlist[ina].ich >= ichmin:
                    ixa = hlist[ina].ich - (hlist[ina].ich // nx) * nx
                    if abs(self.i_tower_dist(ixa, ixc)) <= 1 and hlist[ina].amp > amp:
                        is_peak = False
                        break
                    ina -= 1
            if not is_peak:
                continue
            if len(peak_ch) >= MAX_N_PEAKS:
                raise RuntimeError(f"EmcCluster::GetSubClusters: too many peaks in a cluster (>{MAX_N_PEAKS})")
            if amp > maxamp:
                maxamp = amp
                maxc = len(peak_ch)
            peak_ch.append(ic)
        npk = len(peak_ch)

        if not do_splitting:
            peak = hlist[peak_ch[maxc]] if npk > 0 else self.max_tower(hits)
            return [hlist], [peak]

        # there was only one peak
        if npk <= 1:
            peak = hlist[peak_ch[0]] if npk == 1 else self.max_tower(hits)
            return [hlist], [peak]

        # more than one peak: divide tower energies among showers positioned in every peak
        energy = [[0.0] * nhit for _ in range(npk)]
        tot_energy = [0.0] * nhit
        tmp_energy = [0.0] * nhit
        epk = [0.0] * (npk * 2)
        xpk = [0.0] * (npk * 2)
        ypk = [0.0] * (npk * 2)

        ratio = 1.0
        for it in range(PEAK_ITER):
            tmp_energy = [0.0] * nhit
            for ipk in range(npk):
                ic = peak_ch[ipk]
                if it > 0:
                    ratio = energy[ipk][ic] / tot_energy[ic]
                eg = hlist[ic].amp * ratio
                ixypk = hlist[ic].ich
                iypk = ixypk // nx
                ixpk = ixypk - iypk * nx
                epk[ipk] = eg
                xpk[ipk] = 0.0  # CoG from the peak tower
                ypk[ipk] = 0.0

                ichmax = (iypk + 2) * nx - 1
                ichmin = (iypk - 1) * nx

                # add up energies of towers to the right and up
                for inn in range(ic + 1, nhit):
                    ixy = hlist[inn].ich
                    if ixy > ichmax:
                        break
                    iy = ixy // nx
                    ix = ixy - iy * nx
                    idx = self.i_tower_dist(ixpk, ix)
                    idy = iy - iypk
                    if abs(idx) <= 1:
                        if it > 0:
                            ratio = energy[ipk][inn] / tot_energy[inn]
                        eg = hlist[inn].amp * ratio
                        epk[ipk] += eg
                        xpk[ipk] += eg * idx
                        ypk[ipk] += eg * idy

                # add up energies of towers to the left and down
                for inn in range(ic - 1, -1, -1):
                    ixy = hlist[inn].ich
                    if ixy < ichmin:
                        break
                    iy = ixy // nx
                    ix = ixy - iy * nx
                    idx = self.i_tower_dist(ixpk, ix)
                    idy = iy - iypk
                    if abs(idx) <= 1:
                        if it > 0:
                            ratio = energy[ipk][inn] / tot_energy[inn]
                        eg = hlist[inn].amp * ratio
                        epk[ipk] += eg
                        xpk[ipk] += eg * idx
                        ypk[ipk] += eg * idy

                xpk[ipk] = xpk[ipk] / epk[ipk] + ixpk
                ypk[ipk] = ypk[ipk] / epk[ipk] + iypk

                for inn in range(nhit):
                    ixy = hlist[inn].ich
                    iy = ixy // nx
                    ix = ixy - iy * nx
                    dx = self.f_tower_dist(float(ix), xpk[ipk])
                    dy = ypk[ipk] - iy
                    a = 0.0
                    # predict energy within 2.5 cell square around local peak
                    if abs(dx) < 2.5 and abs(dy) < 2.5:
                        a = epk[ipk] * self.predict_energy(epk[ipk], xpk[ipk], ypk[ipk], ix, iy)
                    energy[ipk][inn] = a
                    tmp_energy[inn] += a

            flim = 0.000001
            for inn in range(nhit):
                tot_energy[inn] = tmp_energy[inn] if tmp_energy[inn] > flim else flim

        igmpk1 = [-1] * npk
        igmpk2 = [-1] * npk
        ng = 0
        for ipk in range(npk):
            nh = 0
            for inn in range(nhit):
                if tmp_energy[inn] > 0:
                    a = hlist[inn].amp * energy[ipk][inn] / tmp_energy[inn]
                    if a > SUBCLUSTER_EMIN:
                        nh += 1
            if nh > 0:
                igmpk1[ipk] = ng
                igmpk2[ipk] = ng
                ng += 1

        tmp_energy = [0.0] * nhit
        for ipk in range(npk):
            if igmpk1[ipk] >= 0:
                for inn in range(nhit):
                    energy[ipk][inn] = 0.0
                    for ig in range(igmpk1[ipk], igmpk2[ipk] + 1):
                        ixy = hlist[inn].ich
                        iy = ixy // nx
                        ix = ixy - iy * nx
                        a = epk[ig] * self.predict_energy(epk[ig], xpk[ig], ypk[ig], ix, iy)
                        energy[ipk][inn] += a
                        tmp_energy[inn] += a

        subclusters = []
        peaks = []
        for ipk in range(npk):
            phit = []
            for inn in range(nhit):
                if tmp_energy[inn] > 0:
                    a = hlist[inn].amp * energy[ipk][inn] / tmp_energy[inn]
                    if a > SUBCLUSTER_EMIN:
                        phit.append(EmcModule(hlist[inn].ich, a, hlist[inn].tof))
            if phit:
                peaks.append(hlist[peak_ch[ipk]])
                subclusters.append(phit)
        return subclusters, peaks


def run_template_clustering(towers, tower_threshold=0.020, peak_threshold=0.200, min_cluster_e=0.0,
                            vz=0.0, subcluster_splitting=True, correct_position=True,
                            correct_shower_depth=True):
    """Cluster CEMC towers with the sPHENIX template algorithm.

    towers: iterable of calo_common.Tower (non-CEMC towers are ignored).
    Returns a JSON-serialisable dict.
    """
    geom = GEOMETRY[LAYER_EMCAL]
    bemc = BEmcRecCEMC(geom, vz=vz, peak_thresh=peak_threshold,
                       use_correct_position=correct_position,
                       use_correct_shower_depth=correct_shower_depth)
    nx = bemc.nx

    input_towers = {}
    hits = []
    for t in towers:
        if t.layer != LAYER_EMCAL:
            continue
        input_towers[(t.ieta, t.iphi)] = t
        # IsAcceptableTower
        if t.energy < tower_threshold:
            continue
        hits.append(EmcModule(t.ieta * nx + t.iphi, t.energy, t.time))

    clusters_out = []
    for hl in bemc.find_clusters(hits):
        subclusters, peaks = bemc.get_subclusters(hl, subcluster_splitting)
        for pp, peak in zip(subclusters, peaks):
            ecl = bemc.total_energy(pp)
            if ecl < min_cluster_e:
                continue
            ecore = bemc.ecore_corrected(pp)
            _, xcg, ycg, xx, yy, xy = bemc.momenta(pp)
            xg, yg, zg = bemc.global_pos(pp)
            r = math.hypot(xg, yg)
            hmax = bemc.max_tower(pp)

            towers_out = []
            ew_num = 0.0
            ew_den = 0.0
            saw_nonzero_t = False
            for h in pp:
                iy = h.ich // nx
                ix = h.ich % nx
                t = input_towers.get((iy, ix))
                towers_out.append({
                    "ieta": iy,
                    "iphi": ix,
                    "eta": t.eta if t else geom.get_etacenter(iy),
                    "phi": t.phi if t else geom.get_phicenter(ix),
                    "energy": h.amp,
                })
                if math.isfinite(h.tof):
                    if abs(h.tof) > 1e-9:
                        saw_nonzero_t = True
                    ew_num += h.amp * h.tof
                ew_den += h.amp
            mean_time = ew_num / ew_den if (ew_den > 0 and saw_nonzero_t) else float("nan")

            xcorr, ycorr = bemc.correct_position(ecl, xcg, ycg)

            clusters_out.append({
                "id": len(clusters_out),
                "energy": ecl,
                "ecore": ecore,
                "eta": math.asinh((zg - vz) / r) if r > 0 else float("nan"),
                "eta_detector": math.asinh(zg / r) if r > 0 else float("nan"),
                "phi": math.atan2(yg, xg),
                "et": ecl / math.cosh(math.asinh((zg - vz) / r)) if r > 0 else float("nan"),
                "x": xg,
                "y": yg,
                "z": zg,
                "r": r,
                "prob": None,
                "chi2": None,
                "mean_time": finite_or_none(mean_time),
                "ntowers": len(pp),
                "peak_tower": {"ieta": peak.ich // nx, "iphi": peak.ich % nx, "energy": peak.amp},
                "max_tower": {"ieta": hmax.ich // nx, "iphi": hmax.ich % nx, "energy": hmax.amp},
                "tower_cog": {"phi_raw": xcg, "eta_raw": ycg, "phi_corr": xcorr, "eta_corr": ycorr},
                "tower_moments": {"phiphi": xx, "etaeta": yy, "etaphi": xy},
                "towers": towers_out,
            })

    for c in clusters_out:
        for k in ("eta", "eta_detector", "et"):
            c[k] = finite_or_none(c[k])

    return {
        "algorithm": "RawClusterBuilderTemplate",
        "detector": geom.name,
        "parameters": {
            "tower_threshold": tower_threshold,
            "peak_threshold": peak_threshold,
            "min_cluster_e": min_cluster_e,
            "vertex_z": vz,
            "subcluster_splitting": subcluster_splitting,
            "correct_position": correct_position,
            "correct_shower_depth": correct_shower_depth,
            "geometry": "former (approximate, cylinder R=93.5 cm, uniform eta/phi bins)",
        },
        "n_input_towers": len(input_towers),
        "n_towers_above_threshold": len(hits),
        "n_clusters": len(clusters_out),
        "clusters": clusters_out,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="calo_features.mmap directory (with --event), or CSV file with columns eta, phi, E (optional: ieta, iphi, layer, time)")
    p.add_argument("--event", type=int, help="event number in the mmap directory")
    p.add_argument("-o", "--output", help="output JSON file (default: stdout)")
    p.add_argument("--tower-threshold", type=float, default=0.020, help="minimum tower energy [GeV] (default 0.020)")
    p.add_argument("--peak-threshold", type=float, default=0.200, help="minimum local-maximum energy [GeV] (default 0.200)")
    p.add_argument("--min-cluster-e", type=float, default=0.0, help="minimum cluster energy saved [GeV] (default 0)")
    p.add_argument("--vz", type=float, default=0.0, help="event vertex z [cm] (default 0)")
    p.add_argument("--no-subcluster-splitting", action="store_true", help="do not split clusters at local maxima")
    p.add_argument("--no-correct-position", action="store_true", help="disable the S-curve/phi-bias position correction")
    p.add_argument("--no-correct-shower-depth", action="store_true", help="disable the shower-depth correction")
    p.add_argument("--no-towers", action="store_true", help="omit per-cluster tower lists from the output")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (negative for compact)")
    args = p.parse_args(argv)

    towers = load_towers(args.input, event=args.event, default_layer=LAYER_EMCAL)
    result = run_template_clustering(
        towers,
        tower_threshold=args.tower_threshold,
        peak_threshold=args.peak_threshold,
        min_cluster_e=args.min_cluster_e,
        vz=args.vz,
        subcluster_splitting=not args.no_subcluster_splitting,
        correct_position=not args.no_correct_position,
        correct_shower_depth=not args.no_correct_shower_depth,
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
