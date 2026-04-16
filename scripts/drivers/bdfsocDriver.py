#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
# PSND SCRIPTS (adapted from COMBRAMM)
# Author: xshinhe
#
# Copyright (c) 2024 Peking Univ. - GNUv3 License
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import os
import sys
import shutil
from copy import deepcopy

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from drivers.bdfDriver import BdfInput
import constants
from QMOutput import QMOutput


class BdfSocInput(BdfInput):
    pass


class BdfSocOutput(QMOutput):

    def __init__(self, name, calcdir, SPcalc=False, verbose=False):
        QMOutput.__init__(self)

        self.dataDict = {}

        self.dataDict["inpfile"] = name
        self.dataDict["calcdir"] = calcdir
        self.dataDict["outfile"] = None

        self.dataDict["charges"] = []
        self.dataDict["elfield"] = {}
        self.dataDict["dipole"] = []
        self.dataDict["osc_strength"] = {}
        self.dataDict["natoms"] = None
        self.dataDict["nroots"] = 0
        self.dataDict["optstate"] = 0

        self.dataDict["energy"] = {}
        self.dataDict["gradient"] = {}
        self.dataDict["hess"] = {}
        self.dataDict["nac"] = {}
        self.dataDict["fullgradcharge"] = {}
        self.dataDict["fullnaccharge"] = {}
        self.dataDict["gradcharges_extraterm"] = False

        self.dataDict["soc"] = {}
        self.dataDict["soc_TT"] = {}
        self.dataDict["energy_triplet"] = {}
        self.dataDict["gradient_triplet"] = {}
        self.dataDict["nac_triplet"] = {}
        self.dataDict["index_singlet"] = {}

        self.dataDict["termination"] = None
        self.dataDict["errormsg"] = []
        self.dataDict["signs"] = []
        self.dataDict["scf_mo_maps"] = None
        self.dataDict["psioverlap"] = None

        self.dataDict["eigenvectors"] = None
        self.dataDict["selfenergy"] = 0.0

        with open(os.path.join(calcdir, name)) as fout:
            self.dataDict["outfile"] = fout.read()

        output = self.dataDict["outfile"].splitlines()
        joined_output = "".join(output)
        if any(marker in joined_output for marker in ("Program Stop", "BDF STOP ...", "STOP!", "Program stop")):
            self.dataDict["termination"] = 1
            self.dataDict["errormsg"] = output[:]
            return

        self.dataDict["termination"] = 0

        au_2_ang = 5.291772104260590e-01
        au_2_ev = 2.7211386033e+01
        au_2_wn = 2.1947463147e+05

        state_for_GRAD = None
        has_GS = False
        istate = 0
        nac_istate = 0
        nac_jstate = 0
        idfile1 = 0
        idfile2 = 0
        natom = None
        istate_triplet = 0

        i = 0
        while i < len(output):
            line = output[i]

            if "ERROR" in line:
                self.dataDict["errormsg"].append(line)
                i += 1
                continue

            if "State          X           Y           Z          Osc." in line:
                k = i + 1
                self.dataDict["osc_strength"][0] = 0
                cnt = 1
                while k < len(output) and output[k].strip() != "":
                    terms = output[k].strip().split()
                    self.dataDict["osc_strength"][cnt] = float(terms[4])
                    k += 1
                    cnt += 1
                i = k + 1
                continue

            if "GSGRAD_cstate=" in line or "EXGRAD_cstate=" in line:
                if "GSGRAD_cstate" in line:
                    has_GS = True
                listfile = line.strip().split()
                if listfile[1] in ["0", "1"]:
                    idstate = listfile[3]
                    if idstate not in self.dataDict["index_singlet"].values():
                        self.dataDict["index_singlet"][istate] = idstate
                        energy = float(output[i + 1].split("=")[1])
                        self.dataDict["energy"][istate] = energy

                        grad_local = [[], [], []]
                        j = i + 3
                        terms = output[j].strip().split()
                        while terms[0] != "Sum":
                            grad_local[0].append(float(terms[1]))
                            grad_local[1].append(float(terms[2]))
                            grad_local[2].append(float(terms[3]))
                            j += 1
                            terms = output[j].strip().split()
                        self.dataDict["gradient"][istate] = np.array(grad_local)
                        istate += 1
                        i = j + 1
                        continue
                elif listfile[1] == "2":
                    energy = float(output[i + 1].split("=")[1])
                    self.dataDict["energy_triplet"][istate_triplet] = energy

                    grad_local = [[], [], []]
                    j = i + 3
                    terms = output[j].strip().split()
                    while terms[0] != "Sum":
                        grad_local[0].append(float(terms[1]))
                        grad_local[1].append(float(terms[2]))
                        grad_local[2].append(float(terms[3]))
                        j += 1
                        terms = output[j].strip().split()
                    self.dataDict["gradient_triplet"][istate_triplet] = np.array(grad_local)
                    istate_triplet += 1
                    i = j + 1
                    continue

            if "GSGRAD_estate=" in line or "EXGRAD_estate=" in line:
                if "GSGRAD_estate" in line:
                    has_GS = True

                energy = float(line.split("=")[1])
                self.dataDict["energy"][istate] = energy

                grad_local = [[], [], []]
                j = i + 2
                terms = output[j].strip().split()
                while terms[0] != "Sum":
                    grad_local[0].append(float(terms[1]))
                    grad_local[1].append(float(terms[2]))
                    grad_local[2].append(float(terms[3]))
                    j += 1
                    terms = output[j].strip().split()
                self.dataDict["gradient"][istate] = np.array(grad_local)
                istate += 1
                i = j + 1
                continue

            if "FNAC_cpair=" in line:
                terms = line.strip().split()
                nac_istate, nac_jstate = int(terms[3]), int(terms[6])
                if len(terms) >= 5:
                    idfile1, idfile2 = int(terms[1]), int(terms[4])
                if idfile1 != 2 and idfile2 != 2:
                    if not has_GS:
                        nac_istate -= 1
                        nac_jstate -= 1
                else:
                    nac_istate -= 1
                    nac_jstate -= 1
                i += 1
                continue

            if "Gradient contribution from Final-NAC(S)-Escaled" in line:
                grad_local = [[], [], []]
                j = i + 1
                terms = output[j].strip().split()
                while terms[0] != "Sum":
                    grad_local[0].append(float(terms[1]))
                    grad_local[1].append(float(terms[2]))
                    grad_local[2].append(float(terms[3]))
                    j += 1
                    terms = output[j].strip().split()

                target_key = "nac" if idfile1 != 2 and idfile2 != 2 else "nac_triplet"
                if nac_istate not in self.dataDict[target_key]:
                    self.dataDict[target_key][nac_istate] = {}
                self.dataDict[target_key][nac_istate][nac_jstate] = grad_local
                if nac_jstate not in self.dataDict[target_key]:
                    self.dataDict[target_key][nac_jstate] = {}
                self.dataDict[target_key][nac_jstate][nac_istate] = [[-x for x in y] for y in grad_local]

                i = j + 1
                continue

            if "Cartesian coordinates (Angstrom)" in line:
                sym = []
                xyz = []
                j = i + 4
                while j < len(output):
                    ll = output[j].strip()
                    if ll == "":
                        j += 1
                        continue
                    if ll[0].strip()[0] == "-":
                        break
                    terms = ll.strip().split()
                    sym.append(terms[1])
                    xyz.append(np.array(terms[3:6]).astype(np.float64))
                    j += 1

                xyz = np.array(xyz)
                natom = len(xyz)

                self.dataDict["sym"] = sym
                self.dataDict["x0"] = xyz / au_2_ang

                mass = np.zeros(3 * len(sym))
                mdict = {"C": 12.01, "H": 1.008, "O": 16.00}
                for k in range(len(mass)):
                    mass[k] = mdict[sym[k // 3]] * 1850
                self.dataDict["mass"] = mass

                i = j + 1
                continue

            if " Results of vibrations:" in line:
                hess = np.zeros((N, N))
                Tmod = np.zeros((N, N))
                w = np.zeros((N))
                j = i + 1
                read_col_num = 0
                while j < len(output) and read_col_num < 3 * natom:
                    if "Irreps" in output[j]:
                        w[read_col_num:read_col_num + 3] = np.array(output[j + 1].strip().split()[1:4]).astype(np.float64)
                        for k in range(natom):
                            tget = np.array(output[j + 5 + k].strip().split()[2:11]).astype(np.float64)
                            Tmod[3 * k:3 * k + 3, read_col_num] = tget[0:3]
                            Tmod[3 * k:3 * k + 3, read_col_num + 1] = tget[3:6]
                            Tmod[3 * k:3 * k + 3, read_col_num + 2] = tget[6:9]
                        read_col_num += 3
                        j += 5 + natom
                    else:
                        j += 1
                wnew = deepcopy(w)
                Tnew = deepcopy(Tmod)
                wnew[0:6] = w[-6:]
                wnew[6:] = w[:-6]
                Tnew[:, 0:6] = Tmod[:, -6:]
                Tnew[:, 6:] = Tmod[:, :-6]
                self.dataDict["Tmod"] = Tnew
                self.dataDict["w"] = wnew / au_2_wn

                i = j + 1
                continue

            if "SocPairNo. =" in line:
                terms = line.strip().split()
                soc_nfile1, soc_nfile2 = int(terms[6]), int(terms[10])
                if soc_nfile1 != soc_nfile2:
                    soc_state_singlet, soc_state_triplet = int(terms[8]), int(terms[12])
                    if soc_nfile1 > soc_nfile2:
                        soc_state_singlet, soc_state_triplet = soc_state_triplet, soc_state_singlet
                    if not has_GS:
                        soc_state_singlet -= 1
                    soc_state_triplet -= 1

                    soc_local = []
                    for offset in range(2, 5):
                        soc_terms = output[i + offset].strip().split()
                        soc_local.append(float(soc_terms[2]) + 1j * float(soc_terms[4]))

                    if soc_state_singlet not in self.dataDict["soc"]:
                        self.dataDict["soc"][soc_state_singlet] = {}
                    self.dataDict["soc"][soc_state_singlet][soc_state_triplet] = soc_local

                    i += 4
                    continue

                if (soc_nfile1 == 2 and soc_nfile2 == 2) or (soc_nfile1 == 1 and soc_nfile2 == 1):
                    soc_state_triplet1, soc_state_triplet2 = int(terms[8]), int(terms[12])
                    soc_state_triplet1 -= 1
                    soc_state_triplet2 -= 1

                    soc_local = []
                    for offset in range(2, 11):
                        soc_terms = output[i + offset].strip().split()
                        soc_local.append(float(soc_terms[2]) + 1j * float(soc_terms[4]))

                    if soc_state_triplet1 not in self.dataDict["soc_TT"]:
                        self.dataDict["soc_TT"][soc_state_triplet1] = {}
                    self.dataDict["soc_TT"][soc_state_triplet1][soc_state_triplet2] = soc_local

                    i += 10
                    continue

            if "MULLIKEN POPULATION ANALYSIS." in line:
                self.dataDict["charges"] = []
                k = i + 4
                while k < len(output) and output[k].strip() != "":
                    self.dataDict["charges"].append(float(output[k].split()[2]))
                    k += 1
                i = k + 1
                continue

            if "ELECTROSTATIC POTENTIAL (VOLT) AND ELECTRIC FIELD COMPONENTS" in line:
                self.dataDict["elfield"] = {}
                k = i + 4
                elfield = [], [], []
                while k < len(output) and output[k].strip() != "":
                    element = output[k].split()
                    ex, ey, ez = float(element[2]), float(element[3]), float(element[4])
                    elfield[0].append(ex)
                    elfield[1].append(ey)
                    elfield[2].append(ez)
                    k += 1
                self.dataDict["elfield"][state_for_GRAD] = elfield
                i = k + 1
                continue

            if "Properties of transitions   1 -> #" in line:
                self.dataDict["osc_strength"] = {}
                k = i + 3
                while k < len(output) and output[k].strip() != "":
                    terms = output[k].strip().split()
                    to_state = int(terms[0]) - 1
                    self.dataDict["osc_strength"][to_state] = float(terms[-2])
                    k += 1
                i = k + 1
                continue

            if " DIPOLE              X           Y           Z         TOTAL" in line:
                k = i + 4
                dip = output[k].split()
                self.dataDict["dipole"] = (
                    float(dip[1]) * constants.Debye2AU,
                    float(dip[2]) * constants.Debye2AU,
                    float(dip[3]) * constants.Debye2AU,
                    float(dip[4]) * constants.Debye2AU,
                )
                i = k + 1
                continue

            if "     INPUT GEOMETRY" in line:
                k = i + 6
                xyz_local = []
                while k < len(output) and output[k].strip() != "":
                    terms = output[k].strip().replace("*", " ").split()
                    xyz_local += [float(terms[2]), float(terms[3]), float(terms[4])]
                    k += 1
                self.dataDict["inputXYZ"] = xyz_local
                i = k + 1
                continue

            if "State  1,  Mult. 1,  E-E(1)=  0.00000" in line:
                self.dataDict["Elmin"] = float(line.split()[-2]) / au_2_ev
                i += 1
                continue

            if "GRADIENT NORM =" in line:
                self.dataDict["gradNORM"] = float(line.split()[3])
                i += 1
                continue

            if "EIGENVECTORS OF THE MASS-WEIGHTED" in line:
                k = i + 3
                colidx = np.array(output[k].split()).astype(np.int32) - 1
                k += 2
                freq[colidx] = np.array(output[k].split()).astype(np.float64) / au_2_wn
                k += 2
                while "CARTESIAN DISPLACEMENT" not in output[k]:
                    if output[k].strip() == "":
                        k += 1
                        continue
                    terms = output[k].split()
                    if len(terms) == len(colidx) + 1:
                        rowidx = int(terms[0]) - 1
                        Tmod[rowidx, colidx] = np.array(terms[1:]).astype(np.float64)
                        k += 1
                    if len(terms) <= len(colidx):
                        colidx = np.array(terms).astype(np.int32) - 1
                        k += 2
                        terms = output[k].split()
                        freq[colidx] = np.array(terms).astype(np.float64)
                        k += 1

                self.dataDict["freq"] = freq
                self.dataDict["Tmod"] = Tmod
                i = k + 1
                continue

            if "TERMINAT" in line:
                self.dataDict["termination"] = 1
                break

            if "COMPUTATION TIME" in line:
                self.dataDict["termination"] = 0
                self.dataDict["errormsg"] = []
                break

            i += 1

        if self.dataDict["soc"] and len(self.dataDict["energy_triplet"]) == 0:
            for k in range(1, len(self.dataDict["energy"])):
                self.dataDict["energy_triplet"][k - 1] = self.dataDict["energy"][k]
                self.dataDict["gradient_triplet"][k - 1] = self.dataDict["gradient"][k]
                if self.dataDict["nac"]:
                    try:
                        self.dataDict["nac_triplet"][k - 1] = {}
                        for l in self.dataDict["nac"][k].keys():
                            self.dataDict["nac_triplet"][k - 1][l - 1] = self.dataDict["nac"][k][l]
                    except KeyError:
                        continue

            for k in range(1, len(self.dataDict["energy_triplet"]) + 1):
                self.dataDict["energy"].pop(k, None)
                self.dataDict["gradient"].pop(k, None)
                self.dataDict["nac"].pop(k, None)

        if self.dataDict["energy"] == {} or self.dataDict["gradient"] == {}:
            self.dataDict["termination"] = 1
            self.dataDict["errormsg"] = output[:]
            return

    def __del__(self):
        del self.log
        del self.dataDict


if __name__ == "__main__":
    from psnd_log import Log
    from pprint import pformat

    Log.startSection(f"[TEST] {__file__}")
    file = "bdf-QM.log"
    calcdir = "qm_bdf/qmCalc00001"
    out = BdfSocOutput(file, calcdir, "TDDFT")
    for k, v in out.dataDict.items():
        if k not in ["log", "outfile"]:
            Log.writeLog(f"{k}: " + pformat(v) + "\n")
