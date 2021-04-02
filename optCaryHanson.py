#!/usr/bin/env python3
from simsopt.core.least_squares_problem import LeastSquaresProblem
from simsopt.solve.serial_solve import least_squares_serial_solve
from simsopt.geo.magneticfieldclasses import ToroidalField
from pyoculus.solvers  import FixedPoint, PoincarePlot
from simsopt.geo.curvehelical import CurveHelical
from simsopt.core.optimizable import Optimizable
from simsopt.geo.biotsavart import BiotSavart
from pyoculus.problems import CartesianBfield
import matplotlib.pyplot as plt
import numpy as np
############################################
# BioSavart class to pyOculus
class SimsgeoBiotSavart(CartesianBfield):
    def __init__(self, bs, R0, Z0, Nfp=1):
        super().__init__(R0, Z0, Nfp)
        self._bs = bs

    def B(self, xyz, args=None):
        point = np.array([xyz])
        self._bs.set_points(point)
        Bfield=self._bs.B()
        return Bfield[0]

    def dBdX(self, xyz, args=None):
        point = np.array([xyz])
        self._bs.set_points(point)
        dB=self._bs.dB_by_dX()
        return dB[0]
# Optimizable class specific to the CaryHanson problem
class objCoilsResidue(Optimizable):
    def __init__(self):
        self.coils     = [CurveHelical(200, 2, 5, 2, 1., 0.3) for i in range(2)]
        self.coils[0].set_dofs(np.concatenate(([np.pi/2,0],[0,0])))
        self.coils[1].set_dofs(np.concatenate(([0,0],[0,0])))
        self.currents  = [3.07e5,-3.07e5]
        self.ndofs     = 8
        self.Btoroidal = ToroidalField(1.0,1.0)
        self.Bhelical  = BiotSavart(self.coils, self.currents)
        self.Bfield    = self.Bhelical+self.Btoroidal
        self.NFP       = 5
        self.magnetic_axis_radius = 0.983
        self.sbsp      = SimsgeoBiotSavart(self.Bfield, self.magnetic_axis_radius, Z0=0, Nfp=self.NFP)
        self._set_names()
    def _set_names(self):
        self.names = ['A1(1)', 'A1(2)', 'B1(1)', 'B1(2)', 'A2(1)', 'A2(2)', 'B2(1)', 'B2(2)']
    def get_dofs(self):
        return np.concatenate((self.coils[0].get_dofs(),self.coils[1].get_dofs()))
    def set_dofs(self, dofs):
        self.coils[0].set_dofs(np.concatenate(([dofs[0],dofs[1]],[dofs[2],dofs[3]])))
        self.coils[1].set_dofs(np.concatenate(([dofs[4],dofs[5]],[dofs[6],dofs[7]])))
        self.Bhelical = BiotSavart(self.coils, self.currents)
        self.Bfield   = self.Bhelical+self.Btoroidal
        self.sbsp     = SimsgeoBiotSavart(self.Bfield, self.magnetic_axis_radius, Z0=0, Nfp=self.NFP)
    def residue1(self):
        pp1     = -5
        qq1     = 3
        guess1  = 1.009
        sbegin1 = 0.99
        send1   = 1.04
        self.fp1 = FixedPoint(self.sbsp, {"Z":0.0})
        self.output1=self.fp1.compute(guess=guess1, pp=pp1, qq=qq1, sbegin=sbegin1, send=send1)
        try:
            self.residue1=self.output1.GreenesResidue
        except:
            self.residue1=0
        return self.residue1
    def residue2(self):
        pp2     = -5
        qq2     = 3
        guess2  = 0.82
        sbegin2 = 0.80
        send2   = 0.9
        self.fp2 = FixedPoint(self.sbsp, {"Z":0.0})
        self.output2=self.fp2.compute(guess=guess2, pp=pp2, qq=qq2, sbegin=sbegin2, send=send2)
        try:
            self.residue2=self.output2.GreenesResidue
        except:
            self.residue2=0
        return self.residue2
    def poincare(self, Rbegin=0.982, Rend=1.008, nPpts=100, nPtrj=4):
        params = dict()
        params["Rbegin"] = Rbegin
        params["Rend"]   = Rend
        params["nPpts"]  = nPpts
        params["nPtrj"]  = nPtrj
        self.p           = PoincarePlot(self.sbsp, params)
        self.poincare_output = self.p.compute()
        self.iota        = self.p.compute_iota()
        return self.p

if __name__ == "__main__":
    ## Start optimizable class
    obj  = objCoilsResidue()
    print('Initial degrees of freedom =',obj.get_dofs())

    ## Create initial Poincare Plot
    # p = obj.poincare()
    # plt.figure()
    # p.plot(s=1)
    # plt.figure()
    # p.plot_iota()

    ## Set degrees of freedom for the optimization
    obj.all_fixed()
    obj.set_fixed('A1(1)', False)
    obj.set_fixed('A1(2)', False)
    # obj.set_fixed('B1(1)', False)
    # obj.set_fixed('B1(2)', False)
    # obj.set_fixed('A2(1)', False)
    # obj.set_fixed('A2(2)', False)
    # obj.set_fixed('B2(1)', False)
    # obj.set_fixed('B2(2)', False)

    ## Define and run optimization problem
    prob = LeastSquaresProblem([(obj.residue1, 0, 1),
                                (obj.residue2, 0, 1)])
    nIterations = 4
    least_squares_serial_solve(prob, max_nfev=nIterations)
    print('Final degrees of freedom =',obj.get_dofs())

    ## Create final Poincare and iota plots
    # p = obj.poincare()
    # plt.figure()
    # p.plot(s=1)
    # plt.figure()
    # p.plot_iota()

    ## Show plots
    # plt.show()