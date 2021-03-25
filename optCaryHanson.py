#!/usr/bin/env python3
from simsopt.geo.magneticfield import MagneticField, ToroidalField, HelicalField
from pyoculus.problems import CartesianBfield
from poincareplot import compute_field_lines
from pyoculus.solvers  import FixedPoint
import matplotlib.pyplot as plt
import numpy as np
############################################
magnetic_axis_radius = 0.98
#### Residue parameters
pp     = -1
qq     = 8
guess  = 1.021
sbegin = 1.015
send   = 1.15

# BioSavart class to pyOculus
class SimsgeoBiotSavart(CartesianBfield):
    def __init__(self, bs, R0, Z0, Nfp=1):
        """! Set up the problem to compute the magnetic field for simsopt.geo
        @param R0 the magnetic axis R coordinate at phi=0 plane
        @param Z0 the magnetic axis Z coordinate at phi=0 plane
        """
        super().__init__(R0, Z0, Nfp)
        # if not isinstance(bs, MagneticField):
        #     raise TypeError("bs should be an instance of simsopt.geo.MagneticField")
        self._bs = bs

    def B(self, xyz, args=None):
        """! The magnetic field, being used by parent class CartesianBfield
        @param xyz array with coordinates \f$(x, y z)\f$
        @returns \f$(B_x, B_y, B_z)\f$
        """
        point = [xyz]
        self._bs.set_points(point)
        Bfield=self._bs.B()
        return Bfield[0]

    def dBdX(self, xyz, args=None):
        """! The derivative of the magnetic field, being used by parent class CartesianBfield
        @param xyz array with coordinates \f$(x, y z)\f$
        @returns \f$\partial (B_x, B_y, B_z)/\partial (x,y,z)\f$
        """
        point = [xyz]
        self._bs.set_points(point)
        dB=self._bs.dB_by_dX()
        return dB[0]

BField=HelicalField()
sbsp = SimsgeoBiotSavart(BField, magnetic_axis_radius, Z0=0, Nfp=BField.nfp)
fp = FixedPoint(sbsp, {"Z":0.0})
output=fp.compute(guess=guess, pp=pp, qq=qq, sbegin=sbegin, send=send)
residue=output.GreenesResidue
print(residue)