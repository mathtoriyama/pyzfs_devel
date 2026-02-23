from __future__ import absolute_import, division, print_function
import numpy as np
import os, sys
from mpi4py import MPI

from gpaw import GPAW
from gpaw.old.pw.descriptor import PWDescriptor
from gpaw.mpi import serial_comm

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform  #, fftshift, ifftshift, irfftn, ifftn
from .wavefunction import Wavefunction
# from ..counter import Counter
from ..parallel import SymmetricDistributedMatrix

from ...common.misc import empty_ase_cell

class GPAWWavefunctionLoader(WavefunctionLoader):

    def __init__(self, gpwfile, comm=MPI.COMM_WORLD):
        self.gpwfile = gpwfile
        self.calc = None
        super(GPAWWavefunctionLoader, self).__init__()

    def scan(self):
        super(GPAWWavefunctionLoader, self).scan()
        
        # Load GPAW calculator
        self.calc = GPAW(self.gpwfile, communicator=serial_comm)
        wfs = self.calc.wfs
        
        # Parse cell
        cell = Cell(empty_ase_cell(
            *self.calc.atoms.get_cell().array.T,
            unit="angstrom"
        ))

        # Create dummy ft objects
        self.wft = FourierTransform(1, 1, 1)
        self.dft = FourierTransform(1, 1, 1)
                
        # Spin / k-point sanity checks
        assert self.calc.get_number_of_spins() == 2
        assert len(wfs.kpt_u) == 2  # up, down
        for kpt in wfs.kpt_u:
            assert kpt.k == 0.0  # Gamma only
        self.gamma = True

        # Occupied orbitals
        occs = [kpt.f_n for kpt in wfs.kpt_u]
        iuorbs = np.where(occs[0] > 0.8)[0]
        idorbs = np.where(occs[1] > 0.8)[0]
        iorb_sb_map = (
            [("up", int(n)) for n in iuorbs] +
            [("down", int(n)) for n in idorbs]
        )
        nuorbs = len(iuorbs)
        ndorbs = len(idorbs)
        norbs = nuorbs + ndorbs
        iorb_fname_map = ["None"] * norbs  # GPAW does not use files

        # G-vectors
        pd = wfs.pd
        assert isinstance(pd, PWDescriptor)
        self.gvecs = wfs.pd.get_reciprocal_vectors(add_q=False) # Splits Gvecs in MPI, but want all processes to have all Gvecs
        """
        assert len(pd.Q_qG) == 1, "Various numbers of reciprocal vectors detected..."
        indices = pd.Q_qG[0]
        self.gvecs = pd.G_Qv[indices]
        """

        self.wfc = Wavefunction(
            cell=cell,
            ft=self.wft,
            nuorbs=nuorbs,
            ndorbs=ndorbs,
            iorb_sb_map=iorb_sb_map,
            iorb_fname_map=iorb_fname_map,
            dft=self.dft,
            gamma=self.gamma,
            gvecs=self.gvecs,
        )
        self.wfc.add_gpaw_calc(self.calc)


    def load(self, iorbs, sdm):
        pass

