from __future__ import absolute_import, division, print_function
import numpy as np
import os, sys
from mpi4py import MPI

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform
from .wavefunction import Wavefunction
from ..counter import Counter
from ..parallel import SymmetricDistributedMatrix

from ...common.misc import empty_ase_cell
from ..units import bohr_to_angstrom

from scipy.ndimage import zoom, _ni_support

from gpaw import GPAW
from gpaw.old.pw.descriptor import PWDescriptor
from gpaw.mpi import serial_comm
from gpaw.utilities.ps2ae import PS2AE

def _compute_offset(sdm, iorb):
    """compute the index for iorb^th wfc, note that some rows in psir_arrs_all
    are zero to facilitate MPI scatter"""
    nproc = iloc = 0
    for iproc in range(sdm.pgrid.nrow):
        mstart, mloc, mend, nstart, nloc, nend = sdm.indexmap[iproc, 0]
        if mstart > iorb:
            break
        nproc = iproc
        iloc = iorb - mstart
    return nproc * sdm.mlocx + iloc


class GPAWWavefunctionLoader(WavefunctionLoader):

    def __init__(self, gpwfile, ae=False, ae_reduce=4, comm=MPI.COMM_WORLD):
        self.gpwfile = gpwfile
        self.ae = ae
        self.ae_reduce = ae_reduce
        super(GPAWWavefunctionLoader, self).__init__()

    def scan(self):
        super(GPAWWavefunctionLoader, self).scan()
        
        # Load GPAW calculator
        self.calc_gpaw = GPAW(  self.gpwfile, 
                                communicator = serial_comm, 
                                )
        self.calc_gpaw_ps2ae = PS2AE(self.calc_gpaw)

        # Parse cell
        cell = Cell(empty_ase_cell(
            *self.calc_gpaw.atoms.get_cell().array.T,
            unit="angstrom"
        ))

        # Create FT objects
        if self.ae:
            self.ae_reduce_arr = _ni_support._normalize_sequence(1./self.ae_reduce, 3)
            self.realgrid = np.array( [int(round(ii * jj)) for ii, jj in zip(self.calc_gpaw_ps2ae.gd.N_c, self.ae_reduce_arr)] )
            # Note: Above matches the dimensions of reduced WF (https://github.com/scipy/scipy/blob/v1.17.0/scipy/ndimage/_interpolation.py#L763-L894)
        else:
            self.realgrid = self.calc_gpaw.wfs.gd.N_c
        self.wft = FourierTransform(self.realgrid[0], self.realgrid[1], self.realgrid[2])
        self.dft = FourierTransform(self.realgrid[0], self.realgrid[1], self.realgrid[2])
                
        # Spin / k-point sanity checks
        assert self.calc_gpaw.get_number_of_spins() == 2
        assert len(self.calc_gpaw.wfs.kpt_u) == 2  # up, down
        for kpt in self.calc_gpaw.wfs.kpt_u:
            assert kpt.k == 0.0  # Gamma only
        self.gamma = True

        # Occupied orbitals
        occs = [kpt.f_n for kpt in self.calc_gpaw.wfs.kpt_u]
        iuorbs = np.where(occs[0] > 0.8)[0]
        idorbs = np.where(occs[1] > 0.8)[0]
        
        nuorbs = len(iuorbs)
        ndorbs = len(idorbs)
        norbs = nuorbs + ndorbs
        
        iorb_sb_map = list(
            ("up", iuorbs[iwfc]) if iwfc < nuorbs else ("down", idorbs[iwfc - nuorbs])
            for iwfc in range(norbs)
        )
        iorb_fname_map = ["None"] * norbs  # GPAW does not use files

        self.wfc = Wavefunction(
            cell=cell,
            ft=self.wft,
            nuorbs=nuorbs,
            ndorbs=ndorbs,
            iorb_sb_map=iorb_sb_map,
            iorb_fname_map=iorb_fname_map,
            dft=self.dft,
            gamma=self.gamma,
        )

        # Set indicator of GPAW calc
        self.wfc.calc_gpaw = self.calc_gpaw

        # Set psi(r) array map
        self.wfc.iorb_psir_arr_map = {}

        # Add new (GPAW-related) functions to self.wfc object
        self.wfc.get_psir = self.get_psir_gpaw
        self.wfc.get_rhog = self.get_rhog_gpaw



    def load(self, iorbs, sdm):
        super(GPAWWavefunctionLoader, self).load(iorbs, sdm)
        assert isinstance(sdm, SymmetricDistributedMatrix)
        comm = sdm.comm
        rank = sdm.pgrid.rank
        onroot = sdm.onroot

        # Distribute WFs if all-electron needed, otherwise the pseudo WFs can be fetched quickly
        if self.ae:

            # processor 0 parse wavefunctions
            psir_arrs_all = None
            arr_len = np.prod(self.realgrid)
            if onroot:
                psir_arrs_all = np.zeros([sdm.mx, arr_len], dtype=complex)
                c = Counter(
                    self.wfc.norbs,
                    percent=0.1,
                    message="(process 0) {n} orbitals ({percent}%) loaded in {dt}...",
                )

                nbands = self.calc_gpaw.get_number_of_bands()
                for ispin in range(2):
                    for iband in range(nbands):

                        # Get all-electron (PAW-reconstructed) wave function
                        psir_ae = self.calc_gpaw_ps2ae.get_wave_function(n=iband, s=ispin, ae=self.ae)

                        # Linearly interpolate
                        psir = zoom(psir_ae, self.ae_reduce_arr, order=1)

                        # Convert units from 1/Angstrom^(3/2) to 1/bohr^(3/2)
                        psir *= bohr_to_angstrom**(3./2)

                        # Normalize
                        psir = self.wfc.normalize(psir)


                        iorb = self.wfc.sb_iorb_map.get(
                            ("up" if ispin == 0 else "down", iband)
                        )
                        if iorb is not None:
                            offset = _compute_offset(sdm, iorb)
                            psir_arrs_all[offset] = psir.flatten()
                            c.count()

            # scatter wavefunctions
            # allocate wfc arrays
            psir_arrs_m = np.zeros([sdm.mlocx, arr_len], dtype=complex)
            psir_arrs_n = np.zeros([sdm.nlocx, arr_len], dtype=complex)
            comm.barrier()

            # root -> first column scatter
            if onroot:
                print("GPAWWavefunctionLoader: root -> first column scattering")
            if sdm.icol == 0:
                sdm.colcomm.Scatter(sendbuf=psir_arrs_all, recvbuf=psir_arrs_m, root=0)
            comm.barrier()

            # first column -> other column bcast
            if onroot:
                print("GPAWWavefunctionLoader: first column -> other column bcast")
            sdm.rowcomm.Bcast(psir_arrs_m, root=0)
            comm.barrier()

            # root -> first row scatter
            if onroot:
                print("GPAWWavefunctionLoader: root -> first row scattering")
            if sdm.irow == 0:
                sdm.rowcomm.Scatter(sendbuf=psir_arrs_all, recvbuf=psir_arrs_n, root=0)
            comm.barrier()

            # first row -> other row bcast
            if onroot:
                print("GPAWWavefunctionLoader: first row -> other row bcast")
            sdm.colcomm.Bcast(psir_arrs_n, root=0)
            comm.barrier()

            if onroot:
                del psir_arrs_all

            for iloc in range(sdm.mloc):
                iorb = sdm.ltog(iloc)
                self.set_psir_arr(iorb, psir_arrs_m[iloc])

            for iloc in range(sdm.nloc):
                iorb = sdm.ltog(0, iloc)[1]
                try:
                    self.set_psir_arr(iorb, psir_arrs_n[iloc])
                except ValueError:
                    pass
            comm.barrier()
        
        else:
            # Fetch pseudo WF from calc.get_pseudo_wave_function routine (quick)
            pass


    def set_psir_arr(self, iorb, psir_arr):
        if iorb in self.wfc.iorb_psir_arr_map:
            raise ValueError("psir_arr {} already set".format(iorb))
        self.wfc.iorb_psir_arr_map[iorb] = psir_arr


    def get_psir_gpaw(self, iorb):
        """Get psi(r) of certain index, GPAW edition"""
        if self.ae:
            return self.wfc.iorb_psir_arr_map[iorb].reshape(*self.realgrid)
        else:
            
            # Get orbital info
            spin = self.wfc.iorb_sb_map[iorb][0]
            if spin == "up":
                ispin = 0
            elif spin == "down":
                ispin = 1
            iband = self.wfc.iorb_sb_map[iorb][1]
            
            # Get pseudo WF
            psir = self.wfc.calc_gpaw.get_pseudo_wave_function(band=iband, spin=ispin) # Units 1/Angstrom^(3/2), https://gpaw.readthedocs.io/devel/paw.html#gpaw.calculator.GPAW.get_pseudo_wave_function
            psir *= bohr_to_angstrom**(3./2) # Convert units from 1/Angstrom^(3/2) to 1/bohr^(3/2)
            psir = self.wfc.normalize(psir) # Normalize
            return psir

    def get_rhog_gpaw(self, iorb):
        return None


