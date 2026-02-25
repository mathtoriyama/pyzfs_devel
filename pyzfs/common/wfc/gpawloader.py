from __future__ import absolute_import, division, print_function
import numpy as np
import os, sys
from mpi4py import MPI
from scipy.interpolate import interpn

from .baseloader import WavefunctionLoader
from ..cell import Cell
from ..ft import FourierTransform
from .wavefunction import Wavefunction
from ..counter import Counter
from ..parallel import SymmetricDistributedMatrix

from ...common.misc import empty_ase_cell
from ..units import bohr_to_angstrom

from gpaw import GPAW
from gpaw.old.pw.descriptor import PWDescriptor
from gpaw.mpi import serial_comm
from gpaw.utilities.ps2ae import PS2AE

def _compute_offset(sdm, iorb):
    """compute the index for iorb^th wfc, note that some rows in psig_arrs_all
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

    def __init__(self, gpwfile, ae=False, comm=MPI.COMM_WORLD):
        self.gpwfile = gpwfile
        self.ae = ae
        super(GPAWWavefunctionLoader, self).__init__()

    def scan(self):
        super(GPAWWavefunctionLoader, self).scan()
        
        # Load GPAW calculator
        self.calc_gpaw = GPAW(  self.gpwfile, 
                                communicator = serial_comm, 
                                )
        wfs = self.calc_gpaw.wfs
        self.calc_gpaw_ps2ae = PS2AE(self.calc_gpaw)

        # Parse cell
        cell = Cell(empty_ase_cell(
            *self.calc_gpaw.atoms.get_cell().array.T,
            unit="angstrom"
        ))

        # Create FT objects
        if self.ae:
            grid = self.calc_gpaw_ps2ae.gd.N_c
        else:
            grid = self.calc_gpaw.wfs.gd.N_c
        self.wft = FourierTransform(grid[0], grid[1], grid[2])
        self.dft = FourierTransform(grid[0], grid[1], grid[2])
                
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

        # G-vectors
        self.gvecs = self.calc_gpaw.wfs.pd.get_reciprocal_vectors(add_q=False) # Splits Gvecs in MPI, but want all processes to have all Gvecs
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


        # Set indicator of GPAW calc
        self.wfc.gpaw = True
        self.wfc.pd = self.calc_gpaw.wfs.pd

        """
        # Setting for all-electron calc
        if self.ae:

            # PS to AE reconstruction object
            from gpaw.utilities.ps2ae import PS2AE
            self.calc_gpaw_ps2ae = PS2AE(self.calc_gpaw)

            # AE grid
            gd_ae = self.calc_gpaw_ps2ae.gd
            self.coords_ae = (gd_ae.coords(0), gd_ae.coords(1), gd_ae.coords(2))
            
            # PS grid
            gd_ps = self.calc_gpaw.wfs.gd
            coords_ps = gd_ps.get_grid_point_coordinates()
            self.coords_ps_t = (coords_ps[0], coords_ps[1], coords_ps[2])


        # Set all wave functions
        for iorb in range(len(self.wfc.iorb_sb_map)):
            self.set_psir_gpaw(iorb)
        """


        # Add function to self.wfc object
        self.wfc.get_psir_gpaw = self.get_psir_gpaw



    def load(self, iorbs, sdm):
        super(GPAWWavefunctionLoader, self).load(iorbs, sdm)
        assert isinstance(sdm, SymmetricDistributedMatrix)
        comm = sdm.comm
        rank = sdm.pgrid.rank
        onroot = sdm.onroot

        # processor 0 parse wavefunctions
        psig_arrs_all = None
        ngvecs = self.wfc.gvecs.shape[0]
        if onroot:
            psig_arrs_all = np.zeros([sdm.mx, ngvecs], dtype=complex)
            c = Counter(
                self.wfc.norbs,
                percent=0.1,
                message="(process 0) {n} orbitals ({percent}%) loaded in {dt}...",
            )

            nbands = self.calc_gpaw.get_number_of_bands()
            for ispin in range(2):
                for iband in range(nbands):
                    psir = self.calc_gpaw.get_pseudo_wave_function(band=iband, spin=ispin)  # Units 1/Angstrom^(3/2), https://gpaw.readthedocs.io/devel/paw.html#gpaw.calculator.GPAW.get_pseudo_wave_function
                    psig = self.calc_gpaw.wfs.pd.fft(psir)
                    iorb = self.wfc.sb_iorb_map.get(
                        ("up" if ispin == 0 else "down", iband)
                    )
                    if iorb is not None:
                        offset = _compute_offset(sdm, iorb)
                        psig_arrs_all[offset] = psig
                        c.count()

        # scatter wavefunctions
        # allocate wfc arrays
        psig_arrs_m = np.zeros([sdm.mlocx, ngvecs], dtype=complex)
        psig_arrs_n = np.zeros([sdm.nlocx, ngvecs], dtype=complex)
        comm.barrier()

        # root -> first column scatter
        if onroot:
            print("QEHDF5WavefunctionLoader: root -> first column scattering")
        if sdm.icol == 0:
            sdm.colcomm.Scatter(sendbuf=psig_arrs_all, recvbuf=psig_arrs_m, root=0)
        comm.barrier()

        # first column -> other column bcast
        if onroot:
            print("QEHDF5WavefunctionLoader: first column -> other column bcast")
        sdm.rowcomm.Bcast(psig_arrs_m, root=0)
        comm.barrier()

        # root -> first row scatter
        if onroot:
            print("QEHDF5WavefunctionLoader: root -> first row scattering")
        if sdm.irow == 0:
            sdm.rowcomm.Scatter(sendbuf=psig_arrs_all, recvbuf=psig_arrs_n, root=0)
        comm.barrier()

        # first row -> other row bcast
        if onroot:
            print("QEHDF5WavefunctionLoader: first row -> other row bcast")
        sdm.colcomm.Bcast(psig_arrs_n, root=0)
        comm.barrier()

        if onroot:
            del psig_arrs_all

        for iloc in range(sdm.mloc):
            iorb = sdm.ltog(iloc)
            self.wfc.set_psig_arr(iorb, psig_arrs_m[iloc])

        for iloc in range(sdm.nloc):
            iorb = sdm.ltog(0, iloc)[1]
            try:
                self.wfc.set_psig_arr(iorb, psig_arrs_n[iloc])
            except ValueError:
                pass
        comm.barrier()

        if self.memory == "high":
            self.wfc.compute_all_psir()
            self.wfc.clear_all_psig_arr()
            self.wfc.compute_all_rhog()
        elif self.memory == "low":
            self.wfc.compute_all_psir()
            self.wfc.clear_all_psig_arr()
        elif self.memory == "critical":
            pass
        else:
            raise ValueError


    def set_psir_gpaw_Old(self, iorb):
        """Set psi(r) for each iorb index, GPAW edition"""

        spin = self.wfc.iorb_sb_map[iorb][0]
        if spin == "up":
            ispin = 0
        elif spin == "down":
            ispin = 1
        iband = self.wfc.iorb_sb_map[iorb][1]

        if not self.ae:
            psir = self.calc_gpaw.get_pseudo_wave_function(band=iband, spin=ispin)  # Units 1/Angstrom^(3/2), https://gpaw.readthedocs.io/devel/paw.html#gpaw.calculator.GPAW.get_pseudo_wave_function
            """
            wfs = self.calc_gpaw_gpaw.wfs
            local_gs = wfs.gd.n_c
            gpaw_wf = wfs.pd.ifft(wfs.kpt_u[ispin].psit_nG[iorb])  # Domain-decomposed local wave function for MPI process
            assert np.all(gpaw_wf.shape == local_gs), "Shapes are wrong... Exiting."
            """

        else:
            # Get all-electron wave function
            psir_ae = self.calc_gpaw_ps2ae.get_wave_function(n=iband, s=ispin, ae=True)

            # Coarsen all-electron WF to same grid as pseudo WF
            psir = interpn(self.coords_ae, psir_ae, self.coords_ps_t)

        psir *= bohr_to_angstrom**(3./2)  # Convert units from 1/Angstrom^(3/2) to 1/bohr^(3/2)
        #self.wfc.iorb_psir_map[iorb] = psir

        # Convert to reciprocal space (save space)
        psig = self.wfc.pd.fft(psir)
        del psir
        self.wfc.iorb_psig_arr_map[iorb] = psig



    def get_psir_gpaw_Old(self, iorb):
        """Get psi(r) of certain index, GPAW edition"""
        #return self.wfc.iorb_psir_map[iorb]

        return self.wfc.pd.ifft( self.wfc.iorb_psig_arr_map[iorb] )




    def get_psir_gpaw(self, iorb):

        spin = self.wfc.iorb_sb_map[iorb][0]
        if spin == "up":
            ispin = 0
        elif spin == "down":
            ispin = 1
        iband = self.wfc.iorb_sb_map[iorb][1]

        if not self.ae:
            psir = self.calc_gpaw.get_pseudo_wave_function(band=iband, spin=ispin)  # Units 1/Angstrom^(3/2), https://gpaw.readthedocs.io/devel/paw.html#gpaw.calculator.GPAW.get_pseudo_wave_function
            """
            wfs = self.calc_gpaw_gpaw.wfs
            local_gs = wfs.gd.n_c
            gpaw_wf = wfs.pd.ifft(wfs.kpt_u[ispin].psit_nG[iorb])  # Domain-decomposed local wave function for MPI process
            assert np.all(gpaw_wf.shape == local_gs), "Shapes are wrong... Exiting."
            """

        else:
            # Get all-electron wave function
            psir = self.calc_gpaw_ps2ae.get_wave_function(n=iband, s=ispin, ae=True)

        psir *= bohr_to_angstrom**(3./2)  # Convert units from 1/Angstrom^(3/2) to 1/bohr^(3/2)

        return psir


