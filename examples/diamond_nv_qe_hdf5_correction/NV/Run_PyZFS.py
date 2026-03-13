
from pyzfs.common.wfc.qeh5loader import QEHDF5WavefunctionLoader
from pyzfs.zfs.main import ZFSCalculation

# Construct wavefunction loader
wfcloader = QEHDF5WavefunctionLoader(   fftgrid='wave',
                                        prefix='pwscf')

# Construct ZFSCalculation
zfscalc = ZFSCalculation(wfcloader=wfcloader)

# Perform ZFS calculation
zfscalc.solve()



