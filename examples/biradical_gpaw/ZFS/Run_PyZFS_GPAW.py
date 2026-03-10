
from pyzfs.common.wfc.gpawloader import GPAWWavefunctionLoader
from pyzfs.zfs.main import ZFSCalculation

# Construct wavefunction loader
wfcloader = GPAWWavefunctionLoader(gpwfile="../C3H6.gpw", ae=False)

# Construct ZFSCalculation
zfscalc = ZFSCalculation(wfcloader=wfcloader)

# Perform ZFS calculation
zfscalc.solve()


