
from ase.build import bulk
from gpaw import GPAW, PW

def nv_minus(n: int) -> None:
    atoms = bulk('C', 'diamond', cubic=True) * n
    atoms.numbers[0] = 7
    del atoms[1]
    atoms.set_initial_magnetic_moments([1] * len(atoms))
    name = f'NC{8 * n**3 - 2}'
    atoms.calc = GPAW(xc='PBE',
                      mode=PW(ecut=300),
                      charge=-1,
                      parallel={'kpt': 1},
                      txt=name + '.txt')
    atoms.get_forces()  # Static calc
    atoms.calc.write(name + '.gpw', 'all')

if __name__ == '__main__':
    n = 2  # Create a 2x2x2 supercell
    nv_minus(n)

