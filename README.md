# Variational Quantum Eigensolver for SU(N) Fermions
### Code

We present the code used in the paper Variational Quantum Eigensolver for SU(N) Fermions, where a VQE is used to recover the ground state of the Fermi-Hubbard model for arbitrary spin SU(`N`) and number of sites `L` using an extended Jordan-Wigner mapping into `N*L` qubits.

In order to execute the following code, the library `Qibo` (available in  and enhanced by ) is needed for the quantum simulation of the VQE and the library Quspin (available in ) is required for the exact diagonalization.

#### Files
---

- `main.py`: main program.

- `functions.py`: quantum primitives used for the VQE implementation in Qibo language.

- `quspin_functions.py`: functions used for the exact diagonalization in Quspin language.

- `/data_vqe`: folder where the data will be saved in. Example of data used in entropy plots.

---

#### How to run the program

In order to recover the results for the VQE for SU(N) fermions, the file `main.py` can be run with the following arguments:

**Arguments**

- `--N` (int): SU(N) components of the fermions. (default = 3)
- `--L` (int): number of sites of the instance. (default = 3)
- `--t` (float): constant for the hopping terms. (default = 1.0)
- `--U` (float): on-site interaction. (default = 1.0)
- `--V` (float): interaction between fermions at distance r=1. (default = 0.0)
- `--r` (int): range of the interaction. (default = 1)
- `--layers` (int): number of maximum layers of the ansatz for the VQE implementation. (default = 3)
- `--phi_max` (float): maximum flux value. (default = 0.5)
- `--phi_num` (int): number of flux values to take into account from 0 to phi_max. (default = 25)
- `--backend` (str): qibo backend to use for the computation. (default = "qibotf")
- `--exact`:  add this flag to get the values using exact diagonalization. 
- `--perturb`: add this flag to do a small perturbation to the last best parameters before the next execution.

**Returns**

Text files containing the computed values for the `ENERGY`, `CURRENT` and `ENTROPY` for the Variational Quantum Eigensolver as well as the final optimized `PARAMETERS`. For the exact diagonalization, the `EXACT_ENERGY` and `EXACT_STATE` are returned in a text file as well.