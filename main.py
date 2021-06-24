#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:26:37 2021

@author: Carlos Bravo-Prieto & Sergi Ramos-Calderer
"""
import argparse
import numpy as np
from qibo import set_backend, models
import functions as fun


def main(N, Np, L, t, U, V, mu, layers, phi_max, phi_num, backend, exact, perturb, open_chain):
    """Find possible ground states via VQE of the Extended Fermi Hubbard Hamiltonian and
    get the corresponding current.
    
    Args:
        N (int): SU(N) components of the fermions.
        Np (list of ints): number of spins per color.
        L (int): number of sites of the instance.
        t (float or list(float)): constant for the hopping terms.
        U (float): on-site interaction.
        V (float or list(float)): interaction between fermions at distance > 0.
        mu (float): chemical potential.
        layers (int): number of layers of the ansatz for the VQE implementation.
        phi_max (float): maximum flux value.
        phi_num (int): number of flux values to take into account from 0 to phi_max.
        backend (str): qibo backend to use for the computation.
        exact (bool): flag to run get the values using exact diagonalization.
        perturb (bool): add a small perturbation to the last best parameters before the
                        next execution.
        open_chain (bool): open chain.
        
    Returns:
        Text files that contain the relevant information, exact diagonalization, energy and
        current from the VQE implementation as well as the entanglement entropy and optimized
        parameters.

    """
    
    set_backend(backend)
    phi_list = np.linspace(0, phi_max, phi_num)
    closed = not open_chain
    if Np == -1:
        Np = [1 for _ in range(N)]
    if exact:  # Calculate exact quantities and save them
        import quspin_functions as qfun
        energies = []
        states = []
        persistent_currents = []
        for phi in phi_list:
            energy, state = qfun.exact_eigenstates(N, L, Np, closed, t, V, U, mu, phi)
            persistent_current = qfun.exact_pc(N, L, Np, closed, t, phi)
            energies.append(energy[0])
            states.append(state[0])
            persistent_currents.append(persistent_current)
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_EXACT_ENERGY",
                       energies, delimiter=", ", newline="\n")
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_EXACT_STATE",
                       states, delimiter=", ", newline="\n")
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_EXACT_CURRENT",
                       persistent_currents, delimiter=", ", newline="\n")
    
    for l in range(1, layers + 1):
        circuit = fun.create_circuit(N, Np, L, l)
        initial_parameters = np.random.uniform(0, 4 * np.pi, len(circuit.get_parameters()))
        persistent_currents = []
        energies = []
        entropies = []
        parameters = []
        print("\n")
        print("Layers: ", l)
        for phi in phi_list:
            print("Flux: ", phi)
            hamiltonian_jw = fun.FH_hamiltonian(N, L, closed, t, U, V, mu, phi)
            vqe = models.VQE(circuit, hamiltonian_jw)
            best, params, _ = vqe.minimize(initial_parameters, method='BFGS')
            print("Energy: ", best, "\n")
            hamiltonian_persistent_current = fun.current_hamiltonian(N, L, closed, t, phi)
            persistent_current = fun.compute_persistent_current(
                params, circuit, hamiltonian_persistent_current)
            entropy_half = fun.entropy_half_chain(params, N, Np, L, l)
            
            energies.append(best)
            persistent_currents.append(persistent_current)
            parameters.append(params)
            entropies.append(entropy_half)
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_{l}_layers_CURRENT",
                       persistent_currents, delimiter=", ", newline="\n")
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_{l}_layers_ENERGY",
                       energies, delimiter=", ", newline="\n")
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_{l}_layers_ENTROPY",
                       entropies, delimiter=", ", newline="\n")
            np.savetxt(f"data_vqe/{N}_{L}_{Np}_{U}_{V}_Hubbard_{l}_layers_PARAMETERS",
                       parameters, delimiter=", ", newline="\n")
            
            if perturb:
                initial_parameters = 0.001 * np.pi * np.random.normal(size=len(params)) + params
            else:
                initial_parameters = params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=3, type=int)
    parser.add_argument("--Np", default=-1, nargs='+', type=int)
    parser.add_argument("--L", default=3, type=int)
    parser.add_argument("--t", default=1, nargs='+', type=float)
    parser.add_argument("--U", default=1, type=float)
    parser.add_argument("--V", default=0, nargs='+', type=float)
    parser.add_argument("--mu", default=0, type=float)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--phi_max", default=0.5, type=float)
    parser.add_argument("--phi_num", default=25, type=int)
    parser.add_argument("--backend", default="qibotf", type=str)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--open_chain", action="store_true")
    args = vars(parser.parse_args())
    main(**args)
