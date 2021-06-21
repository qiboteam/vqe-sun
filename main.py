#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:26:37 2021

@author: Carlos Bravo-Prieto & Sergi Ramos-Calderer
"""
import numpy as np
import functions as fun
from qibo import set_backend, models
from scipy.optimize import minimize
import argparse


def main(N, L, t, U, V, r, layers, phi_max, phi_num, backend, exact, perturb):
    """Find possible ground states via VQE of the Extended Fermi Hubbard Hamiltonian and 
    get the corresponding current.
    
    Args:
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        t (float): constant for the hopping terms.
        U (float): on-site interaction.
        V (float): interaction between fermions at distance r=1.
        r (int): range of the interaction.
        layers (int): number of layers of the ansatz for the VQE implementation.
        phi_max (float): maximum flux value.
        phi_num (int): number of flux values to take into account from 0 to phi_max.
        backend (str): qibo backend to use for the computation.
        exact (bool): flag to run get the values using exact diagonalization.
        perturb (bool): add a small perturbation to the last best parameters before the
                        next execution.
        
    Returns:
        Text files that contain the relevant information, exact diagonalization, energy and 
        current from the VQE implementation as well as the entanglement entropy and optimized 
        parameters.

    """
    set_backend(backend)
    phi = np.linspace(0, phi_max, phi_num)
    if exact:
        import quspin_functions as qfun
        exact = []
        closed = True
        Np = (1 for _ in range(N))
        mu = 0
        for i in phi:
            energy, state = qfun.f(N, L, Np, closed, t, V, U, mu, phi)
            exact.append(energy)
        np.savetxt(f"data_vqe/{N}_{L}_Hubbard_exact_U_{U}_V_{V}_r_{r}_(1,1,1)_EXACT_ENERGY",
               [exact], delimiter=", ", newline="\n")
        np.savetxt(f"data_vqe/{N}_{L}_Hubbard_exact_U_{U}_V_{V}_r_{r}_(1,1,1)_EXACT_STATE",
               state, delimiter=", ", newline="\n")
    
    for l in range(1, layers+1):
        initial_parameters = np.random.uniform(0, 2*np.pi,
                                l*((L-1)*N+(N-1)*L+N*L)+N)
        persistent_current_vector = []
        energy_vector = []
        entropies = []
        parameters = []
        circuit = fun.create_circuit(N, L, l)
        print("\n")
        print("Layers: ", l)
        for i in phi:
            print("Flux: ", i)
            hamiltonian_jw = fun.FH_hamiltonian(
                N=N, L=L, t=t, U=U, V=V, phi=i, r=r)
            vqe = models.VQE(circuit, hamiltonian_jw)
            best, params, _ = vqe.minimize(initial_parameters, method='BFGS', compile=False)
            print("Energy: ", best, "\n")
            energy_vector.append(best)
            parameters.append(params)
            hamiltonian_persistent_current = fun.current_hamiltonian(
                N=N, L=L, t=t, phi=i, r=r)
            persistent_current = fun.compute_persistent_current(
                params, circuit, hamiltonian_persistent_current)
            persistent_current_vector.append(persistent_current)
            entropy_half = fun.entropy_half_chain(
                params, N, L, l)
            entropies.append(entropy_half)
            if perturb:
                initial_parameters = 0.001*np.pi*(-1+2*np.random.random(len(params)))+params
            else:
                initial_parameters = params
            if i == 0.5:
                initial_parameters = np.random.uniform(0, 2*np.pi, l*((L-1)*N+(N-1)*L+N*L)+N)
        np.savetxt(f"data_22/{N}_{L}_Hubbard_{l}_layers_U_{U}_V_{V}_r_{r}_(1,1,1)_CURRENT",
                   [persistent_current_vector], delimiter=", ", newline="\n")
        np.savetxt(f"data_22/{N}_{L}_Hubbard_{l}_layers_U_{U}_V_{V}_r_{r}_(1,1,1)_ENERGY",
                   [energy_vector], delimiter=", ", newline="\n")
        np.savetxt(f"data_22/{N}_{L}_Hubbard_{l}_layers_U_{U}_V_{V}_r_{r}_(1,1,1)_ENTROPY",
                   [entropies], delimiter=", ", newline="\n")
        np.savetxt(f"data_22/{N}_{L}_Hubbard_{l}_layers_U_{U}_V_{V}_r_{r}_(1,1,1)_PARAMETERS",
                   parameters, delimiter=", ", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=3, type=int)
    parser.add_argument("--L", default=3, type=int)
    parser.add_argument("--t", default=1, type=float)
    parser.add_argument("--U", default=1, type=float)
    parser.add_argument("--V", default=0, type=float)
    parser.add_argument("--r", default=1, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--phi_max", default=0.5, type=float)
    parser.add_argument("--phi_num", default=25, type=int)
    parser.add_argument("--backend", default="qibotf", type=str)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    args = vars(parser.parse_args())
    main(**args)