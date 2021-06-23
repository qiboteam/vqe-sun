#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:26:37 2021

@author: Carlos Bravo-Prieto & Sergi Ramos-Calderer
"""
import numpy as np
from qibo import gates, models, callbacks
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z


def _helper(N=None, L=None, closed=None, t=None, U=None, V=None, mu=None, phi=None):
    """Hamiltonain helper function.
        Args:
            N (int): SU(N) components of the fermions.
            L (int): number of sites of the instance.
            closed (bool): closed chain.
            t (float or list(floats)): constant for the hopping terms.
            U (float): on-site interaction.
            V (float or list(floats)): interaction between fermions at distance > 0.
            mu (float): chemical potential.
            phi (float): flux.

        Returns:
            parameters (list): list of parameters specified by the caller

        """
    
    if not isinstance(t, list) and not isinstance(t, tuple):
        t_list = [t]
    else:
        t_list = t
    
    nqubits = N * L
    theta = 2 * np.pi * phi / L
    
    if V is not None:
        if not isinstance(V, list) and not isinstance(V, tuple):
            V_list = [V]
        else:
            V_list = V
        
        if U is not None and mu is not None:
            if closed:
                fV = np.sum([V_ * (1 if i + 1 < L / 2 else 1 / 2) for i, V_ in enumerate(
                    V_list)])
            else:
                fV = np.sum([V_ * (1 - i / L) for i, V_ in enumerate(V_list)])
            C = N * L * (N * fV / 2 + (N - 1) * U / 4 + mu) / 2
            
            return nqubits, t_list, V_list, theta, fV, C
        else:
            return nqubits, t_list, V_list, theta
    elif phi is not None:
        return nqubits, t_list, theta
    else:
        raise RuntimeError("Invalid Hamiltonian helper function called")


def FH_hamiltonian(N, L, closed, t, U, V, mu, phi):
    """Creates the Hamiltonian used as for the cost function of the variational algorithm.
    Args:
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        closed (bool): closed chain.
        t (float or list(floats)): constant for the hopping terms.
        U (float): on-site interaction.
        V (float or list(floats)): interaction between fermions at distance > 0.
        mu (float): chemical potential
        phi (float): flux.

    Returns:
        hamiltonian (qibo.hamiltonians): Hamiltonian for the EFM model for SU(N) in L sites.

    """
    
    nqubits, t_list, V_list, theta, fV, C = _helper(N, L, closed, t, U, V, mu, phi)
    
    # Define Hamiltonian using these symbols
    def sigma_plus(k):
        return 0.5 * X(k) + 0.5j * Y(k)
    
    def sigma_minus(k):
        return 0.5 * X(k) - 0.5j * Y(k)
    
    hamiltonian = C  # Identity term
    
    # Hopping terms
    for r, t_ in enumerate(t_list):
        for i in range(L if closed else L - r - 1):
            for j in range(N):
                q1 = i % L + j * L
                q2 = (i + r + 1) % L + j * L
                
                hamiltonian += -t_ * (np.exp(1j * theta) * sigma_plus(q1) * sigma_minus(q2) +
                                      np.exp(-1j * theta) * sigma_minus(q1) * sigma_plus(q2)) * \
                               np.prod([Z(k) for k in range(min(q1, q2) + 1, max(q1, q2))])
    
    # Coulomb terms
    for r, V_ in enumerate(V_list):
        if V_ != 0:
            for i in range(L if closed else L - r - 1):
                for j in range(N):
                    q2 = (i + r + 1) % L + j * L
                    for k in range(N):
                        q1 = i + k * L
                        
                        ZZ = Z(q1) * Z(q2)
                        
                        hamiltonian += V_ / 4 * ZZ
    
    # Onsite interaction terms
    if U != 0:
        for i in range(L):
            for j in range(1, N):
                for k in range(j):
                    q1 = i + k * L
                    q2 = i + j * L
                    
                    ZZ = Z(q1) * Z(q2)
                    
                    hamiltonian += U / 4 * ZZ
    
    # Onsite terms
    h = -(N * fV + (N - 1) * U / 2 + mu) / 2
    if h != 0:
        for i in range(L):
            for j in range(N):
                Z_ = Z(i + j * L)
                
                hamiltonian += h * Z_
    
    return SymbolicHamiltonian(hamiltonian)


def create_circuit(N, Np, L, layers):
    """Creates the variational ansatz to solve the problem.
    Args:
        N (int): SU(N) components of the fermions.
        Np (list of ints): number of spins per color.
        L (int): number of sites of the instance.
        layers (int): number of layers of the ansatz to add to the circuit.

    Returns:
        circuit (qibo.models.Circuit): Circuit with the required gates to be parameterized.

    """
    nqubits = N * L
    # Variational quantum circuit
    circuit = models.Circuit(nqubits)
    
    # Creates a state of the relevant basis
    for i, n in enumerate(Np):
        for j in range(n):
            circuit.add(gates.X(j + i * L))
            circuit.add(gates.RZ(j + i * L, theta=0))
    # n layers
    for l in range(layers):
        for s in range(N):
            for i in range(L - 1):
                circuit.add(gates.CNOT(i + s * L, i + s * L + 1))
                circuit.add(gates.CRX(i + s * L + 1, i + s * L, theta=0))
                circuit.add(gates.CNOT(i + s * L, i + s * L + 1))
        for i in range(N - 1):
            for s in range(L):
                circuit.add(gates.CRZ(s + L * i, s + L * (i + 1), theta=0))
        
        # Last layer of rotations
        for s in range(N):
            for i in range(L):
                circuit.add(gates.RZ(i + s * L, theta=0))
    
    return circuit


def entropy_half_chain(params, N, Np, L, layers):
    """Compute the entanglement entropy of half the qubit chain.
    Args:
        params (np.array): trained parameters from the VQE algorithm.
        N (int): SU(N) components of the fermions.
        Np (list of ints): number of spins per color.
        L (int): number of sites of the instance.
        layers (int): number of layers of the ansatz to add to the circuit.
    
    Returns:
        entropy (float): value of the half chain entropy.
    """
    
    entropy = callbacks.EntanglementEntropy()
    circuit = create_circuit(N, Np, L, layers)
    circuit.add(gates.CallbackGate(entropy))
    circuit.set_parameters(params)
    circuit.execute()
    return entropy[0].numpy()


def current_hamiltonian(N, L, closed, t, phi):
    """Construct Hamiltonian for the current calculation.
    Args:
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        closed (bool): closed chain.
        t (float or list(float)): constant for the hopping terms.
        phi (float): flux.
    
    Returns:
        hamiltonian (qibo.hamiltonians): Hamiltonian for the current EFM model for SU(N) in L sites.

    """
    
    n, t_list, theta = _helper(N, L, closed, t, phi=phi)
    
    # Define Hamiltonian using these symbols
    def sigma_plus(k):
        return 0.5 * X(k) + 0.5j * Y(k)
    
    def sigma_minus(k):
        return 0.5 * X(k) - 0.5j * Y(k)
    
    hamiltonian = 0
    for r, t_ in enumerate(t_list):
        for i in range(L if closed else L - r - 1):
            for j in range(N):
                q1 = i % L + j * L
                q2 = (i + r + 1) % L + j * L
                
                hamiltonian += 2j * np.pi * t_ * (np.exp(1j * theta) * sigma_plus(q1) *
                                                  sigma_minus(q2) - np.exp(-1j * theta) *
                                                  sigma_minus(q1) * sigma_plus(q2)) / L * \
                               np.prod([Z(k) for k in range(min(q1, q2) + 1, max(q1, q2))])
    
    return SymbolicHamiltonian(hamiltonian)


def compute_persistent_current(params, circuit, hamiltonian):
    """Computation of the expected value of the current for the solution of the VQE algorithm.
    Args:
        params (np.array): trained parameters from the VQE algorithm.
        circuit (qibo.models.Circuit): VQE circuit with paramtererized gates.
        hamiltonian (qibo.hamiltonians): Hamiltonian for the current EFM model for SU(N) in L sites.
        
    Returns:
        current (float): expectaction value of the current for the state found by the VQE.
    
    """
    circuit.set_parameters(params)
    current = hamiltonian.expectation(circuit.execute()).numpy().real
    return current
