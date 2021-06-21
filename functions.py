#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:26:37 2021

@author: Carlos Bravo-Prieto & Sergi Ramos-Calderer
"""
import numpy as np
from qibo import gates, hamiltonians, models, matrices, callbacks
import sympy
from qibo.symbols import X, Y, Z
from qibo.hamiltonians import SymbolicHamiltonian


def FH_hamiltonian(N, L, t, U, V, phi, r):
    """Creates the Hamiltonian used as for the cost function of the variational algorithm.
    Args:
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        t (float): constant for the hopping terms.
        U (float): on-site interaction.
        V (float): interaction between fermions at distance r=1.
        phi (float): flux.
        r (int): range of the interaction.

    Returns:
        hamiltonian (qibo.hamiltonians): Hamiltonian for the EFM model for SU(N) in L sites.

    """
    nqubits = N*L

    def sigma_plus(i):
        return 0.5*X(i) + 0.5j*Y(i)
    def sigma_minus(i):
        return 0.5*X(i) - 0.5j*Y(i)
    def operator(i):
        return (1-Z(i))/2

    # Define the Hamiltonian
    symbolic_ham = 0
    for s in range(N):
        for i in range(L):
            for rr in range(1, r+1):
                temp1 = (np.exp(1j*phi*2*np.pi/L)*sigma_plus(i+s*L)*sigma_minus((
                    i+rr) % L+s*L) + np.exp(-1j*phi*2*np.pi/L)*sigma_minus(i+s*L)*sigma_plus((i+rr) % L+s*L))
                for j in range(min(i, (i+rr) % L)+1, max(i, (i+rr) % L)):
                    temp1 *= Z(j+s*L)
                symbolic_ham -= t*temp1
            for ss in range(N):
                for rr in range(1, r+1):
                    symbolic_ham += 0.25*V*Z(i+s*L)*Z((i+rr) % L+ss*L)
            for ss in range(s+1, N):
                symbolic_ham += 0.25*U*Z(i+s*L)*Z(i+ss*L)
            symbolic_ham -= 0.5*(N*V + 0.5*(N-1)*U)*Z(i+s*L)
    symbolic_ham += 0.25*nqubits*(N*V + 0.5*(N-1)*U)

    return SymbolicHamiltonian(symbolic_ham)


def create_circuit(N, L, layers):
    """Creates the variational ansatz to solve the problem.
    Args:
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        layers (int): number of layers of the ansatz to add to the circuit.

    Returns:
        circuit (qibo.models.Circuit): Circuit with the required gates to be parameterized.

    """
    nqubits = N*L
    # Variational quantum circuit
    circuit = models.Circuit(nqubits)

    # Creates a state of the relevant basis
    for s in range(N):
        circuit.add(gates.X(s*L))
        circuit.add(gates.RZ(s*L, theta=0))
    # n layers
    for l in range(layers):
        for s in range(N):
            for i in range(L-1):
                circuit.add(gates.CNOT(i+s*L, i+s*L+1))
                circuit.add(gates.CRX(i+s*L+1, i+s*L, theta=0))
                circuit.add(gates.CNOT(i+s*L, i+s*L+1))
        for i in range(N-1):
            for s in range(L):
                circuit.add(gates.CRZ(s+L*i, s+L*(i+1), theta=0))

    # Last layer of rotations
        for s in range(N):
            for i in range(L):
                circuit.add(gates.RZ(i+s*L, theta=0))

    return circuit


def entropy_half_chain(params, N, L, layers):
    """Compute the entanglement entropy of half the qubit chain.
    Args:
        params (np.array): trained parameters from the VQE algorithm.
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        layers (int): number of layers of the ansatz to add to the circuit.
    
    Returns:
        entropy (float): value of the half chain entropy.
    """
    nqubits = N*L
    entropy = callbacks.EntanglementEntropy(
        np.linspace(0, int(nqubits/2)-1, int(nqubits/2)))
    circuit = create_circuit(N, L, layers)
    circuit.add(gates.CallbackGate(entropy))
    circuit.set_parameters(params)
    final_state = circuit()
    return entropy[0].numpy()


def current_hamiltonian(N, L, t, phi, r):
    """Construct Hamiltonian for the current calculation.
    Args:
        N (int): SU(N) components of the fermions.
        L (int): number of sites of the instance.
        t (float): constant for the hopping terms.
        phi (float): flux.
        r (int): range of the interaction.
    
    Returns:
        hamiltonian (qibo.hamiltonians): Hamiltonian for the current EFM model for SU(N) in L sites.

    """
    nqubits = N*L

    def sigma_plus(i):
        return 0.5*X(i) + 0.5j*Y(i)
    def sigma_minus(i):
        return 0.5*X(i) - 0.5j*Y(i)

    # Define the Hamiltonian
    symbolic_ham = 0
    for s in range(N):
        for i in range(L):
            for rr in range(1, r+1):
                temp1 = (np.exp(1j*phi*2*np.pi/L)*sigma_plus(i+s*L)*sigma_minus((
                    i+rr) % L+s*L) - np.exp(-1j*phi*2*np.pi/L)*sigma_minus(i+s*L)*sigma_plus((i+rr) % L+s*L))
                for j in range(min(i, (i+rr) % L)+1, max(i, (i+rr) % L)):
                    temp1 *= Z(j+s*L)
                symbolic_ham += 1j*t*temp1*2*np.pi/L

    return SymbolicHamiltonian(symbolic_ham)


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