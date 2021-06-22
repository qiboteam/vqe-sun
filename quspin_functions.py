#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mirko Consiglio
"""

import copy

import numpy as np
from quspin.basis import spinless_fermion_basis_1d, spin_basis_1d, tensor_basis
from quspin.operators import hamiltonian


def generate_fermion_basis(L, Np):
	return tensor_basis(*(spinless_fermion_basis_1d(L, Nf=N) for N in Np))


def generate_fermion_operators(N, L, closed=True, t=None, V=None, U=1., mu=0., phi=0.):
	if t is None:
		t_ = [1]
	elif not isinstance(t, list) and not isinstance(t, tuple):
		t_ = [t]
	else:
		t_ = t
	r_t = len(t_)
	if V is None:
		V_ = [0]
	elif not isinstance(V, list) and not isinstance(V, tuple):
		V_ = [V]
	else:
		V_ = V
	r_V = len(V_)
	
	ops = []
	# Hopping terms
	phase = 1j * 2 * np.pi * phi / L
	for r in range(r_t):
		if t_[r] != 0:
			if closed:
				J_pm = [[-t_[r] * np.exp(phase)] + [i, (i + r + 1) % L]
				        for i in range(L)]
				J_mp = [[t_[r] * np.exp(-phase)] + [i, (i + r + 1) % L]
				        for i in range(L)]
			else:
				J_pm = [[-t_[r] * np.exp(phase)] + [i, i + r + 1]
				        for i in range(L - r - 1)]
				J_mp = [[t_[r] * np.exp(-phase)] + [i, i + r + 1]
				        for i in range(L - r - 1)]
			for k in range(N):
				ops.append(['|' * (N - k - 1) + '+' + '-' + '|' * k, J_pm])
				ops.append(['|' * (N - k - 1) + '-' + '+' + '|' * k, J_mp])
	
	# Coulomb terms
	for r in range(r_V):
		if V_[r] != 0:
			if closed:
				J_nn = [[V_[r], i, (i + r + 1) % L] for i in range(L)]
				J_nn_c = [[V_[r], (i + r + 1) % L, i] for i in range(L)]
			else:
				J_nn = [[V_[r], i, i + r + 1] for i in range(L - r - 1)]
				J_nn_c = [[V_[r], i + r + 1, i] for i in range(L - r - 1)]
			for k in range(N):
				ops.append(['|' * (N - k - 1) + 'nn' + '|' * k, J_nn])
				for l in range(1, N - k):
					ops.append(['|' * (N - k - l - 1) + 'n' +
					            '|' * l + 'n' + '|' * k, J_nn + J_nn_c])
	
	# Onsite interaction terms
	if U != 0:
		J_nn = [[U, i, i] for i in range(L)]
		for j in range(N):
			for k in range(1, N - j):
				ops.append(['|' * (N - j - k - 1) + 'n' +
				            '|' * k + 'n' + '|' * j, J_nn])
	
	# Chemical potential
	if mu != 0:
		h_n = [[mu, i] for i in range(L)]
		for i in range(N):
			ops.append(['|' * (N - i - 1) + 'n' + '|' * i, h_n])
	
	return ops, 0


def generate_spin_basis(L, Np):
	return tensor_basis(*(spin_basis_1d(L, Nup=L - N, pauli=-1) for N in Np))


def generate_spin_operators(N, L, closed=True, t=None, V=None, U=1., mu=0., phi=0., Z_replace=False, Np=None):
	if t is None:
		t_ = [1]
	elif not isinstance(t, list) and not isinstance(t, tuple):
		t_ = [t]
	else:
		t_ = t
	r_t = len(t_)
	if V is None:
		V_ = [0]
	elif not isinstance(V, list) and not isinstance(V, tuple):
		V_ = [V]
	else:
		V_ = V
	r_V = len(V_)
	fV = np.sum([V_[i] * (1 if i + 1 < L / 2 else 0.5) for i in range(r_V)])
	
	ops = []
	# Hopping terms
	phase = 1j * 2 * np.pi * phi / L
	for r in range(r_t):
		if t_[r] != 0:
			J_pm = [[-t_[r] * np.exp(phase)] + [i + j for j in range(r + 2)]
			        for i in range(L - 1 - r)]
			J_mp = [[-t_[r] * np.exp(-phase)] + [i + j for j in range(r + 2)]
			        for i in range(L - 1 - r)]
			for k in range(N):
				ops.append(['|' * (N - k - 1) + '+' +
				            'z' * r + '-' + '|' * k, J_pm])
				ops.append(['|' * (N - k - 1) + '-' +
				            'z' * r + '+' + '|' * k, J_mp])
			
			if closed and r + 1 < L / 2:
				J_pm = [
					[-t_[r] * np.exp(phase)] + [i + j for j in range(L - r)] for i in range(r + 1)]
				J_mp = [
					[-t_[r] * np.exp(-phase)] + [i + j for j in range(L - r)] for i in range(r + 1)]
				for k in range(N):
					if Z_replace and Np[-k - 1] % 2 == 0:
						J_pm_ = copy.deepcopy(J_pm)
						J_mp_ = copy.deepcopy(J_mp)
						for i in range(r + 1):
							J_pm_[i][0] *= -1
							J_mp_[i][0] *= -1
					else:
						J_pm_ = J_pm
						J_mp_ = J_mp
					ops.append(
						['|' * (N - k - 1) + '-' + ('I' if Z_replace else 'z') * (L - 2 - r) + '+' + '|' * k, J_pm_])
					ops.append(
						['|' * (N - k - 1) + '+' + ('I' if Z_replace else 'z') * (L - 2 - r) + '-' + '|' * k, J_mp_])
	
	# Coulomb terms
	for r in range(r_V):
		if V_[r] != 0:
			if closed:
				J_zz = [[V_[r] / 4, i, (i + r + 1) % L] for i in range(L)]
				J_zz_c = [[V_[r] / 4, (i + r + 1) % L, i] for i in range(L)]
			else:
				J_zz = [[V_[r] / 4, i, i + r + 1] for i in range(L - r - 1)]
				J_zz_c = [[V_[r] / 4, i + r + 1, i] for i in range(L - r - 1)]
			for k in range(N):
				ops.append(['|' * (N - k - 1) + 'zz' + '|' * k, J_zz])
				for l in range(1, N - k):
					ops.append(['|' * (N - k - l - 1) + 'z' +
					            '|' * l + 'z' + '|' * k, J_zz + J_zz_c])
	
	# Onsite interaction terms
	if U != 0:
		J_zz = [[U / 4, i, i] for i in range(L)]
		for j in range(N):
			for k in range(1, N - j):
				ops.append(['|' * (N - j - k - 1) + 'z' +
				            '|' * k + 'z' + '|' * j, J_zz])
	
	# Onsite terms
	h = -(N * fV + (N - 1) * U / 2 + mu) / 2
	if h != 0:
		h_z = [[h, i] for i in range(L)]
		for i in range(N):
			ops.append(['|' * (N - i - 1) + 'z' + '|' * i, h_z])
	
	# Constant
	C = N * L * (N * fV / 2 + (N - 1) * U / 4 + mu) / 2
	
	return ops, C


def fidelity(s1, s2):
	return abs(np.vdot(s1, s2)) ** 2


# Helper function to calculate energies and eigenstates
def exact_eigenstates(N, L, Np, closed, t, V, U, mu, phi, k=1, l=0):
	basis = generate_spin_basis(L, Np)
	
	static, C = generate_spin_operators(N, L, closed, t, V, U, mu, phi)
	
	no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
	H = hamiltonian(static, [], N=L * len(Np), basis=basis, **no_checks)
	
	try:  # k is the number of eigenstates to get, l is used for obtaining more in case of errors
		w, v = H.eigsh(k=k + l, which='SA')
	except:
		w, v = H.eigh()
	eigenstates = np.array([i for _, i in sorted(
		zip(w, np.transpose(v)), key=lambda x: x[0])])
	energies = sorted(w) + C
	
	return energies[:k], eigenstates[:k]
