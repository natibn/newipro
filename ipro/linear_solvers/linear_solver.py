import numpy as np
from abc import ABC, abstractmethod
from oracles.ppo_oracle import PPOOracle

class LinearSolver(ABC):
    @abstractmethod
    def solve(self, weight_vector: np.ndarray):
        pass


class DummyLinearSolver(LinearSolver):
    """Solver para testes rápidos — apenas retorna o vetor de peso como resultado."""
    def solve(self, weight_vector: np.ndarray):
        return weight_vector, None


class PPOBasedSolver(LinearSolver):
    """Solver que usa o PPOOracle para resolver o subproblema com vetor de pesos."""
    def __init__(self, oracle: PPOOracle):
        self.oracle = oracle

    def solve(self, weight_vector: np.ndarray):
        return self.oracle.solve(weight_vector)
