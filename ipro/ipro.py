import numpy as np


class IPRO:
    def __init__(self, problem_id, dimensions, oracle, objectives, linear_solver, direction, offset, tolerance, max_iterations, update_freq):
        self.problem_id = problem_id
        self.dimensions = dimensions
        self.oracle = oracle
        self.objectives = objectives
        self.linear_solver = linear_solver
        self.direction = direction
        self.offset = offset
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.update_freq = update_freq
        self.pf = []
        self.lower = []
        self.upper = []

    def init_phase(self):
        self.ideal = []
        self.nadir = []
        self.ideal_points = []
        self.nadir_points = []
        self.subproblems = []
        self.weights = []

        for i in range(len(self.objectives)):
            w = self.unit_vector(i)
            ideal_point, _ = self.linear_solver.solve(w)
            nadir_point, _ = self.linear_solver.solve(-1 * w)
            self.ideal.append(ideal_point[i])
            self.nadir.append(nadir_point[i])
            self.ideal_points.append(ideal_point)
            self.nadir_points.append(nadir_point)
            self.subproblems.append({"weight": w, "ideal": ideal_point, "nadir": nadir_point})
            self.weights.append(w)

        self.lower = [self.nadir]
        self.upper = [self.ideal]

        return {
            "ideal": self.ideal,
            "nadir": self.nadir,
            "ideal_points": self.ideal_points,
            "nadir_points": self.nadir_points,
            "lower": self.lower,
            "upper": self.upper,
            "subproblems": self.subproblems,
            "weights": self.weights
        }

    def unit_vector(self, i):
        vec = [0] * len(self.objectives)
        vec[i] = 1
        return np.array(vec)

    def step(self):
        # Escolhe ponto referencial (pode ser o último ponto da fronteira inferior)
        ref = self.lower[-1] if self.lower else self.ideal

        # Gera pesos normalizados para os objetivos
        weights = np.random.rand(len(self.objectives))
        weights /= np.sum(weights)

        # Consulta ao oráculo para obter solução Pareto
        point = self.oracle(ref, weights, self.direction, self.offset, self.tolerance)

        # Atualiza fronteira se o ponto for dominante
        if self.is_pareto(point):
            self.pf.append(point)
            self.lower.append(point)

    def is_dominant(self, point):
        # Critério menos restritivo: considera dominante se o ponto for igual ou melhor que o ideal em todos os objetivos
        return all(p >= i for p, i in zip(point, self.ideal))
    
    def dominates(self, a, b):
        # Retorna True se a domina b (maximização)
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    def is_pareto(self, point):
        # Remove pontos dominados pelo novo ponto
        self.pf = [p for p in self.pf if not self.dominates(point, p)]
        # Adiciona se não for dominado por nenhum ponto existente
        if not any(self.dominates(p, point) for p in self.pf):
            self.pf.append(point)
            return True
        return False
    
    def decompose_problem(self, iteration):
        subproblem = {
            "weight": self.weights[iteration],
            "ideal": self.ideal_points[iteration],
            "nadir": self.nadir_points[iteration]
        }
        return subproblem
    
    def maybe_add_completed(self, subproblem, point, weights):
        sp = subproblem[0] if isinstance(subproblem, list) else subproblem
        if self.is_pareto(point):
            self.pf.append(point)
            sp["completed"] = True
            sp["solution"] = point
            sp["weight"] = weights
            return sp
        return None

    def maybe_add_solution(self, subproblem, point):
        if self.is_pareto(point):
            subproblem["referent"] = point
            return subproblem
        else:
            return None
        
    def estimate_error(self):
        # Exemplo de cálculo de erro: diferença máxima entre pontos da fronteira inferior e ideal
        if not self.lower or not self.ideal:
            print("Não há pontos suficientes para estimar o erro.")
            return False
        # Considera o último ponto da fronteira inferior
        last_lower = self.lower[-1]
        print(f"Último ponto da fronteira inferior: {last_lower}")
        print(f"Ponto ideal: {self.ideal}")
        diff = np.abs(np.array(last_lower) - np.array(self.ideal))
        print(f"Diferença absoluta entre os pontos: {diff}")
        error = np.max(diff)
        print(f"Erro máximo estimado: {error}")
        convergiu = error <= self.tolerance
        print(f"Convergiu? {'Sim' if convergiu else 'Não'} (tolerância = {self.tolerance})")
        return convergiu