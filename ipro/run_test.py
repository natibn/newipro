# run_test.py
from ipro import IPRO
from oracles.ppo_oracle import PPOOracle
from envs.deep_sea_treasure import DeepSeaTreasureEnv

if __name__ == "__main__":
    env = DeepSeaTreasureEnv()
    oracle = PPOOracle(env, train_steps=5000)

    class PPOAsLinearSolver:
        def __init__(self, oracle):
            self.oracle = oracle

        def solve(self, weights):
            return self.oracle.solve(weights)

    solver = PPOAsLinearSolver(oracle)

    ipro = IPRO(
        problem_id="dst",
        dimensions=2,
        oracle=oracle,
        objectives=["treasure", "time"],
        linear_solver=solver,
        direction="maximize",
        offset=0.1,
        tolerance= 1.e-15,
        max_iterations=10,
        update_freq=1
    )

    init_vars = ipro.init_phase()
    iteration = 0
    done = False
    current_subproblem = init_vars.get("subproblems", None)
    current_weights = None

    while not done and iteration < ipro.max_iterations:
        print("="*50)
        print(f"Iteração {iteration + 1}/{ipro.max_iterations}")

        # Verificação de limite para evitar IndexError
        if iteration >= len(ipro.weights):
            print("Não há mais subproblemas para decompor. Encerrando simulação.")
            break

        # Se não houver subproblema inicial, decompõe para a iteração atual
        if current_subproblem is None or current_weights is None:
            print("Decompondo problema para a iteração atual...")
            current_subproblem = ipro.decompose_problem(iteration)
            current_weights = current_subproblem["weight"]
            print(f"Subproblema gerado: {current_subproblem}")
            print(f"Pesos atuais: {current_weights}")

        # Resolve o subproblema usando o solver linear (PPOAsLinearSolver)
        print("Resolvendo subproblema com PPOAsLinearSolver...")
        point, _ = solver.solve(current_weights)
        print(f"Solução encontrada: {point}")

        # Adiciona a solução ao conjunto de soluções, se aplicável
        print("Verificando se a solução deve ser adicionada ao conjunto de soluções...")
        new_subproblem = ipro.maybe_add_solution(current_subproblem, point)
        if not new_subproblem:
            print("Nenhum novo subproblema gerado. Adicionando subproblema como concluído.")
            ipro.maybe_add_completed(current_subproblem, point, current_weights)
            # Prepara para próxima iteração
            current_subproblem = None
            current_weights = None
        else:
            print("Novo subproblema gerado para próxima iteração.")
            current_subproblem = new_subproblem
            current_weights = new_subproblem["weight"]

        print("Estimando erro para verificar critério de parada...")
        done = ipro.estimate_error()
        print(f"Erro estimado: {done}")
        iteration += 1

    # Exibe as soluções de Pareto encontradas
    print("Soluções Pareto encontradas:")
    for p in ipro.pf:
        print(dict(zip(ipro.objectives, p)))
