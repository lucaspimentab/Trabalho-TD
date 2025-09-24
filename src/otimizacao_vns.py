import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Dados
c = pd.read_csv("data/custos.csv", header=None).values       # custos c(i,j)
a = pd.read_csv("data/recursos.csv", header=None).values     # recursos a(i,j)
b = pd.read_csv("data/capacidades.csv", header=None).values.flatten()  # capacidade b(i)

m, n = c.shape  # m agentes, n tarefas

# Funções objetivo
def f1(sol):
    """Custo total"""
    return sum(c[sol[j], j] for j in range(n))

def f2(sol):
    """Desequilíbrio de carga"""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    return max(load) - min(load)

def is_feasible(sol):
    """Verifica se a solução respeita as capacidades b(i)"""
    load = np.zeros(m)
    for j in range(n):
        load[sol[j]] += a[sol[j], j]
    return np.all(load <= b)

# Heurística construtiva
def random_feasible_solution():
    """Gera solução viável aleatória"""
    sol = -np.ones(n, dtype=int)
    load = np.zeros(m)
    tasks = list(range(n))
    random.shuffle(tasks)
    for j in tasks:
        feasible_agents = [i for i in range(m) if load[i] + a[i, j] <= b[i]]
        if feasible_agents:
            i = random.choice(feasible_agents)
            sol[j] = i
            load[i] += a[i, j]
        else:
            return None
    return sol

def greedy_solution():
    """Constrói solução inicial tentando sempre o menor custo viável"""
    sol = -np.ones(n, dtype=int)
    load = np.zeros(m)
    for j in range(n):
        order = np.argsort(c[:, j])
        placed = False
        for i in order:
            if load[i] + a[i, j] <= b[i]:
                sol[j] = i
                load[i] += a[i, j]
                placed = True
                break
        if not placed:
            return random_feasible_solution()
    return sol

# Estruturas de vizinhança
def swap(sol):
    """Troca as tarefas de duas posições"""
    s = sol.copy()
    i, j = random.sample(range(n), 2)
    s[i], s[j] = s[j], s[i]
    return s

def shift(sol):
    """Move uma tarefa para outro agente aleatório"""
    s = sol.copy()
    j = random.randrange(n)
    agents = list(range(m))
    agents.remove(s[j])
    s[j] = random.choice(agents)
    return s

def two_swap(sol):
    """Troca duas tarefas diferentes entre agentes distintos"""
    s = sol.copy()
    j1, j2 = random.sample(range(n), 2)
    s[j1], s[j2] = s[j2], s[j1]
    return s

neighborhoods = [swap, shift, two_swap]

# Busca local e VNS
def best_improvement(sol, obj_func):
    """Busca local (Best Improvement)"""
    best = sol.copy()
    best_val = obj_func(best)
    improved = True
    while improved:
        improved = False
        for nh in neighborhoods:
            candidate = nh(best)
            if is_feasible(candidate):
                val = obj_func(candidate)
                if val < best_val:
                    best, best_val = candidate, val
                    improved = True
    return best, best_val

def VNS(obj_func, max_iter=500):
    """Variable Neighborhood Search"""
    sol = greedy_solution()
    while sol is None or not is_feasible(sol):
        sol = random_feasible_solution()

    best = sol.copy()
    best_val = obj_func(best)
    history = [best_val]

    for _ in range(max_iter):
        k = random.randint(0, len(neighborhoods) - 1)
        s_perturb = neighborhoods[k](best)
        if not is_feasible(s_perturb):
            continue
        s_local, val_local = best_improvement(s_perturb, obj_func)
        if val_local < best_val:
            best, best_val = s_local, val_local
        history.append(best_val)

    return best, best_val, history

# Execução e resultados
def run_experiments(obj_func, name="f1"):
    results = []
    histories = []
    best_global = None
    best_val_global = float("inf")

    for run in range(5):
        best, val, hist = VNS(obj_func)
        results.append(val)
        histories.append(hist)
        if val < best_val_global:
            best_global, best_val_global = best, val
        print(f"Execução {run+1} - {name}: {val:.2f}")

    print(f"{name} -> min: {np.min(results):.2f}, mean: {np.mean(results):.2f}, "
          f"std: {np.std(results):.2f}, max: {np.max(results):.2f}")

    # Curvas de convergência
    plt.figure()
    for hist in histories:
        plt.plot(hist, alpha=0.7)
    plt.title(f"Convergência - {name}")
    plt.xlabel("Iteração")
    plt.ylabel(name)
    plt.grid()
    plt.savefig(f"graphs/{name}_convergencia.png", dpi=150)
    plt.close()

    # Melhor solução encontrada (carga por agente)
    loads = np.zeros(m)
    for j in range(n):
        loads[best_global[j]] += a[best_global[j], j]

    plt.figure()
    plt.bar(range(1, m+1), loads, tick_label=[f"Agente {i+1}" for i in range(m)])
    plt.title(f"Melhor solução - {name}\nValor: {best_val_global:.2f}")
    plt.xlabel("Agentes")
    plt.ylabel("Carga total")
    plt.grid(axis="y")
    plt.savefig(f"graphs/{name}_melhor.png", dpi=150)
    plt.close()

    return results, histories, best_global, best_val_global

# Main
if __name__ == "__main__":
    print("=== Otimização f1 (Custo) ===")
    run_experiments(f1, "f1")

    print("=== Otimização f2 (Equilíbrio) ===")
    run_experiments(f2, "f2")