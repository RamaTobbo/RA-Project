# Code/Michalewicz_Code.py
import os
import re
import copy
import random
import numpy as np
import pandas as pd


# ---------------------------
# Output folder (CSV files)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "GeneratedData")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------
# Safe-ish expression eval
# ---------------------------
SAFE_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "pi": np.pi,
    "e": np.e,
    "abs": abs,
}

ALLOWED_EXPR = re.compile(r"^[0-9XxYy+\-*/().,^ \tA-Za-z]*$")


def safe_eval_expression(expr: str, x: float, y: float) -> float:
    expr = (expr or "").strip()
    if len(expr) == 0 or len(expr) > 250:
        raise ValueError("Empty/too long function.")
    if not ALLOWED_EXPR.match(expr):
        raise ValueError("Invalid characters in function.")

    expr = expr.replace("^", "**")

    scope = {"__builtins__": {}}
    locals_ = dict(SAFE_FUNCS)
    locals_.update({"X": float(x), "Y": float(y), "x": float(x), "y": float(y)})

    return float(eval(expr, scope, locals_))


def fitness(fun_expr: str, position):
    return safe_eval_expression(fun_expr, position[0], position[1])


# ==========================================================
# ========================= WOLF ============================
# ==========================================================
class Wolf:
    def __init__(self, fun_expr, dim, minx, maxx, seed):
        self.id = seed
        rnd = random.Random(seed)
        self.position = [(maxx - minx) * rnd.random() + minx for _ in range(dim)]
        self.fitness = fitness(fun_expr, self.position)


def gwo(fun_expr, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)
    population = [Wolf(fun_expr, dim, minx, maxx, i) for i in range(n)]
    population = sorted(population, key=lambda w: w.fitness)

    alpha, beta, gamma = copy.copy(population[:3])
    fit_rows = [["Iteration", "Fitness"]]

    Iter = 0
    while Iter < max_iter:
        for i in range(n):
            fit_rows.append([Iter, population[i].fitness])

        a = 2 * (1 - Iter / max_iter)

        for i in range(n):
            A1 = a * (2 * rnd.random() - 1)
            A2 = a * (2 * rnd.random() - 1)
            A3 = a * (2 * rnd.random() - 1)
            C1 = 2 * rnd.random()
            C2 = 2 * rnd.random()
            C3 = 2 * rnd.random()

            Xnew = [0.0 for _ in range(dim)]
            for j in range(dim):
                X1 = alpha.position[j] - A1 * abs(C1 * alpha.position[j] - population[i].position[j])
                X2 = beta.position[j] - A2 * abs(C2 * beta.position[j] - population[i].position[j])
                X3 = gamma.position[j] - A3 * abs(C3 * gamma.position[j] - population[i].position[j])
                Xnew[j] = (X1 + X2 + X3) / 3.0

                # clamp
                Xnew[j] = min(max(Xnew[j], minx), maxx)

            fnew = fitness(fun_expr, Xnew)
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        population = sorted(population, key=lambda w: w.fitness)
        alpha, beta, gamma = copy.copy(population[:3])
        Iter += 1

    return fit_rows


def driverWolf(it, gen, lb, ub, fun_expr):
    rows = gwo(fun_expr, gen, it, 2, lb, ub)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "wolf_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= BEE =============================
# ==========================================================
def abc_algorithm(fun_expr, lb, ub, colony_size=10, max_iter=50, num_trials=5):
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    num_vars = len(lb)
    colony = np.zeros((colony_size, num_vars))
    fitvals = np.zeros(colony_size)
    trials = np.ones(colony_size) * num_trials

    # init
    for i in range(colony_size):
        colony[i] = np.random.uniform(lb, ub)
        fitvals[i] = fitness(fun_expr, colony[i])

    log_rows = [["Iteration", "Phase", "Id", "Fitness"]]

    for iter_ in range(max_iter):
        # employed
        for i in range(colony_size):
            k = np.random.randint(colony_size)
            while k == i:
                k = np.random.randint(colony_size)

            phi = np.random.uniform(-1, 1, num_vars)
            new = colony[i] + phi * (colony[i] - colony[k])
            new = np.maximum(new, lb)
            new = np.minimum(new, ub)

            new_fit = fitness(fun_expr, new)
            if new_fit < fitvals[i]:
                colony[i] = new
                fitvals[i] = new_fit
                trials[i] = num_trials
            else:
                trials[i] -= 1

        for i in range(colony_size):
            log_rows.append([iter_, 0, i, fitvals[i]])

        # onlooker (simple probability)
        probs = (1 / (1 + fitvals - np.min(fitvals) + 1e-9))
        probs = probs / np.sum(probs)

        for _ in range(colony_size):
            i = np.random.choice(np.arange(colony_size), p=probs)
            k = np.random.randint(colony_size)
            while k == i:
                k = np.random.randint(colony_size)

            phi = np.random.uniform(-1, 1, num_vars)
            new = colony[i] + phi * (colony[i] - colony[k])
            new = np.maximum(new, lb)
            new = np.minimum(new, ub)

            new_fit = fitness(fun_expr, new)
            if new_fit < fitvals[i]:
                colony[i] = new
                fitvals[i] = new_fit
                trials[i] = num_trials
            else:
                trials[i] -= 1

        for i in range(colony_size):
            log_rows.append([iter_, 1, i, fitvals[i]])

        # scout
        for i in range(colony_size):
            if trials[i] <= 0:
                colony[i] = np.random.uniform(lb, ub)
                fitvals[i] = fitness(fun_expr, colony[i])
                trials[i] = num_trials

        for i in range(colony_size):
            log_rows.append([iter_, 2, i, fitvals[i]])

    return log_rows


def driverBee(it, gen, lb, ub, fun_expr):
    rows = abc_algorithm(fun_expr, [lb, lb], [ub, ub], colony_size=it, max_iter=gen)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "bee_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= BAT =============================
# ==========================================================
class BatAlgorithm:
    def __init__(self, NP, N_Gen, lb, ub, fun_expr, A=1.0, r=0.5, Qmin=0.0, Qmax=2.0):
        self.D = 2
        self.NP = NP
        self.N_Gen = N_Gen
        self.A = A
        self.r = r
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.lb = lb
        self.ub = ub
        self.fun_expr = fun_expr

        self.Q = np.zeros(NP)
        self.v = np.zeros((NP, self.D))
        self.Sol = np.random.uniform(lb, ub, (NP, self.D))
        self.Fitness = np.array([fitness(fun_expr, s) for s in self.Sol])

        best_idx = np.argmin(self.Fitness)
        self.best = self.Sol[best_idx].copy()
        self.f_min = self.Fitness[best_idx]

    def simplebounds(self, val):
        return np.minimum(np.maximum(val, self.lb), self.ub)

    def move_bat(self):
        log_rows = [["Iteration", "Fitness"]]

        for t in range(self.N_Gen):
            for i in range(self.NP):
                self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * np.random.rand()
                self.v[i] = self.v[i] + (self.Sol[i] - self.best) * self.Q[i]
                S = self.Sol[i] + self.v[i]
                S = self.simplebounds(S)

                if np.random.rand() > self.r:
                    S = self.best + 0.001 * np.random.randn(self.D)
                    S = self.simplebounds(S)

                Fnew = fitness(self.fun_expr, S)

                if (Fnew <= self.Fitness[i]) and (np.random.rand() < self.A):
                    self.Sol[i] = S
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    self.best = S.copy()
                    self.f_min = Fnew

                log_rows.append([t, self.Fitness[i]])

        return log_rows


def driverBat(it, gen, lb, ub, fun_expr):
    algo = BatAlgorithm(NP=it, N_Gen=gen, lb=lb, ub=ub, fun_expr=fun_expr)
    rows = algo.move_bat()
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "bat_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= FISH ============================
# ==========================================================
class FishFunc:
    def __init__(self, fun_expr, lb, ub):
        self.fun_expr = fun_expr
        self.lower_bound = lb
        self.upper_bound = ub

    def calculate_fitness(self, pos):
        return fitness(self.fun_expr, pos)


class Fish:
    def __init__(self, obj_func, positions, iterations_number):
        self.f = obj_func
        self.current_position = positions
        self.weight = iterations_number / 2.0
        self.fitness = np.inf
        self.delta_fitness = 0
        self.delta_position = [0.0, 0.0]

    def evaluate(self):
        self.fitness = self.f.calculate_fitness(self.current_position)

    def update_position_individual_movement(self, step_ind):
        new_pos = []
        for p in self.current_position:
            new = p + (step_ind * np.random.uniform(-1, 1))
            new = min(max(new, self.f.lower_bound), self.f.upper_bound)
            new_pos.append(new)

        new_fit = self.f.calculate_fitness(new_pos)
        if new_fit < self.fitness:
            self.delta_fitness = abs(new_fit - self.fitness)
            self.delta_position = [a - b for a, b in zip(new_pos, self.current_position)]
            self.current_position = new_pos
            self.fitness = new_fit
        else:
            self.delta_position = [0.0, 0.0]
            self.delta_fitness = 0

    def feed(self, max_delta_fitness):
        self.weight = self.weight + (self.delta_fitness / max_delta_fitness) if max_delta_fitness != 0 else 1.0

    def update_position_collective_movement(self, sum_delta_fitness):
        instinct = [dp * self.delta_fitness for dp in self.delta_position]
        if sum_delta_fitness != 0:
            instinct = [v / sum_delta_fitness for v in instinct]

        self.current_position = [
            min(max(self.current_position[i] + instinct[i], self.f.lower_bound), self.f.upper_bound)
            for i in range(2)
        ]

    def update_position_volitive_movement(self, barycenter, step_vol, op):
        self.current_position = [
            min(max(self.current_position[i] + ((self.current_position[i] - barycenter[i]) * step_vol * np.random.rand() * op),
                    self.f.lower_bound),
                self.f.upper_bound)
            for i in range(2)
        ]


class FSS:
    def __init__(self, obj_func, iterations_number, num_individuals):
        self.f = obj_func
        self.iterations = iterations_number
        self.num = num_individuals
        self.cluster = []
        self.global_best = float("inf")
        self.total_weight = float(num_individuals)

        self.initial_step_ind = 0.5
        self.final_step_ind = 0.01
        self.initial_step_vol = 0.5
        self.final_step_vol = 0.01

        self.step_ind = self.initial_step_ind * (obj_func.upper_bound - obj_func.lower_bound)
        self.step_vol = self.initial_step_vol * (obj_func.upper_bound - obj_func.lower_bound)

    def _rand(self):
        return np.random.uniform(self.f.lower_bound, self.f.upper_bound)

    def _init_cluster(self):
        self.cluster = [
            Fish(self.f, [self._rand(), self._rand()], self.iterations)
            for _ in range(self.num)
        ]

    def _barycenter(self):
        wsum = sum(f.weight for f in self.cluster) or 1.0
        pos = np.sum([np.array(f.current_position) * f.weight for f in self.cluster], axis=0)
        return (pos / wsum).tolist()

    def _update_steps(self, i):
        self.step_ind = self.initial_step_ind - i * (self.initial_step_ind - self.final_step_ind) / self.iterations
        self.step_vol = self.initial_step_vol - i * (self.initial_step_vol - self.final_step_vol) / self.iterations

    def search(self):
        rows = [["Iteration", "Phase", "Id", "Fitness"]]
        self._init_cluster()

        for i in range(self.iterations):
            for f in self.cluster:
                f.evaluate()
                self.global_best = min(self.global_best, f.fitness)

            # phase 0
            for f in self.cluster:
                f.update_position_individual_movement(self.step_ind)
            for j, f in enumerate(self.cluster):
                f.evaluate()
                rows.append([i, 0, j, f.fitness])

            # feeding
            max_df = max([f.delta_fitness for f in self.cluster]) if self.cluster else 0
            for f in self.cluster:
                f.feed(max_df)

            # phase 1
            sum_df = sum([f.delta_fitness for f in self.cluster]) if self.cluster else 0
            for f in self.cluster:
                f.update_position_collective_movement(sum_df)
            for j, f in enumerate(self.cluster):
                f.evaluate()
                rows.append([i, 1, j, f.fitness])

            # phase 2
            bary = self._barycenter()
            current_total_weight = sum([f.weight for f in self.cluster])
            op = -1 if current_total_weight > self.total_weight else 1
            for f in self.cluster:
                f.update_position_volitive_movement(bary, self.step_vol, op)
            for j, f in enumerate(self.cluster):
                f.evaluate()
                rows.append([i, 2, j, f.fitness])

            self.total_weight = current_total_weight
            self._update_steps(i)

        return rows


def driverFish(it, gen, lb, ub, fun_expr):
    rows = FSS(FishFunc(fun_expr, lb, ub), gen, it).search()
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "fish_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ✅ REQUIRED BY FLASK
# ==========================================================
def check(it, gen, lb, ub, fun_expr) -> str:
    """
    Runs all 4 algorithms on the CUSTOM equation.
    Saves CSVs into Code/GeneratedData/.
    Returns: "Done" or "error"
    """
    try:
        if lb >= ub:
            return "error"
        # validate expression
        _ = safe_eval_expression(fun_expr, lb, lb)

        driverWolf(it, gen, lb, ub, fun_expr)
        driverBee(it, gen, lb, ub, fun_expr)
        driverBat(it, gen, lb, ub, fun_expr)
        driverFish(it, gen, lb, ub, fun_expr)

        return "Done"
    except Exception:
        return "error"
