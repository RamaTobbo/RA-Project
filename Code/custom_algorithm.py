#!/usr/bin/env python
# coding: utf-8

"""
FLASK VERSION of your Dash script (single-file).

What this does:
- Shows a Flask form page at /
- User submits: individuals (it), generations (gen), lower bound (lb), upper bound (ub), function
- Runs: Wolf + Bee + Bat + Fish drivers
- Runs your 2 R scripts: Preprocessing.R and SummaryVis.R
- Redirects to /results when done
- NO Dash, NO callbacks, NO app.run_server()

Folder assumptions (match your structure):
- templates/
    - custom_function.html
    - custom_results.html
- Code/GeneratedData/   (or GeneratedData/) for CSV outputs (see OUTPUT_DIR below)
- Preprocessing.R and SummaryVis.R in the same folder as this file (or update paths)
"""

import os
import re
import subprocess
import random
import copy
import math
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS


# =========================
# Flask app
# =========================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If your CSVs should go to Code/GeneratedData, keep it like this:
OUTPUT_DIR = os.path.join(BASE_DIR, "Code", "GeneratedData")
# If your project uses just GeneratedData/ at root, use this instead:
# OUTPUT_DIR = os.path.join(BASE_DIR, "GeneratedData")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# R scripts
# =========================
# Put the R scripts next to app.py (or change these paths)
r_script_path1 = os.path.join(BASE_DIR, "Preprocessing.R")
r_script_path2 = os.path.join(BASE_DIR, "SummaryVis.R")

# Optional: if you MUST set R path on Windows, keep it (otherwise remove)
# os.environ['PATH'] += r';C:\Program Files\R\R-4.2.1\bin'


# =========================
# SAFE eval helpers (important)
# =========================
SAFE_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "pi": np.pi,
    "e": np.e,
    "pow": pow,
    "abs": abs,
}

ALLOWED_EXPR = re.compile(r"^[0-9X Yx y+\-*/().,^ \tA-Za-z]*$")

def safe_eval_expression(expr: str, X: float, Y: float) -> float:
    """
    Safely evaluate a math expression in terms of X and Y.
    Examples:
      cos(X)*(sin(Y)**2)
      X**2 + Y**2
      exp(X) + sqrt(abs(Y))

    Blocks imports / builtins.
    """
    expr = (expr or "").strip()
    if len(expr) == 0 or len(expr) > 250:
        raise ValueError("Empty/too long function.")
    if not ALLOWED_EXPR.match(expr):
        raise ValueError("Invalid characters in function.")

    # Convert caret to python power if user types ^
    expr = expr.replace("^", "**")

    # Evaluate with *no builtins*
    scope = {"__builtins__": {}}
    locals_ = dict(SAFE_FUNCS)
    locals_.update({"X": float(X), "Y": float(Y), "x": float(X), "y": float(Y)})

    return float(eval(expr, scope, locals_))


# =========================
# Your objective function wrapper
# =========================
def fitness(func_expr: str, position):
    X = position[0]
    Y = position[1]
    return safe_eval_expression(func_expr, X, Y)


# =========================
# CHECK + RUN ALL (Wolf/Bee/Bat/Fish) + R scripts
# =========================
def check(it, gen, lb, ub, fun_expr: str) -> str:
    """
    Validates inputs and runs all algorithms.
    Returns: "error" or "Done"
    """
    # validate bounds
    if lb >= ub:
        return "error"

    # quick test eval at (lb, lb) to detect invalid expression
    try:
        _ = safe_eval_expression(fun_expr, lb, lb)
    except Exception:
        return "error"

    # Run algorithms
    driverWolf(it, gen, lb, ub, fun_expr)
    driverBee(it, gen, lb, ub, fun_expr)
    driverBat(it, gen, lb, ub, fun_expr)
    driverFish(it, gen, lb, ub, fun_expr)

    # Run R scripts (only if they exist)
    if os.path.exists(r_script_path1):
        subprocess.run(["Rscript", r_script_path1], check=False, cwd=BASE_DIR)
    if os.path.exists(r_script_path2):
        subprocess.run(["Rscript", r_script_path2], check=False, cwd=BASE_DIR)

    return "Done"


# ==========================================================
# ========================= WOLF ============================
# ==========================================================
class wolf:
    def __init__(self, fitness_fn, dim, minx, maxx, seed, func_expr):
        self.id = seed
        self.rnd = random.Random(seed)
        self.position = [0.0 for _ in range(dim)]
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
        self.fitness = fitness_fn(func_expr, self.position)


def gwo(fitness_fn, max_iter, n, dim, minx, maxx, func_expr):
    rnd = random.Random(0)
    population = [wolf(fitness_fn, dim, minx, maxx, i, func_expr) for i in range(n)]
    population = sorted(population, key=lambda temp: temp.fitness)
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[:3])

    Iter = 0
    while Iter < max_iter:
        # log fitness for each wolf
        for i in range(n):
            fit.append([Iter, population[i].fitness])

        a = 2 * (1 - Iter / max_iter)

        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2 * rnd.random(), 2 * rnd.random()

            Xnew = [0.0 for _ in range(dim)]
            for j in range(dim):
                X1 = alpha_wolf.position[j] - A1 * abs(C1 * alpha_wolf.position[j] - population[i].position[j])
                X2 = beta_wolf.position[j] - A2 * abs(C2 * beta_wolf.position[j] - population[i].position[j])
                X3 = gamma_wolf.position[j] - A3 * abs(C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j] = (X1 + X2 + X3) / 3.0
                if Xnew[j] > maxx:
                    Xnew[j] = maxx
                if Xnew[j] < minx:
                    Xnew[j] = minx

            fnew = fitness_fn(func_expr, Xnew)
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        population = sorted(population, key=lambda temp: temp.fitness)
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[:3])
        Iter += 1

    return alpha_wolf.position


def driverWolf(it, gen, lb, ub, func_expr):
    global fit
    fit = []
    fit.append(["Iteration", "Fitness"])
    dim = 2
    best_position = gwo(fitness, gen, it, dim, lb, ub, func_expr)
    pd.DataFrame(fit).to_csv(os.path.join(OUTPUT_DIR, "wolf_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= BEE =============================
# ==========================================================
def abc_algorithm(func_expr, obj_function, lb, ub, colony_size=10, max_iter=10, num_trials=5):
    num_variables = len(lb)
    colony = np.zeros((colony_size, num_variables))
    fitness_vals = np.zeros(colony_size)
    alltrials = np.zeros(colony_size)
    trials = num_trials

    for i in range(colony_size):
        colony[i, :] = np.random.uniform(lb, ub)
        fitness_vals[i] = obj_function(func_expr, colony[i, :])
        alltrials[i] = trials

    best_solution = colony[np.argmin(fitness_vals), :]
    best_fitness = np.min(fitness_vals)

    for iter_ in range(max_iter):
        # employed bees
        for i in range(colony_size):
            k = np.random.randint(colony_size)
            while k == i:
                k = np.random.randint(colony_size)

            phi = np.random.uniform(-1, 1, num_variables)
            new_solution = colony[i, :] + phi * (colony[i, :] - colony[k, :])
            new_solution = np.maximum(new_solution, lb)
            new_solution = np.minimum(new_solution, ub)
            new_fitness = obj_function(func_expr, new_solution)

            if new_fitness < fitness_vals[i]:
                colony[i, :] = new_solution
                fitness_vals[i] = new_fitness
                alltrials[i] = trials
            else:
                alltrials[i] -= 1

        for i in range(colony_size):
            fit.append([iter_, 0, i, fitness_vals[i]])

        # onlooker bees (your original probability is odd, but keeping it)
        i = 0
        while i < colony_size:
            denom = np.sum(fitness_vals) if np.sum(fitness_vals) != 0 else 1.0
            if np.random.uniform(0, 1) < (fitness_vals[i] / denom):
                k = np.random.randint(colony_size)
                while k == i:
                    k = np.random.randint(colony_size)

                phi = np.random.uniform(-1, 1, num_variables)
                new_solution = colony[i, :] + phi * (colony[i, :] - colony[k, :])
                new_solution = np.maximum(new_solution, lb)
                new_solution = np.minimum(new_solution, ub)
                new_fitness = obj_function(func_expr, new_solution)

                if new_fitness < fitness_vals[i]:
                    colony[i, :] = new_solution
                    fitness_vals[i] = new_fitness
                    alltrials[i] = trials
                else:
                    alltrials[i] -= 1
            i += 1

        for i in range(colony_size):
            fit.append([iter_, 1, i, fitness_vals[i]])

        # scout bees
        for i in range(colony_size):
            if alltrials[i] <= 0:
                colony[i, :] = np.random.uniform(lb, ub)
                fitness_vals[i] = obj_function(func_expr, colony[i, :])
                alltrials[i] = trials

        for i in range(colony_size):
            fit.append([iter_, 2, i, fitness_vals[i]])

        if np.min(fitness_vals) < best_fitness:
            best_solution = colony[np.argmin(fitness_vals), :]
            best_fitness = np.min(fitness_vals)

    return best_solution


def driverBee(it, gen, lb, ub, func_expr):
    global fit
    fit = []
    fit.append(["Iteration", "Phase", "Id", "Fitness"])
    _ = abc_algorithm(func_expr, fitness, [lb, lb], [ub, ub], it, gen)
    pd.DataFrame(fit).to_csv(os.path.join(OUTPUT_DIR, "bee_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= BAT =============================
# ==========================================================
class BatAlgorithm:
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function, func_expr):
        self.D = D
        self.NP = NP
        self.N_Gen = N_Gen
        self.A = A
        self.r = r
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.Lower = Lower
        self.Upper = Upper

        self.f_min = 0.0
        self.Lb = [0] * self.D
        self.Ub = [0] * self.D
        self.Q = [0] * self.NP
        self.v = [[0 for _ in range(self.D)] for _ in range(self.NP)]
        self.Sol = [[0 for _ in range(self.D)] for _ in range(self.NP)]
        self.Fitness = [0] * self.NP
        self.best = [0] * self.D
        self.Fun = function
        self.func_expr = func_expr

    def best_bat(self):
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]

    def init_bat(self):
        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper

        for i in range(self.NP):
            self.Q[i] = 0
            for j in range(self.D):
                rnd = np.random.uniform(0, 1)
                self.v[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd
            self.Fitness[i] = self.Fun(self.func_expr, self.Sol[i])
        self.best_bat()

    @staticmethod
    def simplebounds(val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def move_bat(self):
        S = [[0.0 for _ in range(self.D)] for _ in range(self.NP)]
        self.init_bat()

        for t in range(self.N_Gen):
            for i in range(self.NP):
                rnd = np.random.uniform(0, 1)
                self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * rnd

                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] - self.best[j]) * self.Q[i]
                    S[i][j] = self.Sol[i][j] + self.v[i][j]
                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j])

                rnd = np.random.random_sample()
                if rnd > self.r:
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j])

                Fnew = self.Fun(self.func_expr, S[i])

                rnd = np.random.random_sample()
                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew

                fit.append([t, self.Fitness[i]])


def driverBat(it, gen, lb, ub, func_expr):
    global fit
    fit = []
    fit.append(["Iteration", "Fitness"])
    algo = BatAlgorithm(D=2, NP=it, N_Gen=gen, A=5, r=0.5, Qmin=0.0, Qmax=2.0,
                        Lower=lb, Upper=ub, function=fitness, func_expr=func_expr)
    algo.move_bat()
    pd.DataFrame(fit).to_csv(os.path.join(OUTPUT_DIR, "bat_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= FISH ============================
# ==========================================================
class AFunction:
    upper_bound = 1
    lower_bound = -1
    fun_expr = ""

    def calculate_fitness(self, position):
        raise NotImplementedError


class FishFunc(AFunction):
    def __init__(self, fun_expr, lb, ub):
        AFunction.upper_bound = ub
        AFunction.lower_bound = lb
        AFunction.fun_expr = fun_expr

    def calculate_fitness(self, x):
        return fitness(AFunction.fun_expr, x)


class Fish:
    def __init__(self, objective_function, positions, iterations_number, dimensions):
        self.fitness_function = objective_function
        self.current_position = positions
        self.weight = iterations_number / 2.0
        self.fitness = np.inf
        self.delta_fitness = 0
        self.delta_position = [0] * dimensions
        self.dimensions = dimensions

    def evaluate(self):
        self.fitness = self.fitness_function.calculate_fitness(self.current_position)

    def update_position_individual_movement(self, step_ind):
        new_positions = []
        for pos in self.current_position:
            new = pos + (step_ind * np.random.uniform(-1, 1))
            if new > self.fitness_function.upper_bound:
                new = self.fitness_function.upper_bound
            elif new < self.fitness_function.lower_bound:
                new = self.fitness_function.lower_bound
            new_positions.append(new)

        new_fitness = self.fitness_function.calculate_fitness(new_positions)
        if new_fitness < self.fitness:
            self.delta_fitness = abs(new_fitness - self.fitness)
            self.delta_position = [x - y for x, y in zip(new_positions, self.current_position)]
            self.current_position = list(new_positions)
            self.fitness = new_fitness
        else:
            self.delta_position = [0] * self.dimensions
            self.delta_fitness = 0

    def feed(self, max_delta_fitness):
        if max_delta_fitness != 0:
            self.weight = self.weight + (self.delta_fitness / max_delta_fitness)
        else:
            self.weight = 1

    def update_position_collective_movement(self, sum_delta_fitness):
        instinct = [self.delta_position[i] * self.delta_fitness for i in range(len(self.delta_position))]
        if sum_delta_fitness != 0:
            instinct = [val / sum_delta_fitness for val in instinct]

        new_positions = []
        for i in range(len(self.current_position)):
            new = self.current_position[i] + instinct[i]
            if new > self.fitness_function.upper_bound:
                new = self.fitness_function.upper_bound
            elif new < self.fitness_function.lower_bound:
                new = self.fitness_function.lower_bound
            new_positions.append(new)
        self.current_position = list(new_positions)

    def update_position_volitive_movement(self, barycenter, step_vol, search_operator):
        new_positions = []
        for i, pos in enumerate(self.current_position):
            new = pos + (((pos - barycenter[i]) * step_vol * np.random.uniform(0, 1)) * search_operator)
            if new > self.fitness_function.upper_bound:
                new = self.fitness_function.upper_bound
            elif new < self.fitness_function.lower_bound:
                new = self.fitness_function.lower_bound
            new_positions.append(new)
        self.current_position = list(new_positions)


class FSS:
    def __init__(self, objective_function, iterations_number, num_of_individuos, dimensions=2):
        self.function = objective_function
        self.dimensions = dimensions
        self.iterations_number = iterations_number
        self.num_of_individuos = num_of_individuos
        self.cluster = []
        self.global_best = float("inf")
        self.global_best_position = []

        self.total_weight = 1 * self.num_of_individuos
        self.initial_step_ind = 0.5
        self.final_step_ind = 0.01
        self.step_ind = self.initial_step_ind * (objective_function.upper_bound - objective_function.lower_bound)

        self.initial_step_vol = 0.5
        self.final_step_vol = 0.01
        self.step_vol = self.initial_step_vol * (objective_function.upper_bound - objective_function.lower_bound)

        self.list_global_best_values = []

    def _get_random_number(self):
        return np.random.uniform(self.function.lower_bound, self.function.upper_bound)

    def _initialize_cluster(self):
        self.cluster = []
        for _ in range(self.num_of_individuos):
            fish_obj = Fish(
                objective_function=self.function,
                positions=[self._get_random_number() for _ in range(self.dimensions)],
                iterations_number=self.iterations_number,
                dimensions=self.dimensions,
            )
            self.cluster.append(fish_obj)

    def evaluate_cluster(self):
        for fish_obj in self.cluster:
            fish_obj.evaluate()

    def updates_optimal_solution(self):
        for fish_obj in self.cluster:
            if fish_obj.fitness < self.global_best:
                self.global_best = fish_obj.fitness
                self.global_best_position = list(fish_obj.current_position)

    def apply_individual_movement(self):
        for fish_obj in self.cluster:
            fish_obj.update_position_individual_movement(self.step_ind)

    def apply_feeding(self):
        max_delta_fitness = max([fish_obj.delta_fitness for fish_obj in self.cluster])
        for fish_obj in self.cluster:
            fish_obj.feed(max_delta_fitness)

    def apply_instinctive_collective_movement(self):
        sum_delta_fitness = sum([fish_obj.delta_fitness for fish_obj in self.cluster])
        for fish_obj in self.cluster:
            fish_obj.update_position_collective_movement(sum_delta_fitness)

    def _calculate_barycenter(self):
        sum_weights = sum([fish_obj.weight for fish_obj in self.cluster])
        if sum_weights == 0:
            sum_weights = 1.0
        sum_position_and_weights = [[x * fish_obj.weight for x in fish_obj.current_position] for fish_obj in self.cluster]
        sum_position_and_weights = np.sum(sum_position_and_weights, 0)
        return [s / sum_weights for s in sum_position_and_weights]

    def apply_collective_volitive_movement(self):
        barycenter = self._calculate_barycenter()
        current_total_weight = sum([fish_obj.weight for fish_obj in self.cluster])
        search_operator = -1 if current_total_weight > self.total_weight else 1
        for fish_obj in self.cluster:
            fish_obj.update_position_volitive_movement(barycenter, self.step_vol, search_operator)

    def update_step(self, current_i):
        self.step_ind = self.initial_step_ind - current_i * float(
            self.initial_step_ind - self.final_step_ind
        ) / self.iterations_number
        self.step_vol = self.initial_step_vol - current_i * float(
            self.initial_step_vol - self.final_step_vol
        ) / self.iterations_number

    def update_total_weight(self):
        self.total_weight = sum([fish_obj.weight for fish_obj in self.cluster])

    def search(self):
        self._initialize_cluster()
        for i in range(self.iterations_number):
            self.evaluate_cluster()
            self.updates_optimal_solution()

            # individual movement
            self.apply_individual_movement()
            self.evaluate_cluster()
            self.updates_optimal_solution()
            for j, fish_obj in enumerate(self.cluster):
                fit.append([i, 0, j, fish_obj.fitness])

            # feeding
            self.apply_feeding()

            # instinctive movement
            self.apply_instinctive_collective_movement()
            for j, fish_obj in enumerate(self.cluster):
                fit.append([i, 1, j, fish_obj.fitness])

            # volitive movement
            self.apply_collective_volitive_movement()
            for j, fish_obj in enumerate(self.cluster):
                fit.append([i, 2, j, fish_obj.fitness])

            self.update_step(i)
            self.update_total_weight()

            self.evaluate_cluster()
            self.updates_optimal_solution()
            self.list_global_best_values.append(self.global_best)


def driverFish(it, gen, lb, ub, fun_expr):
    global fit
    fit = []
    fit.append(["Iteration", "Phase", "Id", "Fitness"])
    fss = FSS(FishFunc(fun_expr, lb, ub), iterations_number=gen, num_of_individuos=it, dimensions=2)
    fss.search()
    pd.DataFrame(fit).to_csv(os.path.join(OUTPUT_DIR, "fish_fun_fitness.csv"), header=False, index=False)


# ==========================================================
# ========================= FLASK ROUTES =====================
# ==========================================================
@app.route("/", methods=["GET"])
def home():
    # Your form should be in templates/custom_function.html
    return render_template("custom_function.html")


@app.route("/run", methods=["POST"])
def run_optimization():
    """
    Receives form fields:
      it, gen, lb, ub, function
    """
    try:
        it = int(request.form.get("it", "50"))
        gen = int(request.form.get("gen", "75"))
        lb = float(request.form.get("lb", "0"))
        ub = float(request.form.get("ub", "1"))
        fun_expr = request.form.get("function", "").strip()
    except Exception:
        return render_template("custom_function.html", error="Invalid numeric inputs.")

    status = check(it, gen, lb, ub, fun_expr)
    if status == "error":
        return render_template(
            "custom_function.html",
            error="Error: lower bound must be < upper bound, and the function must be valid.",
            last_it=it, last_gen=gen, last_lb=lb, last_ub=ub, last_fun=fun_expr
        )

    # Save last run info to show on results page
    return redirect(url_for("results", fun=fun_expr, it=it, gen=gen, lb=lb, ub=ub))


@app.route("/results", methods=["GET"])
def results():
    """
    Show results page. You can load CSVs from OUTPUT_DIR in your HTML,
    or display links to them.
    """
    fun_expr = request.args.get("fun", "")
    it = request.args.get("it", "")
    gen = request.args.get("gen", "")
    lb = request.args.get("lb", "")
    ub = request.args.get("ub", "")

    files = {
        "wolf": "wolf_fun_fitness.csv",
        "bee": "bee_fun_fitness.csv",
        "bat": "bat_fun_fitness.csv",
        "fish": "fish_fun_fitness.csv",
    }

    return render_template(
        "custom_results.html",
        function_expr=fun_expr,
        it=it, gen=gen, lb=lb, ub=ub,
        output_dir=OUTPUT_DIR,
        files=files
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=7771)
