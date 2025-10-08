#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import dash
# import dash_core_components as dcc
# import dash_html_components as html
from flask import Flask
from flask_cors import CORS
from dash import Dash, dcc, html, Input, Output, State


from flask import send_from_directory
from dash.dependencies import Input, Output, State
#ackley multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos,sin,tan
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
import copy    
import sys  
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import random
import copy
import numpy as np
import functools
import random
import math
import numpy as np
from flask import send_from_directory, redirect
import webbrowser
from flask import Flask
from flask_cors import CORS


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


os.environ['PATH'] += ';C:\\Program Files\\R\\R-4.2.1\\bin'


# In[ ]:


# Define the path to your R script
r_script_path2 = "SummaryVis.R"
r_script_path1="Preprocessing.R"


# In[ ]:


# objective function
def fitness(func,position):
    X=position[0]
    Y=position[1]
    return eval(func)


# In[ ]:


func='X+Y'
def check(it,gen,lb,ub,fun):
    print(it,gen,lb,ub,fun)
    evalfun=''
    X=lb
    Y=lb
    global func
    func= fun
    try:
        evalfun = str(eval(func))
    except:
        evalfun = 'Error'

    if(lb>=ub or evalfun=='Error'):
        return "error"
    else: 
        webbrowser.open('Redirecting.html')
        print('Wolf')
        driverWolf(it,gen,lb,ub,func)
        print('Bee')
        driverBee(it,gen,lb,ub,func)
        print('Bat')
        driverBat(it,gen,lb,ub,func)
        print("Fish")
        driverFish(it,gen,lb,ub,func)

        subprocess.run(["Rscript",r_script_path1])
        subprocess.run(["Rscript", r_script_path2])

        return "Done"


# # Wolf

# In[ ]:





# In[ ]:


# wolf class
class wolf:
  def __init__(self, fitness, dim, minx, maxx, seed, func):
    self.id=seed
    self.rnd = random.Random(seed)
    self.position = [0.0 for i in range(dim)]

    for i in range(dim):
      self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

    self.fitness = fitness(func,self.position) 



# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx, func):
    rnd = random.Random(0)
    info=[max_iter,n]
    # create n random wolves
    population = [ wolf(fitness, dim, minx, maxx, i, func) for i in range(n)]
    for i in range(n):
      print(population[i].id," ",population[i].position)


    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key = lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gaama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])


    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        if(Iter%10==0):
            print()
            print()
            print()
            print("ITER : ",Iter)

        for i in range(n):
          fit.append([Iter,population[i].fitness])

        # linearly decreased from 2 to 0
        a = 2*(1 - Iter/max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
              2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                  C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                  C2 *  beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                  C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j]+= X1[j] + X2[j] + X3[j]

            for j in range(dim):
                Xnew[j]/=3.0
                if Xnew[j]>maxx:
                    Xnew[j]=maxx
                if Xnew[j]<minx:
                    Xnew[j]=minx

            # fitness calculation of new solution
            fnew = fitness(func,Xnew)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew

        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key = lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

        Iter+= 1
    # end-while

    # returning the best solution
    return alpha_wolf.position

#----------------------------


# Driver code for rastrigin function
def driverWolf(it,gen,lb,ub,func):
    global fit
    fit=[]
    labels1=["Iteration","Fitness"]
    fit.append(labels1) 
    print("\nBegin grey wolf optimization on  function ackley\n")
    dim = 2

    num_particles = it
    max_iter = gen
    print("Setting num_particles = " + str(num_particles))
    print("Setting max_iter    = " + str(max_iter))
    print("\nStarting GWO algorithm\n")



    best_position = gwo(fitness, max_iter, num_particles, dim, lb,ub,func)

    print("\nGWO completed\n")
    print("\nBest solution found:")
    print(["%.16f"%best_position[k] for k in range(dim)])
    err = fitness(func,best_position)
    print("fitness of best solution = %.6f" % err)
    print("func : "+func) 
    print("\nEnd GWO for ackley\n")
    pd.DataFrame(fit).to_csv("GeneratedData/wolf_fun_fitness.csv",header=False,index=False)


# # Bee

# In[ ]:


def abc_algorithm(func,obj_function, lb, ub, colony_size=10, max_iter=10 ,num_trials=5):
    """
    Artificial Bee Colony Algorithm
    :param obj_function: objective function to be optimized
    :param lb: lower bound of decision variables
    :param ub: upper bound of decision variables
    :param colony_size: number of bees in the colony
    :param max_iter: maximum number of iterations
    :param num_trials: number of trials before abandoning a source
    :return: best solution found by the algorithm
    """

    num_variables = len(lb)
    colony = np.zeros((colony_size, num_variables))
    fitness = np.zeros(colony_size)
    alltrials=np.zeros(colony_size)
    trials=num_trials
    # initialize the colony
    for i in range(colony_size):
        colony[i, :] = np.random.uniform(lb, ub)
        fitness[i] = obj_function(func,colony[i, :])
        alltrials[i]=trials

    # find the best solution
    best_solution = colony[np.argmin(fitness), :]
    best_fitness = np.min(fitness)

    # main loop of the algorithm
    for iter in range(max_iter):

        # employed bees phase
        for i in range(colony_size):
            k = np.random.randint(colony_size)
            while k == i:
                k = np.random.randint(colony_size)

            phi = np.random.uniform(-1, 1, num_variables)
            new_solution = colony[i, :] + phi * (colony[i, :] - colony[k, :])
            new_solution = np.maximum(new_solution, lb)
            new_solution = np.minimum(new_solution, ub)
            new_fitness = obj_function(func,new_solution)

            if new_fitness < fitness[i]:
                colony[i, :] = new_solution
                fitness[i] = new_fitness
                alltrials[i] = trials
            else:
                alltrials[i] -= 1
        for i in range(colony_size):
            fit.append([iter,0,i,fitness[i]])


        # onlooker bees phase
        i = 0
        t = 0
        while i < colony_size:
            if np.random.uniform(0, 1) < fitness[i] / np.sum(fitness):
                t += 1
                k = np.random.randint(colony_size)
                while k == i:
                    k = np.random.randint(colony_size)

                phi = np.random.uniform(-1, 1, num_variables)
                new_solution = colony[i, :] + phi * (colony[i, :] - colony[k, :])
                new_solution = np.maximum(new_solution, lb)
                new_solution = np.minimum(new_solution, ub)
                new_fitness = obj_function(func,new_solution)

                if new_fitness < fitness[i]:
                    colony[i, :] = new_solution
                    fitness[i] = new_fitness
                    alltrials[i] = trials
                else:
                    alltrials[i] -= 1

            i += 1
        for i in range(colony_size):
            fit.append([iter,1,i,fitness[i]])


        # scout bees phase
        for i in range(colony_size):
            if alltrials[i] <= 0:
                colony[i, :] = np.random.uniform(lb, ub)
                fitness[i] = obj_function(func,colony[i, :])
                alltrials[i] = trials
        for i in range(colony_size):
            fit.append([iter,2,i,fitness[i]])

        # update the best solution
        if np.min(fitness) < best_fitness:
            best_solution = colony[np.argmin(fitness), :]
            best_fitness = np.min(fitness)
    return best_solution


# In[ ]:


def driverBee(it,gen,lb,ub,func):
    global fit
    fit=[]
    labels1=["Iteration","Phase","Id","Fitness"]
    fit.append(labels1)
    print(abc_algorithm(func,fitness,[lb,lb],[ub,ub],it,gen))
    pd.DataFrame(fit).to_csv("GeneratedData/bee_fun_fitness.csv",header=False,index=False)


# # Bat

# In[ ]:


class BatAlgorithm():
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function,func):
        info=[N_Gen,NP]
        self.D = D  #dimension
        self.NP = NP  #population size 
        self.N_Gen = N_Gen  #generations
        self.A = A  #loudness
        self.r = r  #pulse rate
        self.Qmin = Qmin  #frequency min
        self.Qmax = Qmax  #frequency max
        self.Lower = Lower  #lower bound
        self.Upper = Upper  #upper bound

        self.f_min = 0.0  #minimum fitness

        self.Lb = [0] * self.D  #lower bound
        self.Ub = [0] * self.D  #upper bound
        self.Q = [0] * self.NP  #frequency

        self.v = [[0 for i in range(self.D)] for j in range(self.NP)]  #velocity
        self.Sol = [[0 for i in range(self.D)] for j in range(self.NP)]  #population of solutions
        self.Fitness = [0] * self.NP  #fitness
        self.best = [0] * self.D  #best solution
        self.Fun = function


    def best_bat(self):
        i = 0
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
            self.Fitness[i] = self.Fun(func, self.Sol[i])
        self.best_bat()

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def move_bat(self):
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]

        self.init_bat()

        for t in range(self.N_Gen):
            for i in range(self.NP):
                rnd = np.random.uniform(0, 1)
                self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * rnd
                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] -
                                                   self.best[j]) * self.Q[i]
                    S[i][j] = self.Sol[i][j] + self.v[i][j]

                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                self.Ub[j])

                rnd = np.random.random_sample()

                if rnd > self.r:
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                self.Ub[j])

                Fnew = self.Fun(func, S[i])

                rnd = np.random.random_sample()

                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew

                fit.append([t,self.Fitness[i]])
        print(self.f_min," ",self.best)


# In[ ]:


def driverBat(it,gen,lb,ub,func):
    global fit
    fit=[]
    labels1=["Iteration","Fitness"]
    fit.append(labels1)
    print(func)
    Algorithm = BatAlgorithm(D=2, NP=it, N_Gen=gen, A=5, r=0.5, Qmin=0.0, Qmax=2.0,
                             Lower=lb,Upper=ub, function=fitness,func=func)
    Algorithm.move_bat()
    pd.DataFrame(fit).to_csv("GeneratedData/bat_fun_fitness.csv",header=False,index=False)


# # Fish

# In[ ]:


class Fish():

    def __init__(self, objective_function, positions):
        self.fitness_function = objective_function
        self.current_position = positions
        self.weight = iterations_number / 2.0
        self.fitness = np.inf
        self.delta_fitness = 0
        self.delta_position = []

    def evaluate(self):
        new_fitness = self.fitness_function.calculate_fitness(self.current_position)
        self.fitness = new_fitness

    def update_position_individual_movement(self, step_ind):
        new_positions = []
        for pos in self.current_position:
            new = pos + (step_ind * np.random.uniform(-1, 1))
            if new > self.fitness_function.upper_bound:
                new = self.fitness_function.upper_bound
            elif new < self.fitness_function.lower_bound:
                new = self.fitness_function.lower_bound
            new_positions.append(new)
        assert len(new_positions) == len(self.current_position)

        new_fitness = self.fitness_function.calculate_fitness(new_positions)
        if new_fitness < self.fitness:
            self.delta_fitness = abs(new_fitness - self.fitness)
            self.fitness = new_fitness
            self.delta_position = [x - y for x, y in zip(new_positions, self.current_position)]
            self.current_position = list(new_positions)
        else:
            self.delta_position = [0] * dimensions
            self.delta_fitness = 0

    def feed(self, max_delta_fitness):
        print("self.delta_fitness",self.delta_fitness," ;max_delta_fitness", max_delta_fitness)
        if max_delta_fitness != 0:
            self.weight = self.weight + (self.delta_fitness / max_delta_fitness)
        else:
            self.weight = 1

    def update_position_collective_movement(self, sum_delta_fitness):
        collective_instinct = []
        for i, _ in enumerate(self.delta_position):
            collective_instinct.append(self.delta_position[i] * self.delta_fitness)
        if sum_delta_fitness != 0:
            collective_instinct = [val / sum_delta_fitness for val in collective_instinct]

        new_positions = []
        for i, _ in enumerate(self.current_position):
            new = self.current_position[i] + collective_instinct[i]
            if new > self.fitness_function.upper_bound:
                new = self.fitness_function.upper_bound
            elif new < self.fitness_function.lower_bound:
                new = self.fitness_function.lower_bound
            new_positions.append(new)

        assert len(new_positions) == len(self.current_position)
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
        # volitive_step = [x - y for x, y in zip(self.current_position,barycenter)] / np.linalg.norm([self.current_position, barycenter])
        # volitive_step = np.random.uniform(0, 1) * step_vol * volitive_step * search_operator
        # new_positions = [x + y for x, y in zip(self.current_position, volitive_step)]

        assert len(new_positions) == len(self.current_position)
        self.current_position = list(new_positions)


# In[ ]:


class FSS():

    def __init__(self, objective_function):
        self.function = objective_function
        self.dimensions = 2
        self.iterations_number = iterations_number
        self.num_of_individuos = num_of_individuos
        self.cluster = []
        self.global_best = float('inf')
        self.global_best_position = []

        # Params
        self.total_weight = 1 * self.num_of_individuos
        self.initial_step_ind = 0.5
        self.final_step_ind = 0.01
        self.step_ind = self.initial_step_ind * (objective_function.upper_bound - objective_function.lower_bound)
        self.initial_step_vol = 0.5
        self.final_step_vol = 0.01
        self.step_vol = self.initial_step_vol * (objective_function.upper_bound - objective_function.lower_bound)
        self.list_global_best_values = []

    def search(self):
        self._initialize_cluster()
        for i in range(self.iterations_number):


            self.evaluate_cluster()
            self.updates_optimal_solution()


            self.apply_individual_movement()
            self.evaluate_cluster()
            self.updates_optimal_solution()
            print("AFTER IND MOVEMENT")
            for j in range(0,len(self.cluster)):
                fit.append([i,0,j,self.cluster[j].fitness])
                print(j,self.cluster[j].current_position)

            print("iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))
            print("#############################################################################")

            self.apply_feeding()

            self.apply_instintive_collective_movement()
            print("AFTER instintive_collective_movement")
            for j in range(0,len(self.cluster)):
                fit.append([i,1,j,self.cluster[j].fitness])
                print(j,self.cluster[j].current_position)

            print("iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))
            print("#############################################################################")

            self.apply_collective_volitive_movement()
            print("AFTER collective_volitive_movement")
            for j in range(0,len(self.cluster)):
                fit.append([i,2,j,self.cluster[j].fitness])
                print(j,self.cluster[j].current_position)

            print("iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))
            print("#############################################################################")


            self.update_step(i)
            self.update_total_weight()

            self.evaluate_cluster()
            self.updates_optimal_solution()
            self.list_global_best_values.append(self.global_best)
            print("END ITERATION")
            for j in range(0,len(self.cluster)):
                print(j,self.cluster[j].current_position)

            print("00000 iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))

    def update_total_weight(self):
        self.total_weight = sum([fish.weight for fish in self.cluster])

    def _initialize_cluster(self):
        self.cluster = []
        for _ in range(self.num_of_individuos):
            fish = Fish(
                positions=[self._get_random_number() for _ in range(dimensions)],
                objective_function=self.function
            )
            self.cluster.append(fish)

    def evaluate_cluster(self):
        for fish in self.cluster:
            fish.evaluate()

    def updates_optimal_solution(self):
        for i in range(0,len(self.cluster)):
            if self.cluster[i].fitness < self.global_best:
                self.global_best = self.cluster[i].fitness
                self.global_best_position = list(self.cluster[i].current_position)
                print(i," ",self.cluster[i].current_position," ",self.cluster[i].fitness)


    def apply_individual_movement(self):
        for fish in self.cluster:
            fish.update_position_individual_movement(self.step_ind)

    def apply_feeding(self):
        max_delta_fitness = max([fish.delta_fitness for fish in self.cluster])
        for fish in self.cluster:
            fish.feed(max_delta_fitness)

    def apply_instintive_collective_movement(self):
        sum_delta_fitness = sum([fish.delta_fitness for fish in self.cluster])

        for fish in self.cluster:
            fish.update_position_collective_movement(sum_delta_fitness)

    def _calculate_barycenter(self):
        sum_weights = sum([fish.weight for fish in self.cluster])
        sum_position_and_weights = [[x * fish.weight for x in fish.current_position] for fish in self.cluster]
        sum_position_and_weights = np.sum(sum_position_and_weights, 0)
        return [s / sum_weights for s in sum_position_and_weights]

    def apply_collective_volitive_movement(self):
        barycenter = self._calculate_barycenter()
        current_total_weight = sum([fish.weight for fish in self.cluster])
        search_operator = -1 if current_total_weight > self.total_weight else 1
        for fish in self.cluster:
            fish.update_position_volitive_movement(barycenter, self.step_vol, search_operator)

    def update_step(self, current_i):
        self.step_ind = self.initial_step_ind - current_i * float(
            self.initial_step_ind - self.final_step_ind) / iterations_number
        self.step_vol = self.initial_step_vol - current_i * float(
            self.initial_step_vol - self.final_step_vol) / iterations_number

    def _get_random_number(self):
        return np.random.uniform(self.function.lower_bound, self.function.upper_bound)


# In[ ]:


from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
# np.random.seed(42)
class AFunction:


    upper_bound = 1
    lower_bound = -1

    def calculate_fitness(self, position):
        pass
class FishFunc(AFunction):  
    def __init__(self,fun,lb,ub):
        AFunction.upper_bound = ub
        AFunction.lower_bound = lb
        AFunction.fun=fun

    def calculate_fitness(self, x):
        return fitness(self.fun,x)
def driverFish(it,gen,lb,ub,fun):
    global fit
    fit=[]
    labels1=["Iteration","Phase","Id","Fitness"]
    fit.append(labels1)
    global iterations_number
    iterations_number = gen
    global num_of_individuos 
    num_of_individuos = it
    global dimensions
    dimensions=2
    fss = FSS(FishFunc(fun,lb,ub))
    print(func)
    fss.search()
    best_fitness=fss.list_global_best_values
    pd.DataFrame(fit).to_csv("GeneratedData/fish_fun_fitness.csv",header=False,index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Dash

# In[ ]:


server = Flask(__name__)
app = dash.Dash(__name__, server=server)
CORS(app.server)
app.layout = html.Div(
    [
        html.Link(href='/styles.css', rel='stylesheet'),
        html.H1("Set the parameters of the optimization algorithm", className='custom-header'),
        html.H2("The higher the values, the more time it will take to train."),
        html.Div(
        className="part1",
        children=[
        # Number of iterations section
        html.Div(
            className='slider-wrapper',
            children=[
                html.Label('Choose the number of individuals:'),
                html.Br(),html.Br(),
                dcc.Slider(
                    id='slider-1',
                    min=5,
                    max=1000,
                    step=1,
                    value=50,
                    marks={
                        5: {'label': '5'},
                        100: {'label': '100'},
                        200: {'label': '200'},
                        300: {'label': '300'},
                        400: {'label': '400'},
                        500: {'label': '500'},
                        600: {'label': '600'},
                        700: {'label': '700'},
                        800: {'label': '800'},
                        900: {'label': '900'},
                        1000: {'label': '1000'},
                    },
                    className='custom-slider'
                ),
                html.Br(),
                html.Div(id='slider-output-1', className='slider-output')
            ]
        ),

        # Number of generations section
        html.Div(
            className='slider-wrapper',
            children=[
                html.Label('Choose the number of generations:'),
                html.Br(),html.Br(),
                dcc.Slider(
                    id='slider-2',
                    min=5,
                    max=1000,
                    step=1,
                    value=75,
                    marks={
                       5: {'label': '5'},
                        100: {'label': '100'},
                        200: {'label': '200'},
                        300: {'label': '300'},
                        400: {'label': '400'},
                        500: {'label': '500'},
                        600: {'label': '600'},
                        700: {'label': '700'},
                        800: {'label': '800'},
                        900: {'label': '900'},
                        1000: {'label': '1000'},
                    },
                    className='custom-slider'
                ),
                html.Br(),
                html.Div(id='slider-output-2', className='slider-output')
            ]
        ),

        # Selecting the bounds section
        html.Div(
            [
                html.Div(
                    children=[
                        html.P(''),
                        html.Label('Enter the lower bound:'),
                        dcc.Input(id='lb', type='number', value=0),
                    ]
                ),
                html.Br(),
                html.Div(
                    children=[
                        html.Label('Enter the upper bound:'),
                        dcc.Input(id='ub', type='number', value=0),
                    ]
                ),
                html.Br(),
                html.Div(
                    id='output',
                    children=[
                        html.P(id='lbo'),
                        html.P(id='ubo'),
                        html.P(id='result',className="result")
                    ]
                )
            ]
        )]),
        html.Div(
        className="part2",
        children=[
        html.Label('Enter the fitness function'),
        html.Br(),
        html.Label('using the below buttons'),
        html.Br(),html.Br(),
        # Calculator section
        html.Link(href='/assets/styles.css', rel='stylesheet'),
        html.Div(id='display', className='display'),
            html.Br(),
                # Calculator section
                html.Div(
                    [
                        html.Button('7', id='btn-7', className='num-button bigger-button'),
                        html.Button('8', id='btn-8', className='num-button bigger-button'),
                        html.Button('9', id='btn-9', className='num-button bigger-button'),
                        html.Button('+', id='btn-+', className='operator-button bigger-button'),
                        html.Button('-', id='btn--', className='operator-button bigger-button'),
                    ],
                    className='button-row'
                ),
                html.Div(
                    [
                        html.Button('4', id='btn-4', className='num-button bigger-button'),
                        html.Button('5', id='btn-5', className='num-button bigger-button'),
                        html.Button('6', id='btn-6', className='num-button bigger-button'),
                        html.Button('(', id='btn-(', className='operator-button bigger-button'),
                        html.Button(')', id='btn-)', className='operator-button bigger-button')
                    ],
                    className='button-row'
                ),
                html.Div(
                    [
                        html.Button('1', id='btn-1', className='num-button bigger-button'),
                        html.Button('2', id='btn-2', className='num-button bigger-button'),
                        html.Button('3', id='btn-3', className='num-button bigger-button'),
                        html.Button('*', id='btn-*', className='operator-button bigger-button'),
                        html.Button('/', id='btn-/', className='operator-button bigger-button')
                    ],
                    className='button-row'
                ),
                html.Div(
                    [
                        html.Button('0', id='btn-0', className='num-button bigger-button'),
                        html.Button('.', id='btn-decimal', className='num-button bigger-button dec-button'),
                        html.Button('cos', id='btn-cos', className='operator-button bigger-button'),
                        html.Button('sin', id='btn-sin', className='operator-button bigger-button'),
                        html.Button('tan', id='btn-tan', className='operator-button bigger-button'),
                    ],
                    className='button-row'
                ),
                html.Div(
                    [
                        html.Button('pow', id='btn-**', className='operator-button bigger-button'),
                        html.Button('exp', id='btn-exp', className='operator-button bigger-button'),
                        html.Button('sqrt', id='btn-sqrt', className='operator-button bigger-button'),
                        html.Button('pi', id='btn-pi', className='operator-button bigger-button'),
                        html.Button('e', id='btn-e', className='operator-button bigger-button'),
                    ],
                    className='button-row'
                ),
                html.Div(
                    [
                        html.Button('X', id='btn-X', className='operator-button bigger-button'),
                        html.Button('Y', id='btn-Y', className='operator-button bigger-button'),
                        html.Button('C', id='btn-clear', className='clear-button bigger-button'),
                        html.Button('Submit', id='btn-equals', className='equals-button biggest-button', n_clicks=0)
                    ],
                    className='button-row'
                )
            ]
        )])



@app.callback(
    Output('slider-output-1', 'children'),
    Output('slider-output-2', 'children'),
    [Input('slider-1', 'value'),
     Input('slider-2', 'value')]
)
def update_output(value1, value2):
    return f"You have selected: Number of individuals: {value1}", f"You have selected: Number of generations: {value2}"


@app.callback(
    Output('lbo', 'children'),
    Output('ubo', 'children'),
    Output('result', 'children'),
    Input('lb', 'value'),
    Input('ub', 'value')
)
def validate_bounds(lower_bound, upper_bound):
    if lower_bound >= upper_bound:
        return "", "", "Lower bound should be strictly smaller than the upper bound."
    else:
        return f"Lower bound: {lower_bound}", f"Upper bound: {upper_bound}", ""


@app.callback(
    Output('display', 'children'),
    [Input('slider-1', 'value'),
     Input('slider-2', 'value'),
     Input('lb', 'value'),
     Input('ub', 'value'),
     Input('btn-1', 'n_clicks'),
     Input('btn-2', 'n_clicks'),
     Input('btn-3', 'n_clicks'),
     Input('btn-4', 'n_clicks'),
     Input('btn-5', 'n_clicks'),
     Input('btn-6', 'n_clicks'),
     Input('btn-7', 'n_clicks'),
     Input('btn-8', 'n_clicks'),
     Input('btn-9', 'n_clicks'),
     Input('btn-0', 'n_clicks'),
     Input('btn-+', 'n_clicks'),
     Input('btn--', 'n_clicks'),
     Input('btn-*', 'n_clicks'),
     Input('btn-/', 'n_clicks'),
     Input('btn-(', 'n_clicks'),
     Input('btn-)', 'n_clicks'),
     Input('btn-X', 'n_clicks'),
     Input('btn-Y', 'n_clicks'),
     Input('btn-cos', 'n_clicks'),
     Input('btn-sin', 'n_clicks'),
     Input('btn-tan', 'n_clicks'),
     Input('btn-**', 'n_clicks'),
     Input('btn-exp', 'n_clicks'),
     Input('btn-sqrt', 'n_clicks'),
     Input('btn-pi', 'n_clicks'),
     Input('btn-e', 'n_clicks'),
     Input('btn-decimal', 'n_clicks'),
     Input('btn-clear', 'n_clicks'),
     Input('btn-equals', 'n_clicks')],
    [State('display', 'children')]
)
def update_display(it,gen,lb,ub,btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9, btn0,
                   btn_plus, btn_minus, btn_multiply, btn_divide, btn_parin, btn_parout,
                   btn_cos, btn_sin, btn_tan, btn_exp, btn_pow, btn_e, btn_pi, btn_sqrt,
                   btn_X, btn_Y, btn_decimal, btn_clear, btn_equals, current_display):
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_display

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'btn-clear':
        return '0'

    if current_display == '0':
        current_display = ''

    if current_display is None:
        current_display = ''

    if trigger_id in ['btn-1', 'btn-2', 'btn-3', 'btn-4', 'btn-5', 'btn-6', 'btn-7', 'btn-8', 'btn-9', 'btn-0']:
        current_display += trigger_id.split('-')[1]
    elif trigger_id in ['btn-+', 'btn-*', 'btn-/', 'btn-(', 'btn-)', 'btn-X', 'btn-Y', 'btn-e', 'btn-pi', 'btn-**']:
        current_display += ' ' + trigger_id.split('-')[1] + ' '
    elif trigger_id in ['btn-exp', 'btn-sqrt', 'btn-cos', 'btn-sin', 'btn-tan']:
        current_display += ' ' + trigger_id.split('-')[1] + '( '
    elif trigger_id == 'btn-decimal':
        current_display += '.'
    elif trigger_id == "btn--":
        current_display += ' - '
    elif trigger_id == 'btn-equals':
        current_display=check(it,gen,lb,ub,current_display)

        if current_display!="error":
            current_display="Redirecting ..."
            parent_folder = os.path.dirname(os.getcwd())
            vis_html_path = os.path.join(parent_folder, 'fun_fit_2.html')
            webbrowser.open(vis_html_path)
            #webbrowser.open('vis.html') 




    return current_display
'''
@app.callback(Output('url', 'pathname'), [Input('btn-equals', 'n_clicks')])
def redirect_to_vis_html(n_clicks):
    if n_clicks and n_clicks > 0:
        return '/vis.html'
    else:
        return '/'

@app.callback(Output('hidden-div', 'children'), [Input('btn-equals', 'n_clicks')])
def redirect_to_interface(n_clicks):
    if n_clicks and n_clicks > 0:
        #return html.Iframe(src='vis.html', style={'width': '100%', 'height': '500px'})
        #webbrowser.get().open_new('vis.html')
        return webbrowser.get().open_new('vis.html')

'''        


if __name__ == '__main__':
    #webbrowser.open_new('http://127.0.0.1:7771/')

    app.run_server(debug=False,port=7771)


# In[ ]:


check(10,50,0,2,'cos(X)*(sin(Y)**2)')


# In[ ]:





# In[ ]:


def calc(a,b):
    if(a<0 or b<0):
        return"error"
    else:
        c=calc2(a,b)
        return"done"
def calc2(a,b):
    return str(a+b)


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Addition App"),
    html.Div([
        dcc.Input(id='input1', type='number', placeholder='Enter number 1'),
        dcc.Input(id='input2', type='number', placeholder='Enter number 2'),
        html.Button('Calculate', id='button', n_clicks=0),
    ]),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    [Input('button', 'n_clicks')],
    [State('input1', 'value'), State('input2', 'value')]
)
def calculate_sum(n_clicks, input1, input2):
    if n_clicks > 0:
        if input1 is not None and input2 is not None:
            try:
                sum_val = calc(input1,input2)
                return f'The sum is {sum_val}'
            except ValueError:
                return 'Invalid input! Please enter valid numbers.'
        else:
            return 'Please enter both numbers.'
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=False,port=5500)


# In[ ]:




