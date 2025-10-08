#!/usr/bin/env python
# coding: utf-8

# # Michalewicz Activation Function

# In[1]:


#Michalewicz multimodal function
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

# objective function
def Michalewicz(position):
 return -1 * ( (np.sin(position[0]) * np.sin((1 * position[0]**2) / np.pi)**20) + (np.sin(position[1]) * np.sin((2 * position[1]**2) / np.pi)**20) )


# # WOLF 

# In[2]:


tocsv=[]
fit=[]
labels=["Iteration","Id","X","Y"]
labels1=["Iteration","Fitness"]
tocsv.append(labels)
fit.append(labels1)


# In[3]:


import random
import copy    # array-copying convenience
import sys     # max float
import numpy as np

# wolf class
class wolf:
  def __init__(self, fitness, dim, minx, maxx, seed):
    self.id=seed
    self.rnd = random.Random(seed)
    self.position = [0.0 for i in range(dim)]

    for i in range(dim):
      self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
    self.fitness = fitness(self.position) # curr fitness



# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx):
    rnd = random.Random(0)
    info=[max_iter,n]
    tocsv.append(info)
    # create n random wolves
    population = [ wolf(fitness, dim, minx, maxx, i) for i in range(n)]
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
        print()
        print()
        print()
        print("ITER : ",Iter)

        for i in range(n):
          tocsv.append([Iter,population[i].id,population[i].position[0]*200,population[i].position[1]*200])
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

            # fitness calculation of new solution
            fnew = fitness(Xnew)

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

print("\nBegin grey wolf optimization on  function Michalewicz\n")
dim = 2
fitness = Michalewicz


print("Goal is to minimize Michalewicz's function in " + str(dim) + " variables")
print("Function has known min = 0.0 at (", end="")
for i in range(dim-1):
  print("0, ", end="")
print("0)")

num_particles = 30
max_iter = 50

print("Setting num_particles = " + str(num_particles))
print("Setting max_iter    = " + str(max_iter))
print("\nStarting GWO algorithm\n")



best_position = gwo(fitness, max_iter, num_particles, dim, 0, 3)

print("\nGWO completed\n")
print("\nBest solution found:")
print(["%.6f"%best_position[k] for k in range(dim)])
err = fitness(best_position)
print("fitness of best solution = %.6f" % err)

print("\nEnd GWO for Michalewicz\n")


# In[4]:


pd.DataFrame(tocsv).to_csv("wolf_Michalewicz.csv",header=False,index=False)


# In[5]:


pd.DataFrame(fit).to_csv("fitness_Michalewicz_wolf.csv",header=False,index=False)


# # BEE STARTS HERE

# In[6]:


tocsv=[]
fit=[]
labels=["Iteration","Phase","Id","X","Y"]
labels1=["Iteration","Phase","Id","Fitness"]
tocsv.append(labels)
fit.append(labels1)


# In[7]:


import numpy as np

def abc_algorithm(obj_function, lb, ub, colony_size=10, max_iter=50 ,num_trials=5):
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
    tocsv.append([max_iter,"",colony_size])
    colony = np.zeros((colony_size, num_variables))
    fitness = np.zeros(colony_size)
    alltrials=np.zeros(colony_size)
    trials=num_trials
    # initialize the colony
    for i in range(colony_size):
        colony[i, :] = np.random.uniform(lb, ub)
        fitness[i] = obj_function(colony[i, :])
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
            new_fitness = obj_function(new_solution)

            if new_fitness < fitness[i]:
                colony[i, :] = new_solution
                fitness[i] = new_fitness
                alltrials[i] = trials
            else:
                alltrials[i] -= 1
        for i in range(colony_size):
            tocsv.append([iter,0,i,colony[i,0]*200,colony[i,1]*200])
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
                new_fitness = obj_function(new_solution)

                if new_fitness < fitness[i]:
                    print("yas",i)
                    colony[i, :] = new_solution
                    fitness[i] = new_fitness
                    alltrials[i] = trials
                else:
                    alltrials[i] -= 1

            i += 1
        for i in range(colony_size):
            tocsv.append([iter,1,i,colony[i,0]*200,colony[i,1]*200])
            fit.append([iter,1,i,fitness[i]])


        # scout bees phase
        for i in range(colony_size):
            if alltrials[i] <= 0:
                colony[i, :] = np.random.uniform(lb, ub)
                fitness[i] = obj_function(colony[i, :])
                alltrials[i] = trials
        for i in range(colony_size):
            tocsv.append([iter,2,i,colony[i,0]*200,colony[i,1]*200])
            fit.append([iter,2,i,fitness[i]])

        # update the best solution
        if np.min(fitness) < best_fitness:
            best_solution = colony[np.argmin(fitness), :]
            best_fitness = np.min(fitness)
    return best_solution


# In[8]:


print(abc_algorithm(Michalewicz,[0,0],[3,3]))


# In[9]:


pd.DataFrame(tocsv).to_csv("bee_Michalewicz.csv",header=False,index=False)


# In[10]:


pd.DataFrame(fit).to_csv("fitness_bee_Mich.csv",header=False,index=False)


# # BAT

# In[11]:


# get_ipython().system('pip install BatAlgorithm')


# In[12]:


import random

from BatAlgorithm import *


# In[13]:


from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid

def Fun(D, sol):
    return -1 * ( (np.sin(sol[0]) * np.sin((1 * sol[0]**2) / np.pi)**20) + (np.sin(sol[1]) * np.sin((2 * sol[1]**2) / np.pi)**20) )

#For reproducive results
random.seed(20)

for i in range(0,1000):
    Algorithm = BatAlgorithm(2, 30, 50, 1, 0.1, 0.0, 2.0, -5.0, 5.0, Fun)
    Algorithm.move_bat()


# In[14]:


tocsv=[]
fit=[]
labels=["Iteration","Id","X","Y"]
labels1=["Iteration","Fitness"]
tocsv.append(labels)
fit.append(labels1)


# In[15]:


import random 
import numpy as np

class BatAlgorithm():
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function):
        info=[N_Gen,NP]
        tocsv.append(info)
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
            self.Fitness[i] = self.Fun(self.D, self.Sol[i])
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

                Fnew = self.Fun(self.D, S[i])

                rnd = np.random.random_sample()

                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew
                tocsv.append([t,i,self.Sol[i][0]*200,self.Sol[i][1]*200])
                fit.append([t,self.Fitness[i]])
        print(self.f_min)


# In[16]:


Algorithm = BatAlgorithm(2, 50, 50, 1, 0.1, 0.0, 2.0, 0, 3, Fun)
Algorithm.move_bat()


# In[17]:


pd.DataFrame(tocsv).to_csv("bat_Mich.csv",header=False,index=False)


# In[18]:


pd.DataFrame(fit).to_csv("fitness_bat_Mich.csv",header=False,index=False)


# # Fish

# In[19]:


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
            #collective_instinct.append(self.delta_position[i] * self.delta_fitness)
            collective_instinct.append((self.delta_position[i])/3 * self.delta_fitness)

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


# In[20]:


iterations_number = 50
num_of_individuos = 15
dimensions = 2


# In[21]:


import copy
import numpy as np
import functools
import random
import math
import numpy as np


np.random.seed(42)

class FSS():

    def __init__(self, objective_function):
        self.function = objective_function
        self.dimensions = dimensions
        self.iterations_number = iterations_number
        self.num_of_individuos = num_of_individuos
        self.cluster = []
        self.global_best = float('inf')
        self.global_best_position = []

        # Params
        self.total_weight = 1 * self.num_of_individuos
        self.initial_step_ind = 1.5
        self.final_step_ind = 0.05
        self.step_ind = self.initial_step_ind * (objective_function.upper_bound - objective_function.lower_bound)
        self.initial_step_vol = 1.5
        self.final_step_vol = 0.05
        self.step_vol = self.initial_step_vol * (objective_function.upper_bound - objective_function.lower_bound)
        self.list_global_best_values = []

    def search(self):
        self._initialize_cluster()
        tocsv.append([self.iterations_number,"",self.num_of_individuos])
        for i in range(self.iterations_number):


            self.evaluate_cluster()
            self.updates_optimal_solution()


            self.apply_individual_movement()
            self.evaluate_cluster()
            self.updates_optimal_solution()
            print("AFTER IND MOVEMENT")
            for j in range(0,len(self.cluster)):
                tocsv.append([i,0,j,self.cluster[j].current_position[0]*200,
                              self.cluster[j].current_position[1]*200,self.cluster[j].weight])
                fit.append([i,0,j,self.cluster[j].fitness])
                print(j,self.cluster[j].current_position," ",self.cluster[j].fitness)

            print("iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))
            print("#############################################################################")

            self.apply_feeding()

            self.apply_instintive_collective_movement()
            print("AFTER instintive_collective_movement")
            for j in range(0,len(self.cluster)):
                tocsv.append([i,1,j,self.cluster[j].current_position[0]*200,
                              self.cluster[j].current_position[1]*200,self.cluster[j].weight])
                fit.append([i,1,j,self.cluster[j].fitness])
                print(j,self.cluster[j].current_position," ",self.cluster[j].fitness)

            print("iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))
            print("#############################################################################")

            self.apply_collective_volitive_movement()
            print("AFTER collective_volitive_movement")
            for j in range(0,len(self.cluster)):
                tocsv.append([i,2,j,self.cluster[j].current_position[0]*200,
                              self.cluster[j].current_position[1]*200,self.cluster[j].weight])
                fit.append([i,2,j,self.cluster[j].fitness])
                print(j,self.cluster[j].current_position," ",self.cluster[j].fitness)

            print("iter: {} = cost: {} , pos: {}".format(i, self.global_best,self.global_best_position))
            print("#############################################################################")


            self.update_step(i)
            self.update_total_weight()

            self.evaluate_cluster()
            self.updates_optimal_solution()
            self.list_global_best_values.append(self.global_best)
            print("END ITERATION")
            for j in range(0,len(self.cluster)):
                print(j,self.cluster[j].current_position," ",self.cluster[j].fitness)

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


# In[22]:


SIMULATIONS =1
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(42)

def rosenbrocks():
	print('Mic')

	best_fitness = []
	for _ in range(SIMULATIONS):
		fss = FSS(Mic())
		fss.search()
		best_fitness.append(fss.list_global_best_values)

	average_best_fitness = np.sum(np.array(best_fitness), axis=0) / SIMULATIONS



# In[23]:


tocsv=[]
labels=["Iteration","Phase","Id","X","Y","Weight"]
tocsv.append(labels)
fit=[]
labels1=["Iteration","Phase","Id","Fitness"]
fit.append(labels1)


# In[24]:


class AFunction:


    upper_bound = 1
    lower_bound = -1

    def calculate_fitness(self, position):
        pass
class Rosenbrocks(AFunction):

    def __init__(self):
        AFunction.upper_bound = 30
        AFunction.lower_bound = -30

    def calculate_fitness(self, x):
        sum_ = 0.0
        for i in range(1, len(x) - 1):
            sum_ += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return sum_
class Mic(AFunction):  
    def __init__(self):
        AFunction.upper_bound = 3
        AFunction.lower_bound = 0

    def calculate_fitness(self, x):
        return -1 * ( (np.sin(x[0]) * np.sin((1 * x[0]**2) / np.pi)**20) + (np.sin(x[1]) * np.sin((2 * x[1]**2) / np.pi)**20) )


# In[25]:


rosenbrocks()


# In[26]:


import pandas as pd
pd.DataFrame(tocsv).to_csv("fish_mic.csv",header=False,index=False)


# In[27]:


pd.DataFrame(fit).to_csv("fish_mic_fitness.csv",header=False,index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
# meta_algorithms.py

def run_algorithm():
    # Example: just run GWO on Michalewicz function
   # import from within this file or adjust
   def run_algorithm():
    # Run your Michalewicz algorithm
    result_value = 0.12345  # Example result
    return f"The best fitness value is: {result_value}"


    dim = 2
    num_particles = 30
    max_iter = 50
    best_position = gwo(Michalewicz, max_iter, num_particles, dim, 0, 3)
    err = Michalewicz(best_position)

    return f"Best position: {best_position}, Fitness: {err}"




