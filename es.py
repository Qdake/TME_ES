from deap import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from scipy.optimize import minimize

from plot import *

import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

IND_SIZE = 2
MIN_VALUE = 15
MAX_VALUE = 25
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3


# DEAP contient plusieurs fonctions classiques de test. Attention, ce framework renvoie un tuple et non une valeur unique. 
# pour tracer la fonction, il faut passer par la fonction indiquée ci-dessous.
def ma_func(x):
    return benchmarks.ackley(x)[0]

# La fonction doit être minimisée, le poids est donc de -1
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Génération d'un individu avec une distribution uniforme dans les bornes indiquées
def generateES(icls, scls, size, imin, imax, smin, smax):
    '''   icls  est la classe Individual
           scls  est la class Strategie
           size  est IND_SIZE
    '''
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

# Fonction utilisée pour mettre une borne inférieure à la mutation
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


# Valeurs par défaut, 
toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", benchmarks.ackley)

# Application de la borne minimale
toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

def launch_es(mu=100, lambda_=200, cxpb=0.4, mutpb=0.3, ngen=1000, display=False, verbose=False):
    '''mu – The number of individuals to select for the next generation.  
        lambda_ – The number of children to produce at each generation.  
        cxpb – The probability that an offspring is produced by crossover. 
        mutpb – The probability that an offspring is produced by mutation.  
        ngen – The number of generation.
    '''
    # Initialisation 
    random.seed()

    population = toolbox.population(n=mu)
    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
        

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Boucle de l'algorithme évolutionniste
    for gen in range(1, ngen + 1):
    
        ### A completer pour implementer un ES en affichant regulièrement les resultats a l'aide de la fonction plot_results fournie ###
        ### Vous pourrez tester plusieurs des algorithmes implémentés dans DEAP pour générer une population d'"enfants" 
        ### à partir de la population courante et pour sélectionner les géniteurs de la prochaine génération
        
       #offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
      
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)
            
        population[:] = toolbox.select(population + offspring, mu)
        
        plot_results(ma_func, population)
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, halloffame

    