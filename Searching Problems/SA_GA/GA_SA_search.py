#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import random
import sys
import math
from numpy.random import choice
import matplotlib.pyplot as plt
import timeit
import csv




if len(sys.argv) != 5:
    print("ERROR: Not enough or too many input arguments.")
    exit()


fn = sys.argv[1]
alg = sys.argv[2]
P1 = int(sys.argv[3])
P2 = float(sys.argv[4])




def write_to_csv(values, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for value in values:
            csv_writer.writerow([value])

data = pd.read_csv(fn, header=None)


# In[2]:


dic = {}
places = []

for index, row in data.iterrows():
    dic[row[0]] = (row[1],row[2])
    places.append(row[0])
    
def eucDist(p1, p2):
    x1 = dic[p1][0]
    y1 = dic[p1][1]
    
    x2 = dic[p2][0]
    y2 = dic[p2][1]
    
    deltX = x2 - x1
    deltY = y2 - y1
    
    return np.sqrt(deltX**2 + deltY**2)


def pathCost(p):
    pCost = 0
    for i in range(len(p)):
        if i != (len(p)-1):
            pCost += eucDist(p[i], p[i+1])
        else: 
            pCost += eucDist(p[i], p[0])
            
    return pCost



def swapValues(path): 
    lst = path[:]
    index1 = random.randint(1, len(lst) - 1)
    index2 = random.randint(1, len(lst) - 1)
    
    while index2 == index1:
        index2 = random.randint(1, len(lst) - 1)
    
    lst[index1], lst[index2] = lst[index2], lst[index1]
    
    return lst




# ## Simulated Annealing

# In[ ]:





# In[3]:


def simAn(path, param1, param2):
    
    T = param1
    alpha = param2
    global iterations
    for i in range(9999999999999):
        iterations += 1
        if i != 0:
            T = T * pow((math.e), (-i*alpha))
        if T == 0:
            return path[:]
        
        pc = pathCost(path)
        newPath = swapValues(path)
        npc = pathCost(newPath)
        
        deltaE = pc - npc
        
        if deltaE > 0:
            path = newPath[:]
        else:
            prob = math.e**((deltaE + 0.000001)/(T+ 0.000001))
            random_number = round(random.uniform(0, 1), 3)
            if prob > random_number:
                path = newPath[:]

    return path[:]



# In[4]:


# values = []
# P1 = 10000
# P2 = .05


# In[5]:


# x_values = range(len(values))

# # Plot the values
# plt.plot(x_values, values, linestyle='-', label='Values')

# # Add labels and title
# plt.xlabel('Iteration')
# plt.ylabel('Fitness Function (Path Cost)')
# plt.title(f'Simulated Annealing P1 = {P1}, P2 = {P2}')

# plt.legend()
# plt.show()


# In[ ]:





# ## Genetic Algorithm

# In[6]:


#supporting functions
def randomPaths(m, original):
    paths = []
    n = len(original)
    c1 = math.floor(n*.3)
    c2 = math.floor(n*.7)
    
    for i in range(m):
        new = []
        new.append(original[0])
        temp1 = original[1:c1]
        random.shuffle(temp1)
        new.extend(temp1)
        
        temp2 = original[c1:c2]
        random.shuffle(temp2)
        new.extend(temp2)
        
        temp3 = original[c2:]
        random.shuffle(temp3)
        new.extend(temp3)
        
        
        paths.append(new)
        
    return paths
        
def Roulette(arr):
    costlist = []
    for i in arr:
        cost = pathCost(i)
        costlist.append(cost)
    sumlist = sum(costlist)
    
    cnorm = []
    for j in costlist:
        cnorm.append(j/sumlist)
    
    
    return inverse_weights(cnorm)

def inverse_weights(probabilities):
    sorted_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    weights = [0] * len(probabilities)
    total_weights = sum(range(1, len(probabilities) + 1))
    for rank, index in enumerate(sorted_indices):
        weights[index] = (len(probabilities) - rank) / total_weights
    return weights



def reproduce(p1, p2):
    n = len(p1)
    c1 = math.floor(n*.3)
    c2 = math.floor(n*.7)
    child = []
    for i in range(n):
        if 0 <= i < c1:
            child.append(p1[i])
        if c1 <= i < c2:
            child.append(p2[i])
        if c2 <= i <= n:
            child.append(p1[i])
            
    return child



def findBest(pop):
    bestPath = []
    bestCount = 9999999999999
    for i in pop:
        if pathCost(i) < bestCount:
            bestPath = i[:]
    return bestPath



def mutate(person):
    lst = person[:]
    n = len(lst)
    c1 = math.floor(n*.3)
    c2 = math.floor(n*.7)
    index1 = random.randint(c1, c2-1)
    index2 = random.randint(c1, c2-1)

    while index2 == index1:
        index2 =random.randint(c1, c2-1)

    lst[index1], lst[index2] = lst[index2], lst[index1]
    
    return lst



def pickParents(weights, paths):
    index = list(range(0, len(paths)))
    parent1 = choice(index, p=weights)
    parent2 = choice(index, p=weights)
    
    while parent1 == parent2:
        parent2 = choice(index, p=weights)
    
    return paths[parent1], paths[parent2]


# def vals(pop):
#     maxCount = 0
#     minCount = 9999999999999
#     TotCount = 0
#     for i in pop:
#         count = pathCost(i)
#         if count < minCount:
#             minCount = pathCost(i)
#         if count > maxCount:
#             maxCount = count
#         TotCount += count
#     average = TotCount / len(pop)
    
#     return maxCount, minCount, average


# In[7]:


# main function



def GA(path, iterations, probMut):
    popNum = 100
    random_paths = randomPaths(popNum, path[:]) 
    population = random_paths[:] 
    population2 = []
    for i in range(iterations): 
#         maxx, minn, ave = vals(population)
#         valuesMin.append(minn)
#         valuesMax.append(maxx)
#         valuesAve.append(ave)
        weights = Roulette(population[:])
        for i in range(popNum):
            parent1, parent2 = pickParents(weights[:], population[:])
            child = reproduce(parent1[:], parent2[:])
            
            random_number = round(random.uniform(0, 1), 5)
            if probMut > random_number:
                child = mutate(child[:])
            population2.append(child)
        
        
        population = population2[:]
        population2 = []
        
    return findBest(population)


# In[8]:


# P1 = 10000
# P2= .05

# times = []
# costs = []
# for i in range(1):
#     valuesMin = []
#     valuesMax = []
#     valuesAve = []
#     values = []
#     start = timeit.default_timer()
#     new = GA(places[:], P1, P2)
#     stop = timeit.default_timer()
#     execution_time = stop - start
#     times.append(execution_time)
#     costs.append(pathCost(new))
    
    
        
# print(f"Min :{min(costs):.4f}")
# print(f"Average :{sum(costs):.4f}" )
# print(f"Max :{max(costs):.4f} ")
# print()
# print(f"Min : {min(times):.6f}" )
# print(f"Average : {sum(times):.6f}" )
# print(f"Max : {max(times):.6f}" )
    






# In[9]:


# x_values = range(len(valuesMin))
# plt.figure(figsize=(8, 8))
# plt.plot(x_values, valuesMin, linestyle='-', color='r', label='Min')
# plt.plot(x_values, valuesMax, linestyle='-', color='g', label='Max')
# plt.plot(x_values, valuesAve, linestyle='-', color='b', label='Average')


# plt.xlabel('Iteration')
# plt.ylabel('Fitness Function (Path Cost)')
# plt.title(f'Genetic Algorithm P1 = {P1}, P2 = {P2}')


# plt.legend()
# plt.show()


# In[11]:


if alg == "1":
    start = timeit.default_timer()
    iterations = 0
    new = simAn(places[:], P1, P2)
    stop = timeit.default_timer()
    execution_time = stop - start
    finalPathCost= pathCost(new)
    

    print(f"Hashmeh, Wesam, A20462459 Solution:")
    print(f"Initial State : {places[0]}")
    print()
    print(f"Simulated Annealing")
    print(f"Command Line Parameters: {sys.argv[1]} {sys.argv[2]} {sys.argv[3]} {sys.argv[4]}")
    print(f"Initial solution: {places}")
    print(f"Final solution: {new}")
    print(f"Number of iterations: {iterations}")
    print(f"Execution time: {execution_time} seconds")
    print(f"Complete path cost :  {finalPathCost}")
    
    filename = fn[:-4] + "_SOLUTION_SA.csv"
    new.insert(0,str(finalPathCost))
    write_to_csv(new, filename)
    
    

    
if alg == "2":
    start = timeit.default_timer()
    new = GA(places[:], P1, P2)
    stop = timeit.default_timer()
    execution_time = stop - start
    finalPathCost= pathCost(new)
    
    print(f"Hashmeh, Wesam, A20462459 Solution:")
    print(f"Initial State : {places[0]}")
    print()
    print(f"Genetic Algorithm")
    print(f"Command Line Parameters: {sys.argv[1]} {sys.argv[2]} {sys.argv[3]} {sys.argv[4]}")
    print(f"Initial solution: {places}")
    print(f"Final solution: {new}")
    print(f"Number of iterations: {P1}")
    print(f"Execution time: {execution_time} seconds")
    print(f"Complete path cost :  {finalPathCost}")
    
    filename = fn[:-4] + "_SOLUTION_GA.csv"
    new.insert(0,str(finalPathCost))
    write_to_csv(new, filename)
    


# In[7]:


test = "campus.csv"
test[:-4]+"hello"


# In[ ]:




