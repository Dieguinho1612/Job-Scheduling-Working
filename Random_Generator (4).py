#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pickle
import os
#import import_ipynb
from Jobs_and_Machines import *
from Global_Variables import *


# In[2]:


"""#we will define a set of maximal values to create a list oj jobs and machines later on
def generate_max_values_sets(n, m, max_init_runtime, max_runtime, max_deadline, max_weight):
    
    #we check which max values sets already exist, so it creates a new one with an index increased by 1
    ind = 1
    ind_str = "01"
    while os.path.exists(f'MaxValuesSets/MaxValues_{ind_str}.txt'):
        ind += 1
        ind_str = "0"*(2-len(str(ind))) + str(ind)
        
    #create directory to later save random lists of jobs and machines generated by this set of parameters
    os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}')
    os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}/Jobs_and_Machines_{ind_str}')
    os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}/States_{ind_str}')
    os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}/MLP_Data_{ind_str}')
    
    #save all parameters
    text_to_save = f"n = {n}\
    \nm = {m}\
    \nmax_init_runtime = {max_init_runtime}\
    \nmax_runtime = {max_runtime}\
    \nmax_deadline = {max_deadline}\
    \nmax_weight = {max_weight}"
    
    #create .txt-file for parameters
    with open(f'MaxValuesSets/MaxValues_{ind_str}/MaxValues_{ind_str}.txt', 'w') as f:
        f.write(text_to_save)"""


# In[3]:


def read_max_parameters(ind, prnt=False):
    
    parameters = []
    ind_str = "0"*(2-len(str(ind))) + str(ind)
    with open(f'MaxValuesSets/MaxValues_{ind_str}/MaxValues_{ind_str}.txt', 'r') as f:
        for line in f:
            parameters.append(int(line.split(" = ")[1]))
            if prnt:
                print(line.replace("\n",""))
        
    return(parameters)


# In[ ]:


#we will define a set of maximal values to create a list oj jobs and machines later on
def generate_max_values_sets(ind):
    
    ind_str = "0"*(2-len(str(ind))) + str(ind)
    
    #check if required directories exist already
    if not os.path.exists('MaxValuesSets'):
        os.mkdir('MaxValuesSets') #head directory to save all data related files
    if not os.path.exists(f'MaxValuesSets/MaxValues_{ind_str}'):
        os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}') #one directory for every max values set environment
    if not os.path.exists(f'MaxValuesSets/MaxValues_{ind_str}/Jobs_and_Machines_{ind_str}'): 
        os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}/Jobs_and_Machines_{ind_str}') #directory for list of jobs and machiens
    if not os.path.exists(f'MaxValuesSets/MaxValues_{ind_str}/States_{ind_str}'):
        os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}/States_{ind_str}') #directory for all states
    if not os.path.exists(f'MaxValuesSets/MaxValues_{ind_str}/Data_{ind_str}'):
        os.mkdir(f'MaxValuesSets/MaxValues_{ind_str}/Data_{ind_str}') #directory for NN data created out of states
    
    #we check if max values set already exists
    if os.path.exists(f'MaxValuesSets/MaxValues_{ind_str}/MaxValues_{ind_str}.txt'):
        #check if they match, else raise an error
        if [n,m,max_init_runtime,max_runtime,max_deadline,max_weight] != read_max_parameters(ind):
            raise ValueError(f'Max Values Set {ind_str} already exists but does not match the given parameters!')
    else:
        #save all parameters
        text_to_save = f"n = {n}\
        \nm = {m}\
        \nmax_init_runtime = {max_init_runtime}\
        \nmax_runtime = {max_runtime}\
        \nmax_deadline = {max_deadline}\
        \nmax_weight = {max_weight}"
        #create .txt-file for parameters
        with open(f'MaxValuesSets/MaxValues_{ind_str}/MaxValues_{ind_str}.txt', 'w') as f:
            f.write(text_to_save)


# In[4]:


def generate_random_init_runtimes(m, max_init_runtime):
    
    #generate random initial runtimes up to definied maximal vlaue, then set one to zero to create the starting point for the JS
    rand_init_runtimes = [random.randint(0,max_init_runtime) for i in range(m)]
    if 0 not in rand_init_runtimes:
        rand_init_runtimes[random.randrange(m)] = 0
        
    return rand_init_runtimes


# In[5]:


def generate_random_machines(m, max_init_runtime, max_deadline, max_weight, prnt=False):
    
    rand_init_runtimes = generate_random_init_runtimes(m, max_init_runtime)
    
    #generate machines with random deadlines, weights and initial runtimes up to the defined maximal value
    random_list_machines = [machines(rand_init_runtimes[i],
                                     random.randint(0,max_deadline), 
                                     random.randint(1,max_weight))
                            for i in range(m)]
    #appropietly prepare machines
    if prnt:
        print("")
    random_list_machines = prep_machines(random_list_machines, prnt=prnt)
    
    return random_list_machines


# In[6]:


def generate_random_jobs(n, m, max_runtime, max_deadline, max_weight, prnt=False):
    
    #generate jobs with random processing times, deadlines and weights up to the defined maximal value (processing time > 0)
    random_list_jobs = [jobs([random.randint(1,max_runtime) for i in range(m)], 
                             random.randint(0,max_deadline), 
                             random.randint(1,max_weight)) 
                        for j in range(n)]
    
    #appropietly prepare jobs
    if prnt:
        print("")
    random_list_jobs = prep_jobs(random_list_jobs, prnt=prnt)
    
    return random_list_jobs


# In[7]:


def generate_random_JS(MVS, JS, prnt=False):
    
    #read in parameters from MaxValuesSet
    #[n,m,max_init_runtime,max_runtime,max_deadline,max_weight] = read_max_parameters(MVS, prnt=prnt)
    
    #generate list of random machines with random initial runtimes
    random_list_machines = generate_random_machines(m, max_init_runtime, max_deadline, max_weight, prnt=prnt)
    #generate list of random jobs
    random_list_jobs = generate_random_jobs (n, m, max_runtime, max_deadline, max_weight, prnt=prnt)
    
    """
    #save the generated lists
    MVS_str = "0"*(2-len(str(MVS))) + str(MVS)
    JS_str = "0"*(4-len(str(JS))) + str(JS)
    path = f'MaxValuesSets/MaxValues_{MVS_str}/Jobs_and_Machines_{MVS_str}/random_lists_{MVS_str}_{JS_str}.pickle'
    with open(path, 'wb') as f:
        pickle.dump((random_list_jobs, 
                     random_list_machines), 
                    f, pickle.HIGHEST_PROTOCOL)"""
    
    return random_list_jobs, random_list_machines


# In[ ]:


def generate_random_environment():
    
    global max_runtime, max_init_runtime 
    max_runtime = random.randint(round(max_deadline/3),round(max_deadline*1.5))
    max_init_runtime = random.randint(round(max_runtime/3), max_runtime)
    random_list_jobs, random_list_machines = generate_random_JS("","")
    
    return [max_runtime, max_init_runtime, random_list_jobs, random_list_machines]


# In[8]:


#generate_max_values_sets(8,3,10,20,30,10)


# In[ ]:


#read_max_parameters(1)


# In[ ]:


#list_jobs, list_machines = generate_random_JS(1, 1)


# In[9]:


"""for i in range(80):
    generate_random_JS(1,i+1)
""";


# In[ ]:


"""import shutil

shutil.rmtree('MaxValuesSets/MaxValues_01')""";

