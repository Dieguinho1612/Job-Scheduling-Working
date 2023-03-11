#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow
from tensorflow import keras
#import import_ipynb
from Jobs_and_Machines import *
import States_and_Policies
from States_and_Policies import *
import Data_for_NN
from Data_for_NN import *
from TransformerLayers import *


# In[3]:


"""#change working directory
os.chdir('D:\\Job-Scheduling-Files')
os.getcwd()""";


# In[2]:


n = 8
m = 4
max_weight = 10
max_deadline = 30

import Global_Variables

#create dictionary of all constant variables of this JS
list_var = ['n', 'm', 'max_deadline', 'max_weight']
dict_var = dict((var,eval(var)) for var in list_var)
#pass them as global variables, so that imported notebooks can access them
Global_Variables.set_var_to_global(dict_var)

import Random_Generator
from Random_Generator import *


# In[26]:


def load_NN(NN_name):
    #NN = keras.models.load_model(f'D:\\Job-Scheduling-Files\{NN_name}.h5', custom_objects={'FeedForward': FeedForward, 'Pointer': Pointer, 'MSE_with_Softmax': MSE_with_Softmax, 'costs':costs})
    NN = keras.models.load_model(f'{NN_name}.h5', custom_objects={'FeedForward': FeedForward, 'Pointer': Pointer, 'MSE_with_Softmax': MSE_with_Softmax, 'costs':costs})
    NN.run_eagerly = True
    return NN


# In[4]:


def create_JS_environment():
    
    global max_runtime, max_init_runtime, list_jobs, list_machines
    environment = generate_random_environment()
    max_runtime, max_init_runtime, list_jobs, list_machines = environment
    #create dictionary of all environmental variables of this JS
    list_env = ['max_runtime', 'max_init_runtime', 'list_jobs', 'list_machines']
    dict_env = dict((list_env[i],environment[i]) for i in range(len(environment)))
    #pass them as global variables, so that imported notebooks can access them
    Global_Variables.set_var_to_global(dict_env)
    change_global_var_of_module(States_and_Policies)
    change_global_var_of_module(Data_for_NN)


# In[5]:


def act(NN, state):
    
    n_state = sum(state.jobs_remaining) #get job number of state
    m_state = sum(state.machines_on_duty) #get machine number of state
    
    remaining_jobs = [job for job in list_jobs if state.jobs_remaining[job.index] == 1]
    remaining_jobs.append(None) #stands for machine_turnoff
    data = [np.expand_dims(state.input[0],axis=0), np.expand_dims(state.input[1],axis=0)]
    act_values = NN.predict(data, verbose=0) #hier irgendwie Daten einlesen
    if m_state > 1:
        action = np.argmax(act_values[0]) #index of action with minimal expected cost (costs here are inversed)
    else:
        action = np.argmax(act_values[0][:-1])
    if action < n_state:
        action = state.permutation[0][action]
    job = remaining_jobs[action]
    return action, job


# In[6]:


def create_policy(NN,state):
    
    state_list = []
    action_list = []
    done = False
    while not done:
        
        if not state.input:
            seq_data(state) #create data input in sequential form for NN. Saved under state.input
        
        action, job = act(NN, state)
        machine = list_machines[state.machine]
        
        if job:
            next_state = assign_job(state, job, machine)
        else:
            next_state = turn_off_machine(state, machine)
        
        
        if sum(next_state.jobs_remaining) == 0:
            done = True
            """#add remaining costs until everything is finished, if that is not the case yet
            if next_state.machine_runtimes:
                next_state.costs += max(next_state.machine_runtimes)"""
            
        action_list.append(action)
        state_list.append(state)
        
        state = next_state
        
    
    state_list.append(state)
        
    return action_list, state_list


# In[7]:


def compute_remaining_states(current_state):
        
    #initiate list of states = []
    list_states = []
    
    #list of current states
    current_states = [current_state]
    
    #go through every current state, save their successors, add current states to list of all states
        #then define successor states as current states, clear list of successor states and repeat until done
    while current_states:
        
        #empty list of all successor states
        successor_states = []
        
        #create and add all successor states of current states
        for state in current_states:
            
            #list of all successors of this state
            state_successors = []
            
            #create one state for every remaining job assigned to the free machine with lowest index
            #else:
            if sum(state.jobs_remaining) > 0:
                
                machine = list_machines[state.machine]
                
                #create data for Neural Network
                #state_to_data(state)
                    
                #loop through jobs
                list_jobs_remaining = [job for job in list_jobs if state.jobs_remaining[job.index] == 1]
                for index, job in enumerate(list_jobs):
                    #check if the job still has to be done
                    if state.jobs_remaining[index] == 1:
                        #assign job to machine
                        state_successors.append(assign_job(state,job,machine))    
                
                #check if turning it off is an option
                if sum(state.machines_on_duty) > 1:                       
                    #add successor state created by shutting down machine
                    state_successors.append(turn_off_machine(state, machine))
                        
                
            #add successor list to the attributes of state
            state.successors = state_successors
            
            #add successors of this state to the list of all successors of all current states
            successor_states += state_successors
        
        #add current states to list of all states
        list_states += current_states
        
        #the successor states then become the current states
        current_states = successor_states
        
    return list_states


# In[8]:


def compute_remaining_policy(current_state):
    
    #remaining_states = compute_remaining_states(current_state)
    #backtracking(remaining_states)
    remaining_states = create_all_states(from_state=current_state)
    backtracking(remaining_states)
    remaining_policy = optimal_policy(remaining_states)
    return (remaining_policy)


# In[9]:


def induce_and_compute_policy(NN,state):
    
    state_list = []
    action_list = []

    
    n_state = sum(state.jobs_remaining) #get job number of state
    m_state = sum(state.machines_on_duty) #get machine number of state
    
    while n_state > 2 and m_state > 1:
        
        if not state.input:
            seq_data(state) #create data input in sequential form for NN. Saved under state.input
        
        action, job = act(NN, state)
        machine = list_machines[state.machine]
        
        if job:
            next_state = assign_job(state, job, machine)
        else:
            next_state = turn_off_machine(state, machine)
        
        
        if sum(next_state.jobs_remaining) == 0:
            done = True
            """#add remaining costs until everything is finished, if that is not the case yet
            if next_state.machine_runtimes:
                next_state.costs += max(next_state.machine_runtimes)"""
            
        action_list.append(action)
        state_list.append(state)
        
        state = next_state
        n_state = sum(state.jobs_remaining) #get job number of state
        m_state = sum(state.machines_on_duty) #get machine number of state
        
    
    remaining_policy = compute_remaining_policy(state)
    action_list += remaining_policy[0]
    state_list += remaining_policy[1]
        
    return action_list, state_list


# In[10]:


def policy_costs(policy):
    pol_costs = sum(state.costs for state in policy[1])
    return pol_costs


# In[11]:


def comparative_metric(initial_state):
    state_list = []
    action_list = []
    state = initial_state
    done = False
    while not done:
        
        if not state.input:
            seq_data(state) #create data input in sequential form for NN. Saved under state.input
        
        remaining_jobs = [job for job in list_jobs if state.jobs_remaining[job.index] == 1]
        machine = list_machines[state.machine]
        job_assigned = False
        
        for action in state.permutation[0]:
            job = remaining_jobs[action]
            proc_time = job.processing_time[state.machine]
            alt_times = [state.machine_runtimes[i] + job.processing_time[i] for i in range(m) if state.machines_on_duty[i]==1]
            if not min(alt_times) < proc_time:
                next_state = assign_job(state, job, machine)
                job_assigned = True
                if sum(next_state.jobs_remaining) == 0:
                    done = True
                break
                
        if job_assigned == False:
            next_state = turn_off_machine(state, machine)
            action = len(remaining_jobs)
        
        action_list.append(action)
        state_list.append(state)
        
        state = next_state
    
    state_list.append(state)
        
    return action_list, state_list


# In[12]:


def comparative_metric_opt_end(initial_state):
    state_list = []
    action_list = []
    state = initial_state
    
    
    n_state = sum(state.jobs_remaining) #get job number of state
    m_state = sum(state.machines_on_duty) #get machine number of state
    
    while n_state > 2 and m_state > 1:
        
        if not state.input:
            seq_data(state) #create data input in sequential form for NN. Saved under state.input
        
        remaining_jobs = [job for job in list_jobs if state.jobs_remaining[job.index] == 1]
        machine = list_machines[state.machine]
        job_assigned = False
        
        for action in state.permutation[0]:
            job = remaining_jobs[action]
            proc_time = job.processing_time[state.machine]
            alt_times = [state.machine_runtimes[i] + job.processing_time[i] for i in range(m) if state.machines_on_duty[i]==1]
            if not min(alt_times) < proc_time:
                next_state = assign_job(state, job, machine)
                job_assigned = True
                if sum(next_state.jobs_remaining) == 0:
                    done = True
                break
                
        if job_assigned == False:
            next_state = turn_off_machine(state, machine)
            action = len(remaining_jobs)
        
        action_list.append(action)
        state_list.append(state)
        
        state = next_state
        n_state = sum(state.jobs_remaining) #get job number of state
        m_state = sum(state.machines_on_duty) #get machine number of state
        
    
    remaining_policy = compute_remaining_policy(state)
    action_list += remaining_policy[0]
    state_list += remaining_policy[1]
    
    
        
    return action_list, state_list


# ### Mehrere Schedules vergleichen

# In[13]:


def compare_schedules(NN, num_schedules):
    
    avg_cost_ratios = np.array([0]*4,dtype=np.float64)
    
    for _ in range(num_schedules):
        #create random environment
        create_JS_environment()
        #create all states
        all_states = create_all_states()
        backtracking(all_states)
        initial_state = all_states[0]

        #compute optimal policy and costs
        opt_policy = optimal_policy(all_states)
        opt_costs = policy_costs(opt_policy)

        #policy and costs for NN policy with end being optimally computed
        NN_policy_opt = induce_and_compute_policy(NN, initial_state)
        NN_opt_costs = policy_costs(NN_policy_opt)

        #policy and costs for NN policy
        NN_policy = create_policy(NN, initial_state)
        NN_policy_costs = policy_costs(NN_policy)

        #policy and costs for comparative metric algorithm with end being optimally computed
        comp_metric_policy_opt = comparative_metric_opt_end(initial_state)
        comp_metric_opt_costs = policy_costs(comp_metric_policy_opt)

        #policy and costs for comparative metric algorithm
        comp_metric_policy = comparative_metric(initial_state)
        comp_metric_policy_costs = policy_costs(comp_metric_policy)

        scheduling_costs = np.array([NN_opt_costs, NN_policy_costs, comp_metric_opt_costs, comp_metric_policy_costs])
        avg_cost_ratios += scheduling_costs / opt_costs 
    
    avg_cost_ratios /= num_schedules
    
    for avg_cost_ratio in avg_cost_ratios:
        print(avg_cost_ratio)


# In[14]:


### NN = load_NN("Final_Pointer3")


# In[15]:


### compare_schedules(NN,3)


# ### Estimate New Data

# In[16]:


def increase_n(new_n):
    global n
    n = new_n
    Global_Variables.n = n
    Random_Generator.n = n
    States_and_Policies.n = n
    Data_for_NN.n = n


# In[33]:


def estim_assignment_costs(Target_NN, state):
    for action, job in enumerate(list_jobs):
        machine = list_machines[state.machine]
        successor_state = assign_job(state,job,machine)
        trans_costs = successor_state.costs
        #policy = create_policy(Target_NN, successor_state)
        policy = induce_and_compute_policy(Target_NN, successor_state)
        future_costs = policy_costs(policy)
        state.Qvalues[action] = trans_costs + future_costs


# In[18]:


def estim_Qvalues(Target_NN, state):
    
    estim_assignment_costs(Target_NN, state)
    m_state = sum(state.machines_on_duty)
    if m_state > 1:
        machine = list_machines[state.machine]
        turn_off_state = turn_off_machine(state, machine)
        estim_Qvalues(Target_NN, turn_off_state)
        trans_costs = turn_off_state.costs
        state.Qvalues[-1] = trans_costs + min(turn_off_state.Qvalues)
    else:
        state.Qvalues = state.Qvalues[:-1]


# In[19]:


def estim_data(data_dictionary, Target_NN, init_state):
    state = init_state
    m_state = m
    estim_Qvalues(Target_NN, state)
    states = []
    while m_state>1:
        state_input(state)
        data_dictionary[(n,m_state)][0].append(state.input)
        state_target(state)
        data_dictionary[(n,m_state)][1].append(state.target[2])
        state = state.transition_dic[(n,state.machine)]
        m_state -= 1


# In[20]:


def estim_data_into_LSTM_format(data_dictionary):
    for key in data_dictionary:
        inputs, targets = data_dictionary[key] #tuple of list of inputs and list of targets
        #inputs is a list, for every state their is one entry, being a list itself 
        #These inner lists consist of two entries: Job-data and Machine-data of a state
        #every machine-data consists of 3 entries, so create indexes for the range of m_state repeating every index 3 times
        idxs = [ind+1 for ind in range(key[1]) for _ in range(3)]
        
        seq_inputs = [np.insert(inp[0],idxs,inp[1].flatten(), axis=1) for inp in inputs]
        #now we transform each of these two lists of arrays into an array of arrays.
        #Both arrays will be listed as inputs
        #the final format will be a tuple ([x_rt, x_dw],[t]) with x being the array of input arrays and t the array of target arrays
        
        ##data_dict[key][0] = [np.stack(rt_inputs), np.stack(dw_inputs)]
        data_dictionary[key][0] = [np.stack(seq_inputs)]
        data_dictionary[key][1] = [np.stack(targets)]


# In[21]:


def create_estim_data(Target_NN, num_data):
    data_dictionary = dict(((n,m_state),[[],[]]) 
                       for m_state in range(2,m+1))
    for _ in range(num_data):
        create_JS_environment()
        init_state = create_initial_state()
        estim_data(data_dictionary, Target_NN, init_state)
        
    #estim_data_into_LSTM_format(data_dictionary)
        
    return data_dictionary


# In[36]:


import sys
input = sys.argv[1]
input = int(input)-1

def save_dictionary(data_dictionary):
    #with open(f'D:\\Job-Scheduling-Files\Data\EstimData\estim_data_{input}.pickle', 'wb') as f:
    with open(f'EstimData\estim_data_{input}.pickle', 'wb') as f:
        pickle.dump(data_dictionary, f, pickle.HIGHEST_PROTOCOL)


# In[27]:


Target_NN = load_NN("Final_Pointer3")


# In[28]:


increase_n(9)


# In[34]:


data_dictionary = create_estim_data(Target_NN, 125)


# In[37]:


save_dictionary(data_dictionary)


# In[ ]:


#


# In[60]:


#def save_estim_data(data_dictionary):
#    for key in data_dictionary:
#        #save every merged n-m-combination as a pickle file
#        estim_data_path = 'D:\\Job-Scheduling-Files\Data\EstimData'
#        with open(f'{estim_data_path}\{key[0]}-jobs-{key[1]}-machines.pickle', 'wb') as f:
#            pickle.dump(data_dictionary[key], f, pickle.HIGHEST_PROTOCOL)


# In[48]:


"""def safe_estim_data(data_dict_list):
    for DS, data_dictionary in enumerate(data_dict_list):
        DS_str = "0"*(2-len(str(DS))) + str(DS)
        path = f'D:\\Job-Scheduling-Files\Data\EstimDataSet\estim_data_{DS_str}.pickle'
        with open(path, 'wb') as f:
            pickle.dump(data_dictionary, f, pickle.HIGHEST_PROTOCOL)""";


# In[61]:


#Target_NN = load_NN("Final_Pointer3")


# In[70]:


#increase_n(8)


# In[63]:


#import time


# In[1]:


#st = time.time()
#data_dictionary = create_estim_data(Target_NN, 1000)
#et = time.time()
#et - st


# In[ ]:


#save_estim_data(data_dictionary)

