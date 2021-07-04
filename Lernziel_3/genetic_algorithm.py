import gym
import numpy as np
from numpy import random
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt

# set seed
# torch.manual_seed(0)
# np.random.seed(0)

# create environment
env = gym.make('CartPole-v1')

def get_NN(input_shape, output_shape, hn_count = 8):

    # untuple
    input_shape = input_shape[0]
    output_shape = output_shape.n

    network = nn.Sequential(
        nn.Linear(in_features = input_shape, out_features = hn_count),
        nn.ReLU(),
        nn.Linear(in_features = hn_count, out_features = int(hn_count / 2)),
        nn.ReLU(),
        nn.Linear(in_features = int(hn_count / 2), out_features = output_shape),
        nn.Softmax(),
    )

    # falsify grad of nn
    for param in network.parameters():

        param.requires_grad = False
    
    return network

class Individual():

    def __init__(self, observation_space, action_space, fit_func, env):

        self.network = get_NN(input_shape = observation_space.shape, output_shape = action_space)
        self.fitness = fit_func
        self.last_fitness = 0
        self.env = env
    
    def eval(self):
        "evaluate fitness of inidividuum"

        self.last_fitness = self.fitness(self.network, self.env)

        return self.last_fitness
    
    def mutate(self, sigma = 0.01):
        "mutate individuum"

        for parameter in self.network.parameters():
            parameter += sigma * torch.randn_like(parameter)

def breed(individual_a, individual_b, env, fit_func):
    "breed two individuals and get new one"

    child = Individual(
        observation_space = env.observation_space, action_space = env.action_space, 
        fit_func = fit_func, env = env)

    # get list of tuples of parent parameters [(a_par1, b_par1), (a_par2, b_par2)], child appended as last
    for parameter in zip(
        individual_a.network.parameters(), individual_b.network.parameters(), child.network.parameters()
        ):

        # get random relation from child parameters
        rel = torch.rand_like(parameter[2])

        # update childs parameters
        parameter[2].data.copy_(rel * parameter[0].data + (1 - rel) * parameter[1].data)
    
    return child

def fitness_function(network, env):
    "determine fitness of modell"

    score = 0
    done = False

    observation = env.reset()
    while not done:

        # get action --> transform from softmax vector to action tensor to int
        action = torch.squeeze(np.argmax(network(torch.Tensor(observation)))).item()

        observation_, reward, done, info = env.step(action)


        score += reward
        observation = observation_
    
    return score

def watch_game(network, env):

    score = 0
    done = False

    observation = env.reset()
    env.render()
    i = 0

    while not done:

        # get action --> transform from softmax vector to action tensor to int
        action = torch.squeeze(np.argmax(network(torch.Tensor(observation)))).item()

        observation_, reward, done, info = env.step(action)
        env.render()

        time.sleep(0.05)

        score += reward
        observation = observation_
    print("Score: ", score)
    return score

# start populations:
population_size = 100
max_gens = 100

use_elitism = True
allow_self_reproduction = True

population = [
    Individual(
        observation_space = env.observation_space,action_space = env.action_space,
        fit_func = fitness_function, env = env)
    for i in range(population_size)
    ]

gen_best_score = []
gen_avg_score = []

for generation in range(max_gens):
    

    population.sort(key = lambda indi: indi.eval(), reverse = True)

    best = population[0]
    best_fitness = best.last_fitness
    avg_fitness = np.average([ind.last_fitness for ind in population])
    print(
        f"""Generation {generation}:
        \t Best Fitness: {best_fitness}
        \t Average Fitness: {avg_fitness}
        """)

    # append statistic
    gen_best_score.append(best_fitness)
    gen_avg_score.append(avg_fitness)

    # Breed top 10 Individuums with each over
    population = population[:10]
    new_population = []

    for parent_a in population:

        for parent_b in population:

            if not allow_self_reproduction:

                if parent_a == parent_b:

                    continue
            
            # breed child
            child = breed(individual_a = parent_a, individual_b = parent_b, fit_func = fitness_function, env = env)
            
            # let child mutate
            child.mutate()

            new_population.append(child)
    
    random.shuffle(new_population)
    # cut off overcounted individuums
    new_population = new_population[:population_size]

    if use_elitism:

        population = population[0:1] + new_population
    else:
        population = new_population

# get statistics
x = [i for i in range(max_gens)]

fig = plt.figure()

plt.plot(x, gen_best_score, label = "Best Score")
plt.plot(x, gen_avg_score, label = "Average Score")

plt.title("Statistics for Evolutionary Algorithm")
plt.legend()

fig.savefig("Statistic.png")

# watch_game(best.network, env)