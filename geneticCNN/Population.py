import copy
import torch
import random

from Individual import Individual
from train_Individual import train_Individual


class Population:
    """
    This class config for the Population use genetic algorithm

    # Parameters for init
    1. `*train_loader` : torch.utils.data.DataLoader
        - Contain data for train Individual
    2. `*test_loader` : torch.utils.data.DataLoader
        - Contain data for test Individual
    3. `*optimizer` : torch.optim.<Algorithms>
        - optimizer for train Individual
    4. `*optimizer_setting` : dict
        - setting for optimizer
    5. `*loss_func` : torch.nn.functional.<Loss Functions>
        - loss_func for train/test Individual
    6. `*scheduler` : torch.optim.lr_scheduler.<scheduler>
        - scheduler for train/test Individual
    7. `scheduler_setting` : dict
        - setting for scheduler
    8. `epochs` : int (default 5)
        - train epochs
    9. `population_size` : int (default 16)
        - size of population (mean the number of Individual)
    10. `num_stages` : tuple (default (3,5))
        - list of number of nodes in stages
    11. `gens` : list of tuple (default None)
        - list gens for Individuals
    12. `crossover_probability` : float (default 0.2)
        - probability Individual crossover
    13. `mutation_probability` : float (default 0.8)
        - probability Individual mutation
    14. `crossover_rate` : float (default 0.5)
        - rate stages in Individual crossover
    15. `mutation_rate` : float (default 0.015)
        - rate stages in Individual mutation

    # Returns
    - `Population`
        - class use GA for NAS

    # Properties
    - `Population.run` : def (generation)
        - run the GA with generation pass into (default 3)
    - `Population.train_loader` : torch.utils.data.DataLoader
        - Contain data for train Individual
    - `Population.test_loader` : torch.utils.data.DataLoader
        - Contain data for test Individual
    - `Population.optimizer` : torch.optim.<Algorithms>
        - optimizer for train Individual
    - `Population.optimizer_setting` : dict
        - setting for optimizer
    - `Population.loss_func` : torch.nn.functional.<Loss Functions>
        - loss_func for train/test Individual
    - `Population.scheduler` : torch.optim.lr_scheduler.<scheduler>
        - scheduler for train/test Individual
    - `Population.scheduler_setting` : dict
        - setting for scheduler
    - `Population.epochs` : int
        - train epochs
    - `Population.population_size` : int
        - size of population (mean the number of Individual)
    - `Population.num_stages` : tuple
        - list of number of nodes in stages
    - `Population.gens` : list of tuple
        - list gens for Individuals
    - `Population.crossover_probability` : float
        - probability Individual crossover
    - `Population.mutation_probability` : float
        - probability Individual mutation
    - `Population.crossover_rate` : float
        - rate stages in Individual crossover
    - `Population.mutation_rate` : float
        - rate stages in Individual mutation

    Raises
    ------
    - ValueError
    """

    def __init__(self,
                 train_loader, test_loader, optimizer, optimizer_setting, loss_func, scheduler, scheduler_setting, epochs=5,
                 population_size=16, num_stages=(3, 5), gens=None, crossover_probability=0.2, mutation_probability=0.8, crossover_rate=0.5, mutation_rate=0.015,
                 ):
        super(Population, self).__init__()

        print('init Population with:')
        self.population_size = population_size
        print(f'\t - population_size = {population_size}')
        self.individuals = [Individual(num_stages) if gens is None
                            else Individual(num_stages, gens[i])
                            for i in range(population_size)]
        print(f'\t - num_stages = {num_stages}')
        print(f'\t - gens = {[x.Stages.gen for x in self.individuals]}')
        print(f'\t - list of individuals = {[x.Stages.gen_model for x in self.individuals]}')
        self.crossover_probability = crossover_probability
        print(f'\t - crossover_probability = {crossover_probability}')
        self.mutation_probability = mutation_probability
        print(f'\t - mutation_probability = {mutation_probability}')
        self.crossover_rate = crossover_rate
        print(f'\t - crossover_rate = {crossover_rate}')
        self.mutation_rate = mutation_rate
        print(f'\t - mutation_rate = {mutation_rate}')
        print('---------------------------------------------\n')
        self.maximize = ('', 0, 1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.optimizer_setting = optimizer_setting
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.scheduler_setting = scheduler_setting
        self.epochs = epochs

    def fitness_function(self):
        print('\tfitness_function\n')
        for i in range(self.population_size):
            if self.individuals[i].accuracy is None and self.individuals[i].loss is None:
                print(f'\t\tTraining {self.individuals[i].Stages.gen_model}')
                optimizer = self.optimizer(self.individuals[i].parameters(), **self.optimizer_setting)
                scheduler = self.scheduler(optimizer, **self.scheduler_setting)
                self.individuals[i].accuracy, self.individuals[i].loss = train_Individual(
                    self.individuals[i], self.device,
                    self.train_loader, self.test_loader,
                    optimizer, self.loss_func, scheduler, self.epochs)
                print(f'\t\tFitness = {self.individuals[i].accuracy:0.8f}')
                print('\t\t---------------------------------------------\n')
                if self.individuals[i].accuracy > self.maximize[1]:
                    self.maximize = (self.individuals[i].Stages.gen_model, self.individuals[i].accuracy, self.individuals[i].loss)
        print('\t\tMaximize of Population: {} - Fitness = {:0.8f}\n'.format(*self.maximize))
        print('\t=============================================\n')

    def selection_function(self, n=4):
        # tournament_select
        print('\tselection_function\n')
        weights = [x.accuracy for x in self.individuals]
        index_max = []
        index_tem = list(range(self.population_size))
        while len(index_max) < self.population_size:
            random.shuffle(index_tem)
            matrix_select = [index_tem[i:i+n] for i in range(0, len(index_tem), n)]
            for x in matrix_select:
                index_max += [x[max(range(len(x)), key=lambda i: weights[x[i]])]]

        self.individuals = [copy.deepcopy(self.individuals[index_max[i]]) for i in range(self.population_size)]
        print(f'\t\tgens after select = {[x.Stages.gen for x in self.individuals]}')
        print(f'\t\tlist of individuals after select = {[x.Stages.gen_model for x in self.individuals]}\n')
        print('\t=============================================\n')

    def crossover_function(self):
        print('\tcrossover_function\n')
        num_stages = self.individuals[0].Stages.num_stages
        for i in range(self.population_size // 2):
            if random.random() < self.crossover_probability:
                temp_gen_1 = []
                temp_gen_2 = []
                for j in range(len(num_stages)):
                    if random.random() < self.crossover_rate:
                        temp_gen_1.append(self.individuals[2*i + 1].Stages.gen[j])
                        temp_gen_2.append(self.individuals[2*i].Stages.gen[j])
                    else:
                        temp_gen_1.append(self.individuals[2*i].Stages.gen[j])
                        temp_gen_2.append(self.individuals[2*i + 1].Stages.gen[j])
                if tuple(temp_gen_1) != self.individuals[2*i].Stages.gen and tuple(temp_gen_1) != self.individuals[2*i + 1].Stages.gen:
                    print(f'\t\tCrossover 2 Individual:')
                    print(f'\t\t\t - {self.individuals[2*i].Stages.gen_model}')
                    print(f'\t\t\t - {self.individuals[2*i + 1].Stages.gen_model}')
                    self.individuals[2*i] = Individual(num_stages, tuple(temp_gen_1))
                    self.individuals[2*i + 1] = Individual(num_stages, tuple(temp_gen_2))
                    print(f'\t\tTo 2 new Individual:')
                    print(f'\t\t\t - {self.individuals[2*i].Stages.gen_model}')
                    print(f'\t\t\t - {self.individuals[2*i + 1].Stages.gen_model}')
                    print('\t\t---------------------------------------------\n')
        print('\t=============================================\n')

    def mutation_function(self):
        print('\tmutation_function\n')
        num_stages = self.individuals[0].Stages.num_stages
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                new_gen = []
                for j in range(len(num_stages)):
                    if random.random() < self.mutation_rate:
                        new_gen.append(''.join([random.choice(['0', '1']) for _ in range(int(num_stages[j] * (num_stages[j] - 1) / 2))]))
                    else:
                        new_gen.append(self.individuals[i].Stages.gen[j])
                if tuple(new_gen) != self.individuals[i].Stages.gen:
                    print(f'\t\tMutation Individual: {self.individuals[i].Stages.gen_model}')
                    self.individuals[i] = Individual(num_stages, tuple(new_gen))
                    print(f'\t\tTo new Individual: {self.individuals[i].Stages.gen_model}')
                    print('\t\t---------------------------------------------\n')
        print('\t=============================================\n')

    def run(self, generation=3):
        for i in range(generation):
            print(f'Start generation # {i}...\n')
            self.fitness_function()
            self.selection_function()
            self.crossover_function()
            self.mutation_function()
            print(f'End generation # {i}...\n')
            print('---------------------------------------------\n')
        print(f'Final evaluate Population\n')
        self.fitness_function()
