import random
import matplotlib.pyplot as plt

class Maximize():
    def __init__(self):
        pass
    
    def population(self, gen_len, pop_size):
        ini_pop = []
        for i in range(pop_size):
            ini_pop.append([random.randint(0, 1) for _ in range(gen_len)])
        print('Initial Population = ', ini_pop)
        return ini_pop
    
    def binary_to_decimal(self, binary_list):
        decimal_list = []
        for binary_num in binary_list:
            if isinstance(binary_num, str) and all(bit in '01' for bit in binary_num):
                decimal_list.append(int(binary_num, 2))
            elif isinstance(binary_num, list) and all(isinstance(bit, int) and bit in [0, 1] for bit in binary_num):
                binary_str = ''.join(str(bit) for bit in binary_num)
                decimal_list.append(int(binary_str, 2))
            else:
                print("Error: Element '{}' in the list is not a string or a list representation of a binary number.".format(binary_num))
        print('X Value = ', decimal_list)
        return decimal_list
    
    def fit_cal(self, decimal_list):
        fitness = []
        for i in decimal_list:
            fitness.append(i*i)
        average = sum(fitness)/len(fitness)
        print('Fitness = ', fitness)
        print('----------------------')
        print('Sum = ', sum(fitness))
        print('Average = ', average)
        print('Max = ', max(fitness))
        print('----------------------')
        return fitness
    
    def prob_cal(self, fitness):
        prob = []
        for i in fitness:
            s = sum(fitness)
            d = i / s
            prob.append(round(d * 100, 1))
        print('Probability = ', prob)
        return prob 
        
    def roulette_wheel_selection(self, probabilities, ini_pop):
        cumulative_probabilities = []
        cumulative_sum = 0
        for prob in probabilities:
            cumulative_sum += prob
            cumulative_probabilities.append(cumulative_sum)
        selected_parents = []
        for _ in range(len(probabilities)):
            rand_num = random.uniform(0, cumulative_sum)
            selected_parent_index = 0
            for i, cum_prob in enumerate(cumulative_probabilities):
                if rand_num <= cum_prob:
                    selected_parent_index = i
                    break
            selected_parents.append(selected_parent_index)
            selected_parents.sort()
            selected_parents_bin = []
            for i in selected_parents:
                selected_parents_bin.append(ini_pop[i])
        print('Selected Parents Index =', selected_parents)
        return selected_parents_bin
    
    def crossover(self, selected_parents_bin, cross_prob, num_crossover_points):
        children = []
        for i in range(0, len(selected_parents_bin)-1, 2):
            p1 = selected_parents_bin[i]
            p2 = selected_parents_bin[i+1]
            c1, c2 = p1.copy(), p2.copy()
            if random.random() < cross_prob:
                crossover_points = random.sample(range(1, len(p1)), num_crossover_points)
                crossover_points.sort()
                prev = 0
                for point in crossover_points:
                    c1[prev:point], c2[prev:point] = c2[prev:point], c1[prev:point]
                    prev = point
                c1[prev:], c2[prev:] = c2[prev:], c1[prev:]
            children.append(c1)
            children.append(c2)
        print('Population after Crossover : ', children)
        return children
    
    def mutation(self, children):
        mut_prob = 0.1
        mut_pop = []
        for individual in children:
            mut_ind = []
            for gene in individual:
                if random.random() < mut_prob:
                    muted_gene = 1 - gene
                    mut_ind.append(muted_gene)
                else:
                    muted_gene = gene
                    mut_ind.append(muted_gene)
            mut_pop.append(mut_ind)
        print('Population after mutation', mut_pop)   
        return mut_pop     
    
    def process_parents(self, selected_parents_bin):
        processed_parents = []
        for parent in selected_parents_bin:
            decimal_val = int(''.join(map(str, parent)), 2)
            if decimal_val == 31:
                processed_parents.append(parent)
                break  
        print('test', selected_parents_bin)
        return processed_parents
    
    def geneticAlgorithm(self, gen_len, pop_size, generations, cross_prob, num_crossover_points):
        total_fitness = []     
        initial_population = self.population(gen_len, pop_size)
        for _ in range(generations):
            print('This is Initial population: ', initial_population)
            decimal_num = self.binary_to_decimal(initial_population)
            fitness = self.fit_cal(decimal_num)
            probabilty = self.prob_cal(fitness)
            roulette_selection = self.roulette_wheel_selection(probabilty, initial_population)
            processed_parents = self.process_parents(roulette_selection)
            if processed_parents:
                print("Parent with value 31 found.")
                new_offspring = [random.choice(roulette_selection) for _ in range(len(roulette_selection) - 1)]
                initial_population = processed_parents + new_offspring
            else:
                crossover = self.crossover(roulette_selection, cross_prob, num_crossover_points)
                mutation = self.mutation(crossover)
                initial_population = mutation
            decimal_num_cross = self.binary_to_decimal(initial_population)
            fitness_cross = self.fit_cal(decimal_num_cross)
            total_fitness.append(max(fitness_cross))
        return total_fitness

if __name__ == "__main__":
    demo = Maximize()
    generations = 50
    fitness_over_generations = demo.geneticAlgorithm(5, 4, generations, 0.95, 2)
    
    generations = range(1, generations + 1)
    print('The Maximum Fitness found is:', max(fitness_over_generations))
    print('The Starting Fitness found is:', (fitness_over_generations[0]))
    print('The Minimum Fitness found is:', min(fitness_over_generations))
    
    # plt.plot(generations, fitness_over_generations)
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness Value')
    # plt.title('Best Fitness Value Over Generations')
    # plt.grid(True)
    # plt.show()

    plt.plot(generations, fitness_over_generations, marker='o', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title('Best Fitness Value Over Generations')
    plt.grid(True)
    plt.legend(['Best Fitness'], loc='upper left')
    plt.show()
