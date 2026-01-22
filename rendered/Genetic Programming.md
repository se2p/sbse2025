# Genetic Programming

This chapter introduces some very basic concepts about Genetic Programming. As usual, we will require a couple of imports:


```python
import random
import sys
import re
import matplotlib.pyplot as plt
from statistics import mean

# For presenting as slides
#plt.rcParams['figure.figsize'] = [12, 8]
#plt.rcParams.update({'font.size': 22})
#plt.rcParams['lines.linewidth'] = 3
```

## Classic Genetic Programming

The main difference between regular Genetic Algorithms and Genetic Programming lies in the representation: GAs tend to operate on the genotype, while GPs operate on the phenotype (programs); for this, the programs are traditionally represented as trees. Furthermore, GPs have a more open-ended nature in that the number of elements used in a solution as well as their interconnections must be open to evolution.

In a tree representation, programs are typically generated using two sets: A set of terminals (items of arity 0) and functions (items of arity > 0). The input to a function must be the result of any other function that can be defined, which leads to the important property of _type closure_: Each function must be able to handle all values it might ever receive as input, all terminals must be allowable inputs for all functions, and the output from any function must be a permitted input to any other function. 

The first step to evolving a suitable program for a given problem is thus to define the sets of functions and terminals:


```python
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return 1 if not y else x/y
```


```python
FUNCTIONS = [add, sub, mul, div]
TERMINALS = ['A', 'B', 'C', 'D']
```

### Initialisation

Our usual approach to define the presentation is by introducing code that creates random instances. Let's do the same here, except that this time we need to use a tree representation. For this, we define a suitable datastructure:


```python
class Node:
    
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        if self.data in FUNCTIONS:
            return self.data.__name__
        else:
            return str(self.data)
        
    def to_dot(self, dot):
        dot.node(str(id(self)), str(self))
        
        if self.left:
            left_dot = self.left.to_dot(dot)
            dot.edge(str(id(self)), str(id(self.left)))

        if self.right:
            right_dot = self.right.to_dot(dot)
            dot.edge(str(id(self)), str(id(self.right)))

        return dot      
```

The `to_dot` function is a helper function we will use in order to represent trees visually using GraphViz (which will need to be installed on your machine if you want to run this Jupyter notebook).


```python
from graphviz import Digraph
```


```python
def print_tree(tree):
    dot = Digraph()
    tree.to_dot(dot)
    return dot
```

#### Full Initialisation

A basic approach to creating random trees is to grow full trees. We decide on a maximum depth for our tree, and then add non-terminal nodes until the depth limit is reached. Once we have reached it, we only add terminals as leaves. 


```python
MAX_DEPTH = 3
```


```python
def get_random_terminal():
    return random.choice(TERMINALS)
```


```python
def get_random_function():
    return random.choice(FUNCTIONS)
```


```python
def fill_random_tree(depth=0):
    tree = Node()
    if depth < MAX_DEPTH:
        tree.data = get_random_function()
        tree.left  = fill_random_tree(depth + 1)
        tree.right = fill_random_tree(depth + 1)
    else:
        tree.data = get_random_terminal()
    return tree
```


```python
root = fill_random_tree()
print_tree(root)
```




    
![svg](9/output_21_0.svg)
    



#### Grow Initialisation


An alternative approach to filling entire trees up to their maximum depth is to grow trees of various sizes. For this we add random nodes while we haven't reached the depth limit. The resulting trees will be less bushy, but the ratio between terminals and non-terminals will bias the average size. 


```python
def grow_random_tree(depth=0):
    tree = Node()
    if depth >= MAX_DEPTH or random.random() > 0.5:
        tree.data = get_random_terminal()
    else:
        tree.data = get_random_function()
        tree.left  = grow_random_tree(depth + 1)
        tree.right = grow_random_tree(depth + 1)
    return tree
```


```python
root = grow_random_tree()
print_tree(root)
```




    
![svg](9/output_25_0.svg)
    



If the ratio of terminals to functions is high, then we might end up with many trees of size 1. To avoid this, we can define a minimum depth for the trees we want to generate, and only consider terminals once the minimum size has been reached:


```python
MIN_DEPTH = 2
```


```python
def grow_random_tree(depth=0):
    tree = Node()
    if depth < MIN_DEPTH:
        tree.data = get_random_function()
    elif depth >= MAX_DEPTH:
        tree.data = get_random_terminal()
    else:
        if random.random() > 0.5:
            tree.data = get_random_terminal()
        else:
            tree.data = get_random_function()
                
    if tree.data in FUNCTIONS:
        tree.left  = grow_random_tree(depth + 1)
        tree.right = grow_random_tree(depth + 1)
        
    return tree
```


```python
root = grow_random_tree()
print_tree(root)
```




    
![svg](9/output_29_0.svg)
    



#### Ramped Half and Half

A common practice when creating the initial population is to use a mix of filled and grown trees. For example, half the population may be initialised with the full method, and half with the grow method.


```python
population_size = 40
```


```python
def get_initial_population():
    population = []
    while len(population) < population_size:
        if random.random() < 0.5:
            population.append(fill_random_tree())
        else: 
            population.append(grow_random_tree())
        
    return population
```

By defining a helper function that calculates the size of a tree, we can look at the distribution of tree sizes:


```python
def size(tree):
    if tree.data in FUNCTIONS:
        l = size(tree.left) if tree.left else 0
        r = size(tree.right) if tree.right else 0
        return 1 + l + r
    else:
        return 1
```


```python
plt.hist([size(x) for x in get_initial_population()])
```




    (array([ 2.,  0.,  1.,  0.,  0.,  5.,  0.,  6.,  0., 26.]),
     array([ 7. ,  7.8,  8.6,  9.4, 10.2, 11. , 11.8, 12.6, 13.4, 14.2, 15. ]),
     <BarContainer object of 10 artists>)




    
![png](9/output_36_1.png)
    


 Sometimes this is also done using a ramped approach, to generate trees of various sizes:


```python
def get_initial_population():
    population = []
    stages = MAX_DEPTH - MIN_DEPTH + 1
    for stage in range(MIN_DEPTH, MAX_DEPTH + 1):
        for md in range(int(population_size/stages/2)):
            population.append(grow_random_tree(MAX_DEPTH - stage))
        for md in range(int(population_size/stages/2)):
            population.append(fill_random_tree(MAX_DEPTH - stage))
    return population
```


```python
plt.hist([size(x) for x in get_initial_population()])
```




    (array([ 3.,  6.,  0., 11.,  0.,  2.,  3.,  0.,  5., 10.]),
     array([ 3. ,  4.2,  5.4,  6.6,  7.8,  9. , 10.2, 11.4, 12.6, 13.8, 15. ]),
     <BarContainer object of 10 artists>)




    
![png](9/output_39_1.png)
    


### Ephemeral Random Constants

Our terminals only include constants (`A` etc), but in practice we might also need actual values, such as numbers. Including all possible numbers in the set of terminals does not scale well. A common approach is therefore to include _ephemeral random constants_: A single terminal represents the choice of a random value; when the terminal is chosen then it is instantiated with an actual random value:


```python
EPHEMERAL_CONSTANT = "R"
TERMINALS = ['A', 'B', 'C', 'D', EPHEMERAL_CONSTANT]
EPHEMERAL_RANGE = 5
```


```python
def get_random_terminal():
    t = random.choice(TERMINALS)
    if t == EPHEMERAL_CONSTANT:
        t = random.randint(-EPHEMERAL_RANGE, EPHEMERAL_RANGE)
    return t
```


```python
root = grow_random_tree()
print_tree(root)
```




    
![svg](9/output_44_0.svg)
    



### Search Operators

Since we have a different representation to what we used in the past (which was mainly list-based), we need to adapt our search operators.

#### Crossover

A basic approach to crossing over two trees is to randomly choose two crossover points in the parent trees, and then to cut and swap subtrees below the crossover points. We first need a function with which we can create copies of trees:


```python
def copy(tree):
    t = Node()
    t.data = tree.data
    if tree.left:  t.left = copy(tree.left)
    if tree.right: t.right = copy(tree.right)
    return t 
```

We need a couple further helper functions in order to find subtrees at chosen crossover points. First, here's a helper class that allows us to count up or down while (hackily) traversing a tree:


```python
class Counter:
    def __init__(self, num):
        self.num = num
    
    def reduce(self):
        self.num -= 1
        
    def increase(self):
        self.num += 1
    
    def is_target(self):
        return self.num == 0
    
    def greater_zero(self):
        return self.num > 0
    
    def get_value(self):
        return self.num
```

Using this helper class, we can now traverse the tree and return a copy of the subtree whenever the counter has reached the target:


```python
def get_subtree_at(tree, counter):
    counter.reduce()
    if counter.is_target():
        return copy(tree)
    else:
        ret = None
        if tree.left and counter.greater_zero(): 
            ret = get_subtree_at(tree.left, counter)
        if not ret and tree.right and counter.greater_zero(): 
            ret = get_subtree_at(tree.right, counter)
        return ret
```

Similarly, we can replace a target node with an entirely different subtree given a counter that tells us when we've reached the target node:


```python
def insert_subtree_at(tree, subtree, counter):
    counter.reduce()
    if counter.is_target():
        tree.data  = subtree.data
        tree.left  = subtree.left
        tree.right = subtree.right
    else:
        if tree.left and counter.greater_zero(): 
            insert_subtree_at(tree.left, subtree, counter)
        if tree.right and counter.greater_zero(): 
            insert_subtree_at(tree.right, subtree, counter)
```

Using these two functions we can now define a simple crossover function that picks a random subtree from one parent (`parent1`) and inserts it at a random position in the other parent (`parent2`):


```python
def subtree_crossover(parent1, parent2):
    pos1 = random.randint(1, size(parent1))
    pos2 = random.randint(1, size(parent2))
    
    if size(parent1) == 1:
        subtree = copy(parent1)
    else:
        subtree = get_subtree_at(parent1, Counter(pos1))
    offspring = copy(parent2)
    insert_subtree_at(offspring, subtree, Counter(pos2))
    
    return offspring
```

Let's create some example parents to cross over:


```python
parent1 = grow_random_tree()
print_tree(parent1)
```




    
![svg](9/output_59_0.svg)
    




```python
parent2 = grow_random_tree()
print_tree(parent2)
```




    
![svg](9/output_60_0.svg)
    



Now we can produce an offspring by calling our `crossover` function with these two parents:


```python
offspring = subtree_crossover(parent1, parent2)
print_tree(offspring)
```




    
![svg](9/output_62_0.svg)
    



Often crossover points are not actually sampled with a uniform random distribution though: Given a tree with branching factor 2 or more, the majority of nodes will be leaves, which crossover will mostly cut a single leaf. A common alternative is to apply a 90% probability of choosing a function node, and only a 10% chance of choosing a terminal node.

#### Mutation


Subtree mutation (a.k.a. headless chicken mutation) is a simple mutation where we pick a random subtree, and replace it with a randomly generated subtree:


```python
P_mutate = 0.2
```


```python
def subtree_mutation(tree):
    mutation_point = random.randint(1, size(tree))
    random_subtree = grow_random_tree(MAX_DEPTH - 2)
    insert_subtree_at(tree, random_subtree, Counter(mutation_point))
```


```python
print_tree(parent1)
```




    
![svg](9/output_68_0.svg)
    




```python
offspring = copy(parent1)
subtree_mutation(offspring)
print_tree(offspring)
```




    
![svg](9/output_69_0.svg)
    



An alternative mutation operator is _point mutation_ where, for each node, we replace the node with a certain probability with another node of the same arity.


```python
def point_mutation(tree):
    if random.random() < P_mutate:
        if tree.data in FUNCTIONS: 
            tree.data = random.choice(FUNCTIONS)
        else:
            tree.data = get_random_terminal()
    
    if tree.left:
        point_mutation(tree.left)
    if tree.right:
        point_mutation(tree.right)
```


```python
print_tree(parent1)
```




    
![svg](9/output_72_0.svg)
    




```python
offspring = copy(parent1)
point_mutation(offspring)
print_tree(offspring)
```




    
![svg](9/output_73_0.svg)
    



#### Selection


Since selection is independent of the representation, we don't need any adaptations. Since we use a variable size representation, we will include the size as one of the selection criteria.


```python
tournament_size = 3
def tournament_selection(population):
    candidates = random.sample(population, tournament_size)        
    winner = min(candidates, key=lambda x: (get_fitness(x), size(x)))
    return winner
```

An evolution step of our GA now just needs to integrate these operators:


```python
def evolution_step(population):
    new_population = []
    while len(new_population) < len(population):
        parent1 = copy(selection(population))
        parent2 = copy(selection(population))

        if random.random() < P_xover:
            offspring = crossover(parent1, parent2)
        else:
            offspring = random.choice([parent1, parent2])

        if random.random() < 0.5:
            point_mutation(offspring)
        else:
            subtree_mutation(offspring)

        new_population.append(offspring)

    population.clear()
    population.extend(new_population)

    best_fitness = min([get_fitness(k) for k in population])
    return best_fitness
```

Similarly, the overall GA is also the same as always:


```python
fitness_values = []
size_values = []
```


```python
def ga():
    population = get_initial_population()
    best_fitness = sys.maxsize
    for p in population:
        fitness = get_fitness(p)
        if fitness < best_fitness or (fitness == best_fitness and size(p) < size(best_solution)):
            best_fitness = fitness
            best_solution = copy(p)

    iteration = 0
    while iteration < max_iterations and best_fitness > 0.000001:
        fitness_values.append(best_fitness)
        size_values.append(mean([size(x) for x in population]))
        print(f"GA Iteration {iteration}, best fitness: {best_fitness}, average size: {size_values[-1]}")
        iteration += 1
        evolution_step(population)

        for p in population:
            fitness = get_fitness(p)
            if fitness < best_fitness or (fitness == best_fitness and size(p) < size(best_solution)):
                best_fitness = fitness
                best_solution = copy(p)

    print(f"GA solution after {iteration} iterations, best fitness: {best_fitness}")
    return best_solution

```

### Example Problem: Symbolic Regression

We will use symbolic regression as a first example problem. Given a set of points, we would like to come up with a symbolic expression that represents a function approximating the points. As example equation, we will use $x^2 + x + 1$.

We only have one terminal `x` for this problem, but can also include ephemeral constants:


```python
TERMINALS = ["x", EPHEMERAL_CONSTANT]
```

As fitness function for how close we are to approximating a set of points we use the sum of absolute errors for all points at different values of `x` in the range [−1.0, +1.0]. Thus, a smaller fitness value is better; a fitness of zero indicates a perfect fit. In order to calculate the error for any given point, we need a helper function that evaluates a tree for a given value of `x`:


```python
def evaluate(tree, assignment):
    if tree.data in FUNCTIONS:
        return tree.data(evaluate(tree.left, assignment), evaluate(tree.right, assignment))
    elif tree.data in assignment:
        return assignment[tree.data]
    else:
        return tree.data
```

The dictionary `assignment` will map all terminals to actual values during the evaluation. We sample a number of points for measuring the error:


```python
test_data = {}
for value in [x/10 for x in range(-10, 10)]:
    test_data[value] = value*value + value + 1
```


```python
test_data
```




    {-1.0: 1.0,
     -0.9: 0.91,
     -0.8: 0.8400000000000001,
     -0.7: 0.79,
     -0.6: 0.76,
     -0.5: 0.75,
     -0.4: 0.76,
     -0.3: 0.79,
     -0.2: 0.84,
     -0.1: 0.91,
     0.0: 1.0,
     0.1: 1.11,
     0.2: 1.24,
     0.3: 1.3900000000000001,
     0.4: 1.56,
     0.5: 1.75,
     0.6: 1.96,
     0.7: 2.19,
     0.8: 2.4400000000000004,
     0.9: 2.71}



Now calculating the fitness function reduces to calling the `evaluate` function for each points in our test set, and summing up the differences:


```python
def get_fitness(tree):
    fitness = 0.0
    
    for (x, expected_result) in test_data.items():
        assignment = {"x": x}
        result = evaluate(tree, assignment)
        fitness += (result - expected_result) * (result - expected_result)
    fitness /= len(test_data)
    
    return fitness
```

With this, we can finally call our genetic algorithm:


```python
max_iterations = 200
selection = tournament_selection
crossover = subtree_crossover
P_xover = 0.7
fitness_values = []
size_values = []
result = ga()
print_tree(result)
```

    GA Iteration 0, best fitness: 0.31676750000000004, average size: 9.75
    GA Iteration 1, best fitness: 0.31676750000000004, average size: 10.85
    GA Iteration 2, best fitness: 0.31676750000000004, average size: 9.45
    GA Iteration 3, best fitness: 0.31676750000000004, average size: 9
    GA Iteration 4, best fitness: 0.31676750000000004, average size: 8.5
    GA Iteration 5, best fitness: 0.20333000000000007, average size: 10.25
    GA Iteration 6, best fitness: 0.20333000000000007, average size: 9.3
    GA Iteration 7, best fitness: 0.20333000000000007, average size: 11.55
    GA Iteration 8, best fitness: 0.20333000000000007, average size: 8.85
    GA Iteration 9, best fitness: 0.20333000000000007, average size: 9.65
    GA Iteration 10, best fitness: 0.20333000000000007, average size: 11.75
    GA Iteration 11, best fitness: 0.20333000000000007, average size: 10.75
    GA Iteration 12, best fitness: 0.20333000000000007, average size: 10.9
    GA Iteration 13, best fitness: 0.20333000000000007, average size: 13.4
    GA solution after 14 iterations, best fitness: 2.465190328815662e-33





    
![svg](9/output_94_1.svg)
    



Is the solution actually correct? We can plot the results of this function vs the results of the actual target function to compare:


```python
points = [x/10 for x in range(-10, 11)]
real_values = [(x*x + x + 1) for x in points]
gp_values = []
for x in points:
    assignment = {}
    assignment["x"] = x
    gp_values.append(evaluate(result, assignment))
plt.plot(points, gp_values)

plt.plot(points, real_values, linestyle='--', label = "Target")
plt.plot(points, gp_values,  linestyle=':', label = "GP")
plt.legend()
```




    <matplotlib.legend.Legend at 0x10c14e5c0>




    
![png](9/output_96_1.png)
    


The solution is usually correct; if it is not, try again, and maybe try to adjust the parameters of the GA.

### Handling Bloat

The solution is probably a fairly large tree. Indeed the size is sometimes quite problematic as genetic programming tends to suffer from _bloat_. Let's look at the average population size throughout the evolution:


```python
plt.plot(size_values)
```




    [<matplotlib.lines.Line2D at 0x10c21f250>]




    
![png](9/output_100_1.png)
    


We have already adapted our selection operator to counter bloat somewhat, but mutation and crossover also cause an increase in size. We therefore define alternative mutation and crossover operators.

Shrink mutation replaces a random subtree with a terminal node:


```python
def shrink_mutation(individual):
    num_nodes = size(individual)
    if num_nodes < 2:
        return
    
    mutation_point = random.randint(2, num_nodes - 1)
    node = Node()
    node.data = get_random_terminal()
    insert_subtree_at(individual, node, Counter(mutation_point))
    
    return individual
```


```python
print_tree(parent1)
```




    
![svg](9/output_104_0.svg)
    




```python
offspring = copy(parent1)
shrink_mutation(offspring)
print_tree(offspring)
```




    
![svg](9/output_105_0.svg)
    



Size-fair subtree mutation first picks a random subtree, and then replaces that with a new random subtree that is at most as big as the replaced subtree:


```python
def depth(node):
    d = 0
    if node.left:
        d = max(d, depth(node.left))
    if node.right:
        d = max(d, depth(node.right))
    return d + 1
        
```


```python
def sizefair_subtree_mutation(tree):
    mutation_point = random.randint(1, size(tree))
    replaced_tree = get_subtree_at(tree, Counter(mutation_point))
    random_subtree = grow_random_tree(MAX_DEPTH - depth(replaced_tree) + 1)
    insert_subtree_at(tree, random_subtree, Counter(mutation_point))
```


```python
print_tree(parent1)
```




    
![svg](9/output_109_0.svg)
    




```python
offspring = copy(parent1)
sizefair_subtree_mutation(offspring)
print_tree(offspring)
```




    
![svg](9/output_110_0.svg)
    



Permutation mutation shuffles the order of arguments for functions with arity > 1:


```python
def permutation_mutation(tree):
    if random.random() < P_mutate:
        if tree.data in FUNCTIONS: 
            tree.left, tree.right = tree.right, tree.left
    
    if tree.left:
        permutation_mutation(tree.left)
    if tree.right:
        permutation_mutation(tree.right)
```


```python
print_tree(parent1)
```




    
![svg](9/output_113_0.svg)
    




```python
offspring = copy(parent1)
permutation_mutation(offspring)
print_tree(offspring)
```




    
![svg](9/output_114_0.svg)
    



There is also a size-fair variant of the crossover operator we defined earlier: We first pick a random crossover point in the first parent. Then, we pick a subtree in the other parent that is not larger than the subtree in the first parent, and perform the crossover with this. We thus define a helper function that gives us all valid positions with subtrees of a maximum size:


```python
def get_smaller_subtree_positions(tree, max_size, positions, counter):
    if size(tree) <= max_size:
        positions.append(counter.get_value())
    counter.increase()
    
    if tree.left:
        get_smaller_subtree_positions(tree.left, max_size, positions, counter)
    if tree.right:
        get_smaller_subtree_positions(tree.right, max_size, positions, counter)
```

We apply this helper function by simply randomly picking one of the valid positions during crossover:


```python
def sizefair_crossover(parent1, parent2):
    
    # First pick a crossover point in parent1
    pos1 = 1
    if size(parent1) > 1:
        pos1 = random.randint(1, size(parent1) - 1)

    # Then select a subtree that isn't larger than what is being replaced from parent2
    subtree = get_subtree_at(parent1, Counter(pos1))
    subtree_size = size(subtree)    
    positions = []
    get_smaller_subtree_positions(parent2, subtree_size, positions, Counter(1))
    pos2 = random.choice(positions)
    subtree = get_subtree_at(parent2, Counter(pos2))
    
    # Then insert that into parent1
    offspring = copy(parent1)
    insert_subtree_at(offspring, subtree, Counter(pos1))
    
    assert size(offspring) <= size(parent1)
    
    return offspring
```


```python
print_tree(parent1)
```




    
![svg](9/output_119_0.svg)
    




```python
print_tree(parent2)
```




    
![svg](9/output_120_0.svg)
    




```python
offspring = sizefair_crossover(parent1, parent2)
print_tree(offspring)
```




    
![svg](9/output_121_0.svg)
    



Now we just need to do some minor adaptations to make sure that these new operators are actually applied during the evolution:


```python
mutation_operators = [subtree_mutation, permutation_mutation, shrink_mutation, point_mutation, sizefair_subtree_mutation]
crossover = sizefair_crossover
```

Besides the adapted operators, another common approach is to apply an upper bound on the size, and reject offspring that exceeds that size. This can have negative implications if it is the only means to control bloat, as larger individuals will then become less likely to produce surviving offspring. However, we can use it in conjunction with out optimised operators:


```python
REJECT_SIZE = 50
```


```python
def evolution_step(population):
    new_population = []
    while len(new_population) < len(population):
        parent1 = copy(selection(population))
        parent2 = copy(selection(population))

        if random.random() < P_xover:
            offspring = crossover(parent1, parent2)
        else:
            offspring = random.choice([parent1, parent2])

        if random.random() < 0.2:
            mutation_operator = random.choice(mutation_operators)
            mutation_operator(offspring)

        if size(offspring) < REJECT_SIZE:
            new_population.append(offspring)

    population.clear()
    population.extend(new_population)

    best_fitness = min([get_fitness(k) for k in population])
    return best_fitness
```

Thus, finally, let's run the search again for the same target function.


```python
max_iterations = 200
population_size = 100
selection = tournament_selection
crossover = sizefair_crossover
P_xover = 0.7
fitness_values = []
size_values = []
result = ga()
print_tree(result)
```

    GA Iteration 0, best fitness: 0.43833, average size: 9.38
    GA Iteration 1, best fitness: 0.43833, average size: 8.14
    GA Iteration 2, best fitness: 0.20333000000000007, average size: 7.3
    GA Iteration 3, best fitness: 0.20333000000000007, average size: 6.44
    GA Iteration 4, best fitness: 0.20333000000000007, average size: 6.14
    GA Iteration 5, best fitness: 0.20333000000000007, average size: 4.82
    GA Iteration 6, best fitness: 0.20333000000000007, average size: 4.74
    GA Iteration 7, best fitness: 0.20333000000000007, average size: 3.8
    GA Iteration 8, best fitness: 0.20333000000000007, average size: 3.84
    GA Iteration 9, best fitness: 0.20333000000000007, average size: 3.64
    GA Iteration 10, best fitness: 0.20333000000000007, average size: 4.02
    GA Iteration 11, best fitness: 0.20333000000000007, average size: 3.84
    GA Iteration 12, best fitness: 0.20333000000000007, average size: 4.12
    GA solution after 13 iterations, best fitness: 1.232595164407831e-33





    
![svg](9/output_128_1.svg)
    



To validate whether we are still affected by bloat we can plot the evolution of size again:


```python
plt.plot(size_values)
```




    [<matplotlib.lines.Line2D at 0x10c2d89d0>]




    
![png](9/output_130_1.png)
    


We can also compare the resulting function with the desired target function:


```python
points = [x/10 for x in range(-10, 11)]
real_values = [(x*x + x + 1) for x in points]
gp_values = []
for x in points:
    assignment = {}
    assignment["x"] = x
    gp_values.append(evaluate(result, assignment))
plt.plot(points, gp_values)

plt.plot(points, real_values,  linestyle='--', label = "Target")
plt.plot(points, gp_values,  linestyle=':', label = "GP")
plt.legend()
```




    <matplotlib.legend.Legend at 0x10c31eb90>




    
![png](9/output_132_1.png)
    


## Example Application: Fault Localisation

Given a test suite with at least one failing and at least one passing test case, the aim of fault localisation is to identify those statements in a program that are likely faulty. Intuitively, statements that are primarily executed by failed test cases are more likely to be faulty than those that are primarily executed by passed test cases. In spectrum-based fault localisation this is done by evaluating the similarity of each statement with the the error vector (i.e. the vector of pass/fail verdicts for each statement). There are many different competing similarity metrics (which we will consider in more details the Software Analysis course). 


The basis for fault localisation is a coverage matrix in which we have coverage information for each statement and each test case. These matrices serve to calculate several factors for each statement:

- $e_f$: Number of times the statement has been executed by failing tests.
- $e_p$: Number of times the statement has been executed by passing tests.
- $n_f$: Number of times the statement has _not_ been executed by failing tests.
- $n_p$: Number of times the statement has _not_ been executed by passing tests.

Based on these factors, different metrics to calculate the suspiciousness of a program statement have been defined, e.g.:

- Tarantula: $\frac{\frac{e_f}{e_f+n_f}}{\frac{e_p}{e_p+n_p} + \frac{e_f}{e_f+n_f}}$
- Barinel: $1 - \frac{e_p}{e_f + e_p}$
- Ochiai: $\frac{e_f}{\sqrt{(e_f + n_f) \cdot (e_f + e_p)}}$

The `middle` example function takes three parameters and should return the parameter that is the middle one when ranked by size.


```python
def middle(x, y, z):
    if y < z:
        if x < y:
            return y
        elif x < z:
            return y
    else:
        if x > y:
            return y
        elif x > z:
            return x
    return z
```


```python
middle(2, 3, 1)
```




    2



Our implementation of `middle` is unfortunately buggy, which we can demonstrate by generating some tests and checking the expected values.


```python
tests = []

for i in range(10):
    x = random.randrange(10)
    y = random.randrange(10)
    z = random.randrange(10)
    m = sorted([x,y,z])[1]
    tests.append((x,y,z,m))
```

Let's check if our tests can trigger a fault, otherwise we will need to generate some more:


```python
for (x,y,z,m) in tests:
    result = middle(x,y,z)
    if result != m:
        print(f"Failed test: {x},{y},{z} == {result} but should be {m}")
```

We will consider the program at the level of its lines:


```python
import inspect

lines = inspect.getsource(middle).splitlines()
for i, line in enumerate(lines):
    print(f"{i}: {line}")
```

    0: def middle(x, y, z):
    1:     if y < z:
    2:         if x < y:
    3:             return y
    4:         elif x < z:
    5:             return y
    6:     else:
    7:         if x > y:
    8:             return y
    9:         elif x > z:
    10:             return x
    11:     return z


In order to apply fault localisation we need to trace test executions and keep track of which lines were executed.


```python
trace = []
```


```python
def trace_line(frame, event, arg):
    if event == "line":
        trace.append(frame.f_lineno)
    return trace_line
```


```python
def middle_instrumented(x,y,z):
    global trace
    sys.settrace(trace_line)
    trace = []
    ret = middle(x,y,z)
    sys.settrace(None)
    return ret
```


```python
middle_instrumented(1,2,3)
trace
```




    [2, 3, 4]




```python
middle_instrumented(3,4,1)
trace
```




    [2, 8, 10, 11]



Now we can derive an execution spectrum for the `middle` function using our tests.


```python
import pandas as pd
```


```python
def get_spectrum(tests, statements):
    matrix = []
    for (x,y,z,m) in tests:
        row = []
        result = middle_instrumented(x,y,z)
        for lineno in statements:
            if lineno in trace:
                row.append(1)
            else:
                row.append(0)
        if result == m:
            row.append(1)
        else:
            row.append(0)
        matrix.append(row)
    
    spectrum = pd.DataFrame(matrix, columns=statements + ["Passed"])
    return spectrum
```


```python
statements = [i for i in range(len(lines))]
```


```python
middle_spectrum = get_spectrum(tests, statements)
middle_spectrum
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>Passed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's calculate Barinel using this spectrum (since it's the simplest formula).


```python
def get_suspiciousness(line, spectrum):
    
    if line not in spectrum.columns:
        return 0
    
    # Number of times the statement has been executed by failing tests.
    e_f = spectrum[(spectrum["Passed"] == 0) & (spectrum[line] == 1)].size
    
    # Number of times the statement has been executed by passing tests.
    e_p = spectrum[(spectrum["Passed"] == 1) & (spectrum[line] == 1)].size

    if e_p + e_f == 0:
        return 0
    
    suspiciousness = 1 - e_p / (e_p + e_f)
        
    return suspiciousness
```


```python
lines = inspect.getsource(middle).splitlines()
for i in range(len(lines)):
    line = lines[i]
    suspiciousness = get_suspiciousness(i+1, middle_spectrum)
    print("%2d %.2f: %s" % (i, suspiciousness, line))
    
```

     0 0.00: def middle(x, y, z):
     1 0.00:     if y < z:
     2 0.00:         if x < y:
     3 0.00:             return y
     4 0.00:         elif x < z:
     5 0.00:             return y
     6 0.00:     else:
     7 0.00:         if x > y:
     8 0.00:             return y
     9 0.00:         elif x > z:
    10 0.00:             return x
    11 0.00:     return z


Indeed the `return y` in line 5 is incorrect.

To make things more challenging we will parse some example spectra based on some more complex Java classes in the format produced by the GZoltar tool.


```python
def read_matrix(matrix_file, spectra_file, faulty_line):
    line_details = []
    fh = open(spectra_file)
    num_line = 0
    for line in fh:
        num_line += 1
        if num_line == 1:
            continue
        result = line.rstrip()
        line_details.append(result)
    fh.close()

    num_statements = len(line_details)

    ep = {}
    ef = {}
    np = {}
    nf = {}

    for i in range(num_statements):
        ep[i] = 0
        ef[i] = 0
        np[i] = 0
        nf[i] = 0

    fh = open(matrix_file)
    num_test = 0
    for line in fh:
        result = line.split(" ")

        test_result = result[-1].rstrip()
        for i in range(num_statements):
            if result[i] == '1' and test_result == '+':
                ep[i] += 1
            elif result[i] == '1' and test_result == '-':
                ef[i] += 1
            elif result[i] == '0' and test_result == '+':
                np[i] += 1
            elif result[i] == '0' and test_result == '-':
                nf[i] += 1

        num_test += 1
    fh.close()

    return (ep, ef, np, nf, line_details, num_statements, faulty_line)
```

Our aim is to use GP in order to evolve a similarity function that performs better than the standard metrics. We need some datapoints for our fitness evaluation again; for this we can use some example matrices:


```python
data_files = [('data/fl/matrix.txt', 'data/fl/spectra.csv', 7),
              ('data/fl/matrix-lift.txt', 'data/fl/spectra-lift.csv', 13),
              ('data/fl/matrix-complex.txt', 'data/fl/spectra-complex.csv', 39),
              ('data/fl/matrix-rational.txt', 'data/fl/spectra-rational.csv', 10)
              ]
eval_cases = []

for (matrix, spectrum, faulty_line) in data_files:
    eval_cases.append(read_matrix(matrix, spectrum, faulty_line))
```

In the following, we will try to use GP in order to derive a formula that produces better results on our dataset than any of these well-established coefficients. This idea is described in more detail in the following paper:

Yoo, S. (2012, September). Evolving human competitive spectra-based fault localisation techniques. In International Symposium on Search Based Software Engineering (pp. 244-258). Springer, Berlin, Heidelberg.

Similar to the existing coefficients, the components of our formula shall be simple arithmetic operations:


```python
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return 1 if not y else x/y
```


```python
FUNCTIONS = [add, sub, mul, div]
```

The terminals will be the factors listed above, as well as the constant 1:


```python
TERMINALS = ['ep', 'ef', 'np', 'nf', 1]
```

In order to evaluate the fitness of a particular formula, we apply it to our four (faulty) example programs, rank the statements by their suspiciousness, and then report the position at which the faulty statement is ranked. A perfect fault localisation means that the faulty statement is listed first. In case of ties, where multiple statements have the same suspiciousness score, we calculate the average position of the statements with the same suspiciousness:


```python
def get_fitness(tree):
    positions = []
    for (ep, ef, np, nf, line_details, num_statements, faulty_line) in eval_cases:
        lines = [line for line in range(num_statements)]
        weights = []
        for line in lines:
            assignment = {"np": np[line], 
                          "nf": nf[line], 
                          "ep": ep[line],
                          "ef": ef[line]}
            result = evaluate(tree, assignment)
            weights.append(result)
        lines, weights = zip(*sorted(zip(lines, weights), key=lambda t: t[1], reverse=True))

        start = lines.index(faulty_line)
        end = start
        faulty_weight = weights[start]
        while end < len(weights) and faulty_weight == weights[end]:
            end += 1

        positions.append((start + end)/2)
        
    return sum(positions)/len(positions)
```

In order to compare different ranking formulas, we will also look at the top 5 statements for each example:


```python
def print_ranking(tree):
    for (ep, ef, np, nf, line_details, num_statements, faulty_line) in eval_cases:
        lines = [line for line in range(num_statements)]
        weights = []
        for line in lines:
            assignment = {"np": np[line], 
                          "nf": nf[line], 
                          "ep": ep[line],
                          "ef": ef[line]}
            result = evaluate(tree, assignment)
            weights.append(result)

        lines, weights = zip(*sorted(zip(lines, weights), key=lambda t: t[1], reverse=True))

        print(f"Faulty line: {faulty_line}: {lines[:5]}")
```

Before evolving our own suspiciousness formula, let's express some standard formulas using our tree datastructure:

Tarantula: $\frac{\frac{e_f}{e_f+n_f}}{\frac{e_p}{e_p+n_p} + \frac{e_f}{e_f+n_f}}$


```python
tarantula = Node(div,
                 Node(div, Node('ef'), Node(add, Node('ef'), Node('nf'))),
                 Node(add, Node(div, Node('ep'), Node(add, Node('ep'), Node('np'))),
                           Node(div, Node('ef'), Node(add, Node('ef'), Node('nf'))))
                )
print_tree(tarantula)
```




    
![svg](9/output_176_0.svg)
    



Barinel: $1 - \frac{e_p}{e_f + e_p}$


```python
barinel = Node(sub, Node(1), Node(div, Node('ep'), Node(add, Node('ef'), Node('ep'))))
print_tree(barinel)
```




    
![svg](9/output_178_0.svg)
    



Barinel should perform better than Tarantula:


```python
get_fitness(tarantula)
```




    13.625




```python
get_fitness(barinel)
```




    6.625



But can we do better?


```python
max_iterations = 50
population_size = 50
selection = tournament_selection
crossover = sizefair_crossover
P_xover = 0.7
fitness_values = []
size_values = []
result = ga()
```

    GA Iteration 0, best fitness: 8.75, average size: 9.541666666666666
    GA Iteration 1, best fitness: 8.75, average size: 9.833333333333334
    GA Iteration 2, best fitness: 8.75, average size: 8.625
    GA Iteration 3, best fitness: 8.75, average size: 6.708333333333333
    GA Iteration 4, best fitness: 8.75, average size: 5.833333333333333
    GA Iteration 5, best fitness: 8.75, average size: 5.791666666666667
    GA Iteration 6, best fitness: 8.75, average size: 5.083333333333333
    GA Iteration 7, best fitness: 8.75, average size: 4.708333333333333
    GA Iteration 8, best fitness: 8.75, average size: 4.666666666666667
    GA Iteration 9, best fitness: 8.75, average size: 4.583333333333333
    GA Iteration 10, best fitness: 8.75, average size: 4.875
    GA Iteration 11, best fitness: 8.75, average size: 4.25
    GA Iteration 12, best fitness: 8.75, average size: 4.458333333333333
    GA Iteration 13, best fitness: 8.75, average size: 3.9166666666666665
    GA Iteration 14, best fitness: 8.75, average size: 4.041666666666667
    GA Iteration 15, best fitness: 8.375, average size: 4.416666666666667
    GA Iteration 16, best fitness: 8.375, average size: 4.625
    GA Iteration 17, best fitness: 8.375, average size: 4.833333333333333
    GA Iteration 18, best fitness: 8.375, average size: 4.708333333333333
    GA Iteration 19, best fitness: 8.375, average size: 4.625
    GA Iteration 20, best fitness: 8.375, average size: 3.8333333333333335
    GA Iteration 21, best fitness: 8.375, average size: 3.9583333333333335
    GA Iteration 22, best fitness: 8.375, average size: 4.333333333333333
    GA Iteration 23, best fitness: 8.375, average size: 4.041666666666667
    GA Iteration 24, best fitness: 8.375, average size: 4.375
    GA Iteration 25, best fitness: 8.375, average size: 4.166666666666667
    GA Iteration 26, best fitness: 8.375, average size: 4.416666666666667
    GA Iteration 27, best fitness: 8.375, average size: 4.416666666666667
    GA Iteration 28, best fitness: 8.375, average size: 4.291666666666667
    GA Iteration 29, best fitness: 8.375, average size: 4.291666666666667
    GA Iteration 30, best fitness: 8.375, average size: 4.291666666666667
    GA Iteration 31, best fitness: 8.375, average size: 3.7916666666666665
    GA Iteration 32, best fitness: 8.375, average size: 3.9166666666666665
    GA Iteration 33, best fitness: 8.375, average size: 3.8333333333333335
    GA Iteration 34, best fitness: 8.375, average size: 3.75
    GA Iteration 35, best fitness: 8.375, average size: 3.375
    GA Iteration 36, best fitness: 8.375, average size: 3.2083333333333335
    GA Iteration 37, best fitness: 8.375, average size: 3.5
    GA Iteration 38, best fitness: 8.375, average size: 3.5416666666666665
    GA Iteration 39, best fitness: 8.375, average size: 3.7083333333333335
    GA Iteration 40, best fitness: 8.375, average size: 4.166666666666667
    GA Iteration 41, best fitness: 8.375, average size: 4.333333333333333
    GA Iteration 42, best fitness: 8.375, average size: 4.166666666666667
    GA Iteration 43, best fitness: 8.375, average size: 3.5416666666666665
    GA Iteration 44, best fitness: 8.375, average size: 3.3333333333333335
    GA Iteration 45, best fitness: 8.375, average size: 3.9583333333333335
    GA Iteration 46, best fitness: 8.375, average size: 3.5833333333333335
    GA Iteration 47, best fitness: 8.375, average size: 3.7916666666666665
    GA Iteration 48, best fitness: 8.375, average size: 4.625
    GA Iteration 49, best fitness: 8.375, average size: 5.083333333333333
    GA solution after 50 iterations, best fitness: 8.375



```python
print_tree(result)
```




    
![svg](9/output_184_0.svg)
    




```python
get_fitness(result)
```




    8.375



Let's now compare the results of the standard metrics and our new one by looking at the top 5 most suspicious statements for each of the coefficients:


```python
print_ranking(tarantula)
```

    Faulty line: 7: (7, 17, 11, 6, 8)
    Faulty line: 13: (6, 7, 8, 13, 18)
    Faulty line: 39: (38, 39, 40, 42, 23)
    Faulty line: 10: (9, 11, 12, 46, 48)



```python
print_ranking(barinel)
```

    Faulty line: 7: (7, 11, 6, 8, 9)
    Faulty line: 13: (13, 0, 1, 2, 3)
    Faulty line: 39: (38, 39, 40, 42, 23)
    Faulty line: 10: (9, 72, 70, 71, 63)



```python
print_ranking(result)
```

    Faulty line: 7: (0, 1, 2, 3, 4)
    Faulty line: 13: (0, 1, 2, 3, 4)
    Faulty line: 39: (38, 39, 40, 42, 4)
    Faulty line: 10: (6, 7, 8, 10, 13)


The ranking is clearly better overall given these four example spectra. Obviously this isn't a really fair comparison, however, since we are measuring the performance on the training data. For more details of how this generalises, please refer to the original paper.

## Strongly-Typed Genetic Programming and Grammatical Evolution

Our implementation of GP is untyped, and thus requires type closure. Sometimes, depending on the problem to be solved, this can be an issue. In _strongly-typed genetic programming_ all nodes have types, and operators need to follow the type rules.

A common way to implement strongly-typed genetic programming is using _Grammar-Guided Genetic Programming (G3P)_: G3P is based on a grammar that describes the syntax of the target language.  Each individual of the population is a derivation tree of that grammer. (Derivation step = The process of replacing a nonterminal symbol with the symbols on the right-hand side of a production rule). G3P, in principle, works similar to our GP above, with the additional caveat that initialisation, crossover, and mutation need to be adapted in order to use the grammar and to ensure that only grammatically valid derivation trees are used.

An alternative way to use a grammar for evolution is using _Grammatical Evolution_ (GE): Whereas a classic GP uses syntax trees as representation, the representation in grammatical evolution is a simple linear genome. Each individual is a linear structure of codons (numbers) that select production rules to be chosen during derivation from a given grammar. 
Grammatical evolution clearly separates the search and solution spaces; syntactic correctness is guaranteed by the translation process; the problem of closure is removed, and any search algorithm can be used.

Since we are using a list representation, we can use our list helper class that can cache fitness values again:


```python
class L(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.
    """
    def __new__(self, *args, **kwargs):
        return super(L, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
```

In GE the chromosome size is fixed. For example, let's start by setting the number of codons to 50. Each codon is a number in a given range, which we will encode using `MAX`:


```python
num_codons = 50
MAX = 50
```

Individuals are simply lists of (randomly generated) codons:


```python
def get_random_individual():
    individual = L([random.randint(0, MAX) for _ in range(num_codons)])
    evaluate(individual)
    return individual
```

This, in turn, means we can use our usual approach for mutating each gene in a chromosome with a probability dependent on the overall number of genes:


```python
def mutate(individual):
    P_mutate = 1/len(individual)
    mutated = L(individual[:])
    for pos in range(len(individual)):
        if random.random() < P_mutate:
            mutated[pos] = random.randint(0, MAX)
    evaluate(mutated)
    return mutated
```

We can use standard crossover operators:


```python
def singlepoint_crossover(parent1, parent2):
    pos = random.randint(0, len(parent1))
    offspring1 = L(parent1[:pos] + parent2[pos:])
    offspring2 = L(parent2[:pos] + parent1[pos:])
    return (offspring1, offspring2)
```

...and we can also use standard parent and survivor selection operators:


```python
elite_size = 2
def elitism(population):
    population.sort(key=lambda k: get_fitness(k))
    return population[:elite_size]
```


```python
tournament_size = 3
def tournament_selection(population):
    candidates = random.sample(population, tournament_size)        
    winner = min(candidates, key=lambda x: get_fitness(x))
    return winner
```

In order to explain the process of decoding codons to production rules, we need to define a grammar for an example problem. Let's assume we want to generate Python code for a function with two parameters, which returns the larger of the two parameters. Our grammar thus covers linear sequences of code with if-statements, where we restrict each if-statement to a single operation. Operations can be updates of a local variable `__return__` which, at the end of the function, will define the return value:


```python
EXAMPLE_GRAMMAR = {
    "<start>":
        ["<code>"],

    "<code>":
        ["<line>", "<code>\n<line>"],

    "<line>":
        ["    <expr>"],

    "<expr>":
        ["<if-statement>",
         "<op>"],

    "<if-statement>":
        ["if <condition>:\n        <op>"],

    "<condition>":
        ["<param> <operator> <param>"],
    "<param>":
        ["x", "y"],
    "<operator>":
        ["<", ">", "=="],
    "<op>":
        ["__return__ = x", "__return__ = y"]
}
```

One of the production rules must be explicitly chosen as the start label"


```python
START_SYMBOL = "<start>"
```

In order to determine whether a production rule contains further non-terminals we use a simple regular expression. Anything within `<` and `>` counts as a non-terminal:


```python
RE_NONTERMINAL = re.compile(r'(<[^<> ]*>)')
```


```python
def nonterminals(expansion):
    return re.findall(RE_NONTERMINAL, expansion)
```

Depending on the grammar used, a derivation can be of any size; we need to put some limit on the size of the derivations:


```python
max_expansions = 50
```

In order to decode an individual we start with the start rule, and apply derivation. Whenever there is a choice of multiple production rules during the derivation, the number of the rule is determined by the next codon; i.e., we always pick the rule with the number _codon_ $%$ _number of rules_. Non-terminals are resolved from left to right (we need to make sure that the derivation is deterministic). Whenever we use a codon, we move to the next number in the chromosome. If we hit the last codon in a chromosome then we wrap and start over from the first one.


```python
def decode(individual, grammar):
    pos = 0
    term = START_SYMBOL
    num_expansions = 0
    count = 0
    
    while len(nonterminals(term)) > 0 and num_expansions < max_expansions:
        
        symbol_to_expand = nonterminals(term)[0]
        expansions = grammar[symbol_to_expand]
        if len(expansions) == 1:
            expansion = expansions[0]
        else:
            codon = individual[pos]
            pos = (pos + 1) % len(individual)
            expansion = expansions[codon % len(expansions)]
            
        new_term = term.replace(symbol_to_expand, expansion, 1)
    
        term = new_term
        num_expansions += 1

    return term
```


```python
print(decode([1,2,3], EXAMPLE_GRAMMAR))
```

        __return__ = y
        if y > x:
            __return__ = y


Given the string representation of a program we need to create an executable version in order to measure the fitness. We already compiled code at runtime in the previous chapter, so we essentially do the same again here. For any given string representation of a function, e.g.:


```python
text = """
def foo():
  print("Foo")
"""
```

We call the `compile` function which produces a code object. This code object we insert into the dictionary of modules such that we can call the function using its name, e.g.:


```python
code = compile(text, filename="<GE>", mode="exec")
current_module = sys.modules[__name__]
exec(code, current_module.__dict__)
```


```python
foo()
```

    Foo


We use a function that puts the decoded chromosome into a function, and we'll include the function signature and return value for now:


```python
def create_function(individual):
    program_text = decode(individual, EXAMPLE_GRAMMAR)
    source = """def generated_function(x, y):
    __return__ = 0
%s
    return __return__
""" % program_text
    code = compile(source, filename="<GE>", mode="exec")
    current_module = sys.modules[__name__]
    exec(code, current_module.__dict__)
    return generated_function
```

In order to calculate the fitness value we just need to run the function a number of times with some test data that represents the function we want to learn. Let's produce some test data again; let our example function return the maximum of the two input parameters:


```python
training_data = []
```


```python
for i in range(20):
    num1 = random.randint(0, MAX)
    num2 = random.randint(0, MAX)
    training_data.append(([num1, num2], max([num1, num2])))

training_data
```




    [([44, 6], 44),
     ([36, 7], 36),
     ([30, 48], 48),
     ([31, 50], 50),
     ([19, 32], 32),
     ([37, 37], 37),
     ([33, 28], 33),
     ([22, 0], 22),
     ([6, 18], 18),
     ([46, 40], 46),
     ([46, 23], 46),
     ([16, 16], 16),
     ([42, 46], 46),
     ([41, 17], 41),
     ([1, 8], 8),
     ([33, 38], 38),
     ([18, 47], 47),
     ([8, 28], 28),
     ([40, 38], 40),
     ([14, 45], 45)]



There may be errors during the execution; for example, if we hit the maximum number of expansions (`max_expansions`) then the code may not compile. There may be an exception if there is a division by zero or some other error. In these cases we assign the function the maximum (worst) fitness value and let selective pressure handle this for us:


```python
def evaluate(individual):
    fitness = len(training_data)
    individual.fitness = fitness
    try:
        p = create_function(individual)

        for (test_data, test_output) in training_data:
            result = p(*test_data)
            if result == test_output:
                fitness -= 1
    except:
        return len(training_data)

    individual.fitness = fitness
    return fitness
```


```python
x = get_random_individual()
x.fitness
```




    8



Since we call the `evaluate` function directly after generating or mutating individuals, retrieving the fitness function just consists of retrieving the cached fitness value:


```python
def get_fitness(individual):
    return individual.fitness
```

In order to use our new individuals we need to modify our GA such that it uses this new representation rather than the tree structure we defined earlier; that is, we need to replace the ramped half and half initialisation with a simple random sampling:


```python
def get_initial_population():
    return [get_random_individual() for _ in range(population_size)]
```

Furthermore, since we now have a fixed size representation we can omit the handling of size as a secondary criterion during selection:


```python
def evolution_step(population):
    new_population = []
    while len(new_population) < len(population):
        parent1 = selection(population)
        parent2 = selection(population)

        if random.random() < P_xover:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        offspring1 = mutate(offspring1)
        offspring2 = mutate(offspring2)
        
        new_population.append(offspring1)
        new_population.append(offspring2)

    population.clear()
    population.extend(new_population)
```


```python
def ga():
    population = get_initial_population()
    best_fitness = sys.maxsize
    for p in population:
        fitness = get_fitness(p)
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = L(p[:])

    iteration = 0
    print(f"GA Iteration {iteration}, best fitness: {best_fitness}")
    while iteration < max_iterations and best_fitness > 0:
        fitness_values.append(best_fitness)
        iteration += 1
        evolution_step(population)

        for p in population:
            fitness = get_fitness(p)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = L(p[:])
                print(f"GA Iteration {iteration}, best fitness: {best_fitness}")

    print(f"GA solution after {iteration} iterations, best fitness: {best_fitness}")
    return best_solution

```

With this, we can apply the grammatical evolution to produce our own max function:


```python
population_size = 20
P_xover = 0.7
max_iterations = 100
num_codons = 20
selection = tournament_selection
crossover = singlepoint_crossover
fitness_values = []
result = ga()
```

    GA Iteration 0, best fitness: 8
    GA Iteration 2, best fitness: 0
    GA solution after 2 iterations, best fitness: 0



```python
print(decode(result, EXAMPLE_GRAMMAR))
```

        if y == y:
            __return__ = y
        if y < x:
            __return__ = x


This problem is probably slightly too easy, so let's increase difficulty by expanding to three parameters. We therefore require a corresponding function template:


```python
def create_function(individual):
    program_text = decode(individual, EXAMPLE_GRAMMAR)
    source = """def generated_function(x, y, z):
    __return__ = 0
%s
    return __return__
""" % program_text
    code = compile(source, filename="<GE>", mode="exec")
    current_module = sys.modules[__name__]
    exec(code, current_module.__dict__)
    return generated_function
```

As well as some test data:


```python
training_data = []
for i in range(1000):
    num1 = random.randint(0, MAX)
    num2 = random.randint(0, MAX)
    num3 = random.randint(0, MAX)
    
    training_data.append(([num1, num2, num3], max([num1, num2, num3])))
```

We'll slightly expand our grammar to also include conjunctions and disjunctions and more relational operators:


```python
EXAMPLE_GRAMMAR = {
    "<start>":
        ["<code>"],

    "<code>":
        ["<line>", "<code>\n<line>"],

    "<line>":
        ["    <expr>"],

    "<expr>":
        ["<if-statement>",
         "<op>"],

    "<if-statement>":
        ["if <condition>:\n        <op>"],

    "<condition>":
        ["<param> <operator> <param>", "<param> <operator> <param> and <param> <operator> <param>", "<param> <operator> <param> or <param> <operator> <param>"],

    "<param>":
        ["x", "y", "z", "__return__"],

    "<operator>":
        ["<", ">", "<=", ">=", "==", "!="],

    "<op>":
        ["__return__ = <param>"]
}
```


```python
population_size = 100
P_xover = 0.7
max_iterations = 500
max_expansions = 80
num_codons = 80
selection = tournament_selection
tournament_size = 4
crossover = singlepoint_crossover
fitness_values = []
result = ga()
```

    GA Iteration 0, best fitness: 350
    GA Iteration 3, best fitness: 299
    GA Iteration 38, best fitness: 195
    GA Iteration 128, best fitness: 162
    GA Iteration 132, best fitness: 154
    GA Iteration 134, best fitness: 0
    GA solution after 134 iterations, best fitness: 0



```python
print(decode(result, EXAMPLE_GRAMMAR))
```

        __return__ = x
        if y == y or x >= y:
            __return__ = y
        if x >= y:
            __return__ = x
        if z > __return__ and z <= z:
            __return__ = z
        __return__ = __return__



```python
create_function(result)
```




    <function __main__.generated_function(x, y, z)>




```python
generated_function(1,2,3)
```




    3



Finding the maximum of three numbers appears to be relatively easy, and the search often finds a solution. Let's try and see if we can also reproduce our `middle` program. For this, we need new tests.


```python
training_data = []
for i in range(1000):
    num1 = random.randint(0, MAX)
    num2 = random.randint(0, MAX)
    num3 = random.randint(0, MAX)
    
    training_data.append(([num1, num2, num3], sorted([num1, num2, num3])[1]))
```


```python
population_size = 100
P_xover = 0.7
max_iterations = 500
max_expansions = 80
num_codons = 80
selection = tournament_selection
tournament_size = 10
crossover = singlepoint_crossover
fitness_values = []
result = ga()
```

    GA Iteration 0, best fitness: 607
    GA Iteration 2, best fitness: 510
    GA Iteration 6, best fitness: 501
    GA Iteration 8, best fitness: 489
    GA Iteration 45, best fitness: 473
    GA Iteration 47, best fitness: 464
    GA Iteration 50, best fitness: 457
    GA Iteration 305, best fitness: 314
    GA Iteration 427, best fitness: 307
    GA solution after 500 iterations, best fitness: 307



```python
print(decode(result, EXAMPLE_GRAMMAR))
```

        __return__ = x
        if __return__ < y:
            __return__ = z
        if y < __return__ and __return__ >= z:
            __return__ = y


This appears to be a somewhat more challenging problem, and the search tends to take much longer to find an acceptable solution (probably longer than is reasonable to wait in this Jupyter notebook).

More practical information about Genetic Programming can be found in the free ebook [A Field Guide to Genetic Programming](http://www0.cs.ucl.ac.uk/staff/W.Langdon/ftp/papers/poli08_fieldguide.pdf)

# Genetic Improvement

The programs we tried to evolve in this chapter have all been comparatively tiny. Indeed GPs are not generally used to evolve large programs, as they cannot easily come up with complex system designs and architectures. While evolving large programs from scratch may currently not be possible, it is possible to improve existing programs. This is the aim of program repair and _Genetic Improvement_ (GI). There are two central hypotheses that motivate Genetic Improvement:

The _Competent Programmer Hypothesis_ assumes that programmers write programs that are close to being correct, and program faults are syntactically small.

The _Plastic Surgery Hypothesis_ states that the content of new code can often be assembled out of fragments of code that already exist. Changes to a codebase contain snippets that already exist in the codebase at the time of the change, and these snippets can be efficiently found and exploited.

Given these two hypotheses, the idea of Genetic Improvement is to use GP to slightly modify a large, existing, code base, rather than to evolve something from scratch. 

The representation is typically based either in patches or modifications of the abstract syntax tree (AST) of the program. Thus, in contrast to the past examples where we implicitly defined our encoding through the `get_random_individual` function, in this case the representation is already defined by the AST of the programs we are trying to evolve.  For example, let's consider the AST of the `middle` function we considered earlier.


```python
import ast
```


```python
middle_ast = ast.parse(inspect.getsource(middle))
```


```python
ast.dump(middle_ast)
```




    "Module(body=[FunctionDef(name='middle', args=arguments(posonlyargs=[], args=[arg(arg='x'), arg(arg='y'), arg(arg='z')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[If(test=Compare(left=Name(id='y', ctx=Load()), ops=[Lt()], comparators=[Name(id='z', ctx=Load())]), body=[If(test=Compare(left=Name(id='x', ctx=Load()), ops=[Lt()], comparators=[Name(id='y', ctx=Load())]), body=[Return(value=Name(id='y', ctx=Load()))], orelse=[If(test=Compare(left=Name(id='x', ctx=Load()), ops=[Lt()], comparators=[Name(id='z', ctx=Load())]), body=[Return(value=Name(id='y', ctx=Load()))], orelse=[])])], orelse=[If(test=Compare(left=Name(id='x', ctx=Load()), ops=[Gt()], comparators=[Name(id='y', ctx=Load())]), body=[Return(value=Name(id='y', ctx=Load()))], orelse=[If(test=Compare(left=Name(id='x', ctx=Load()), ops=[Gt()], comparators=[Name(id='z', ctx=Load())]), body=[Return(value=Name(id='x', ctx=Load()))], orelse=[])])]), Return(value=Name(id='z', ctx=Load()))], decorator_list=[])], type_ignores=[])"



We know from earlier that the `middle` function contains a fault. The classic application of Genetic Improvement is _Automated Program Repair_: Given a program and a test suite in which there is at least one failing test, try to evolve the program such that it passes all test cases. This idea was initially described in the following seminal paper:

Le Goues, C., Nguyen, T., Forrest, S., & Weimer, W. (2011). Genprog: A generic method for automatic software repair. IEEE Transactions on Software Engineering, 38(1), 54-72.

Unlike the GP approaches we defined earlier, the starting point for GI is not a population of random individuals, but the given program we want to improve. Therefore, we will not start by defining a `get_random_individual` function this time, but first introduce mutations on an AST, and then the random individuals will be a result of mutating the existing program. As the mutation will be dependent on executions, we first consider the fitness function. The search in automated program repair is guided by tests: Given a set of tests, we want to maximise the number of passing tests. Let's define some passing and failing tests for our defective `middle` function:


```python
PASSING_TESTS = []
FAILING_TESTS = []

num_tests = 20

while len(PASSING_TESTS) + len(FAILING_TESTS) < 2 * num_tests:
    x = random.randrange(10)
    y = random.randrange(10)
    z = random.randrange(10)
    m = middle(x, y, z) # Actual value
    expected = sorted([x, y, z])[1] # Expected value
    if m == expected:
        if len(PASSING_TESTS) < num_tests:
            PASSING_TESTS.append((x, y, z, m))
    else:
        if len(FAILING_TESTS) < num_tests:
            FAILING_TESTS.append((x, y, z, expected))
```


```python
PASSING_TESTS
```




    [(4, 3, 3, 3),
     (0, 5, 5, 5),
     (3, 8, 1, 3),
     (7, 0, 4, 4),
     (5, 1, 3, 3),
     (9, 2, 8, 8),
     (3, 6, 8, 6),
     (9, 5, 7, 7),
     (5, 9, 0, 5),
     (0, 4, 0, 0),
     (5, 3, 0, 3),
     (6, 8, 0, 6),
     (1, 7, 1, 1),
     (7, 3, 7, 7),
     (6, 2, 4, 4),
     (4, 0, 4, 4),
     (2, 3, 2, 2),
     (7, 8, 2, 7),
     (3, 0, 1, 1),
     (1, 7, 9, 7)]




```python
FAILING_TESTS
```




    [(1, 0, 8, 1),
     (8, 5, 9, 8),
     (1, 0, 2, 1),
     (6, 4, 7, 6),
     (2, 1, 7, 2),
     (3, 0, 9, 3),
     (1, 0, 5, 1),
     (2, 1, 9, 2),
     (6, 2, 9, 6),
     (4, 2, 6, 4),
     (3, 1, 6, 3),
     (3, 0, 5, 3),
     (1, 0, 9, 1),
     (4, 1, 7, 4),
     (8, 4, 9, 8),
     (5, 2, 7, 5),
     (5, 3, 7, 5),
     (6, 0, 9, 6),
     (8, 6, 9, 8),
     (4, 2, 5, 4)]



The fitness function is a weighted combination of the passed tests -- intuitively, it is more progress to fix a failing test than to break a passing test, so we assign a higher weight to making failing tests pass.


```python
WEIGHT_PASSING = 1
WEIGHT_FAILING = 10
```

To calculate the fitness value, we need to compile the AST for a given individual, execute all our tests, and count how many passing tests there are.


```python
def get_fitness(individual):
    global PASSING_TESTS, FAILING_TESTS
    original_middle = middle

    code = compile(individual, '<fitness>', 'exec')
    exec(code, globals())

    passing_passed = 0
    failing_passed = 0

    for x, y, z, m in PASSING_TESTS:
        if middle(x, y, z) == m:
            passing_passed += 1

    for x, y, z, m in FAILING_TESTS:
        if middle(x, y, z) == m:
            failing_passed += 1

    fitness = WEIGHT_PASSING * passing_passed + WEIGHT_FAILING * failing_passed
        
    globals()['middle'] = original_middle
    return fitness
```


```python
get_fitness(middle_ast)
```




    20



When mutating individuals we consider the level of statements. We therefore need to extract all statements from a program in order to decide where to mutate, and how to mutate.


```python
class StatementVisitor(ast.NodeVisitor):

    def __init__(self):
        self.statements = []
        self.statements_seen = set()
        super().__init__()

    def add_statements(self, node, attr):
        elems = getattr(node, attr, [])
        if not isinstance(elems, list):
            elems = [elems]

        for stmt in elems:
            if stmt in self.statements_seen:
                continue

            self.statements.append(stmt)
            self.statements_seen.add(stmt)

    def visit_node(self, node):
        # Any node other than the ones listed below
        self.add_statements(node, 'body')
        self.add_statements(node, 'orelse')

    def visit_Module(self, node: ast.Module):
        # Module children are defs, classes and globals - don't add
        super().generic_visit(node)
        
    def generic_visit(self, node):
        self.visit_node(node)
        super().generic_visit(node)
```


```python
def all_statements(tree, tp = None):
    visitor = StatementVisitor()
    visitor.visit(tree)
    statements = visitor.statements
    if tp is not None:
        statements = [s for s in statements if isinstance(s, tp)]

    return statements
```


```python
all_statements(middle_ast)
```




    [<ast.If at 0x10e685c60>,
     <ast.Return at 0x10e687220>,
     <ast.If at 0x10e6856c0>,
     <ast.If at 0x10e6850f0>,
     <ast.Return at 0x10e686ad0>,
     <ast.If at 0x10e684970>,
     <ast.Return at 0x10e687730>,
     <ast.Return at 0x10e6852d0>,
     <ast.If at 0x10e684940>,
     <ast.Return at 0x10e684c70>]



This allows us to choose random nodes in our AST.


```python
import astor
```


```python
random_node = random.choice(all_statements(middle_ast))
print(astor.to_source(random_node))
```

    return x
    


GenProg defines three types of mutations: AST-Nodes can be _swapped_, _inserted_, or _deleted_. There are two important aspects to this mutation:

- First, the location at which an insertion, swap, or deletion happens depends on the suspiciousness of the nodes as determined using fault localisation. This way, the search should focus on code that is likely faulty.

- Second, for the chosen location the mutations do not use random code, but try to re-use the existing code, as the assumption is that the relevant code is likely already contained in the program.

In order to implement these mutations, we define a helper function that replaces AST nodes.


```python
import copy
```


```python
class NodeReplacer(ast.NodeTransformer):

    def __init__(self, target_node, mutant_node):
        self.target_node = target_node
        self.mutant_node = mutant_node

    def visit(self, node):
        if node == self.target_node:
            return self.mutant_node
        else:
            super().visit(node)
            return node
```

For the insertion mutation we need a helper function that inserts a new node after a chosen target `node`.


```python
def mutate_insert(node, statements):
    new_node = copy.deepcopy(random.choice(statements))

    if isinstance(node, ast.Return):
        if isinstance(new_node, ast.Return):
            return new_node
        else:
            return [new_node, node]

    return [node, new_node]
```

Swapping is very similar, except that we drop the original node.


```python
def mutate_swap(node, statements):     
    replacement_node = copy.deepcopy(random.choice(statements))
    return replacement_node
```

Finally, for deletion we remove a node by setting it to `None`, unless that would make the program syntactically invalid, in which case in Python we need a `pass` statement.


```python
def mutate_delete(node, statements):
    branches = [attr for attr in ['body', 'orelse', 'finalbody'] if hasattr(node, attr) and getattr(node, attr)]
    if branches:
        # Replace `if P: S` by `S`
        branch = random.choice(branches)
        new_node = getattr(node, branch)
        return new_node

    if isinstance(node, ast.stmt):
        # Avoid empty bodies; make this a `pass` statement
        new_node = ast.Pass()
        ast.copy_location(new_node, node)
        return new_node

    return None  # Just delete
```

The final ingredient is the selection of nodes, which we make dependent on the weights as defined by the fault localisation. Note that our implementation redundantly executes all tests, this could of course be optimised.


```python
def select_node(individual):
    tests = PASSING_TESTS + FAILING_TESTS
    statements = all_statements(individual)
    spectrum = get_spectrum(tests, statements)

    if len(statements) < 3:
        return random.choice(statements)
    
    # Crude hack for probabilistic choice
    candidates = random.sample(statements, 3)
    return max(candidates, key=lambda x: get_suspiciousness(x.lineno-1, spectrum))
```

Now we have all ingredients in place, and can implement the overall mutation operator.


```python
def mutate(individual):
    mutant = copy.deepcopy(individual)
    statements = all_statements(mutant)

    node = select_node(mutant)

    mutation = random.choice([mutate_insert, mutate_swap, mutate_delete])
    mutated_node = mutation(node, statements)
    
    mutator = NodeReplacer(node, mutated_node)
    mutant = mutator.visit(mutant)

    ast.fix_missing_locations(mutant)
    return mutant 
```

Our mutation operator only picks a single node and applies a mutation. In the original GenProg implementation, nodes are probabilistically mutated.


```python
print(astor.to_source(mutate(middle_ast)))
```

    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                return y
        elif x > y:
            if x > y:
                return y
            elif x > z:
                return x
            return y
        elif x > z:
            return x
        return z
    


This is also exactly how the random individuals are created: Not by creating random ASTs, but by applying a mutation to the original program.


```python
def get_random_individual():
    node = mutate(middle_ast)
    return node
```


```python
print(astor.to_source(get_random_individual()))
```

    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                return y
        elif x > y:
            return y
        elif x > z:
            return x
        if x > z:
            return x
        return z
    


We need to make sure that our initial population uses this function, rather than the ramped half and half we previously defined.


```python
def get_initial_population():
    return [copy.deepcopy(middle_ast)] + [get_random_individual() for _ in range(population_size - 1)]
```

Given our past implementations and the mutation operators we just defined, the crossover is quite simple: We select nodes to split, and then re-attach the programs. Note that this is once again a simplification compared to GenProg; for example, GenProg only considers 'suspicious' statements as possible crossover locations.


```python
def crossover(parent1, parent2):
    statements1 = all_statements(parent1)
    xover_point_1 = random.choice(statements1)

    statements2 = all_statements(parent2)
    xover_point_2 = random.choice(statements2)
    
    mutator = NodeReplacer(xover_point_1, copy.deepcopy(xover_point_2))
    offspring1 = mutator.visit(parent1)

    mutator = NodeReplacer(xover_point_2, copy.deepcopy(xover_point_1))
    offspring2 = mutator.visit(parent2)

    return offspring1, offspring2
```


```python
parent1 = get_random_individual()
parent2 = get_random_individual()
print(astor.to_source(parent1))
print(astor.to_source(parent2))
```

    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                if x < y:
                    return y
                elif x < z:
                    return y
                return y
        elif x > y:
            return y
        elif x > z:
            return x
        return z
    
    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                return y
        else:
            return y
        return z
    



```python
offspring1, offspring2 = crossover(parent1, parent2)
print(astor.to_source(offspring1))
print(astor.to_source(offspring2))
```

    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                if x < y:
                    return y
                elif x < z:
                    return y
                return y
        elif x > y:
            return y
        elif x > z:
            return x
        return z
    
    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                return y
        else:
            return y
        return z
    


While the selection is independent of the representation, we do need to take into account that we now have a maximisation problem.


```python
def get_size(individual):
    return sum([1 for _ in ast.walk(individual)])
```


```python
tournament_size = 3
def tournament_selection(population):
    candidates = random.sample(population, tournament_size)
    winner = max(candidates, key=lambda x: get_fitness(x))
    return winner
```

GenProg makes several adaptations to the overall genetic algorithm; we'll just adapt our existing standard implementation to make use of the new operators.


```python
def ga():
    population = get_initial_population()
    best_fitness = 0
    for p in population:
        fitness = get_fitness(p)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = copy.deepcopy(p)

    iteration = 0
    while iteration < max_iterations and \
            best_fitness < WEIGHT_PASSING * len(PASSING_TESTS) + WEIGHT_FAILING * len(FAILING_TESTS):
        fitness_values.append(best_fitness)
        print(f"GA Iteration {iteration}, best fitness: {best_fitness}")
        iteration += 1
        evolution_step(population, best_solution)

        for p in population:
            fitness = get_fitness(p)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = copy.deepcopy(p)

    print(f"GA solution after {iteration} iterations, best fitness: {best_fitness}")
    return best_solution
```

Since our mutation operator will always change exactly one node, we need to add a probability for mutation.


```python
P_mutation = 0.4
```


```python
def evolution_step(population, elite):
    new_population = [elite, get_random_individual()]
    viable_population = [p for p in population if get_fitness(p) > 0]
    while len(new_population) < len(population):
        parent1 = copy.deepcopy(selection(viable_population))
        parent2 = copy.deepcopy(selection(viable_population))

        offspring1, offspring2 = crossover(copy.deepcopy(parent1), copy.deepcopy(parent2))

        if random.random() < P_mutation:
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)

        new_population.append(offspring1)
        new_population.append(offspring2)
        new_population.append(parent1)
        new_population.append(parent2)

    population.clear()
    population.extend(new_population)

    best_fitness = max([get_fitness(k) for k in population])
    return best_fitness
```

Can we fix the `middle` program?


```python
P_xover = 0.0
P_mutation = 0.4
max_iterations = 50
population_size = 40
tournament_size = 5
fitness_values = []
result = ga()

print(astor.to_source(result))
```

    GA solution after 0 iterations, best fitness: 220
    def middle(x, y, z):
        if y < z:
            if x < y:
                return y
            elif x < z:
                return x
        elif x > y:
            return y
        elif x > z:
            return x
        return z
    


So how well does the attempted fix work? Let's print out all the failing tests from our test set, if there are any.


```python
code = compile(result, '<fitness>', 'exec')
exec(code, globals())

for x, y, z, m in PASSING_TESTS:
    if middle(x, y, z) != m:
        print(f"Failed {x}, {y}, {z} == {m}")

for x, y, z, m in FAILING_TESTS:
    if middle(x, y, z) != m:
        print(f"Failed {x}, {y}, {z} == {m}")
```

We've taken several shortcuts in this implementation; the original GenProg implementation has some differences in terms of the genetic operators and the search algorithm. A more elaborate discussion and implementation of the can be found in Andreas Zeller's Debugging Book: https://www.debuggingbook.org/
