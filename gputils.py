import random
import operator
import math
from deap import base, creator, gp, tools
from robots import Robot
from tqdm import tqdm
from graphviz import Digraph
import functools


def create_primitive_set(ifelse, logicals, angle_primitives):
    """
    Creates the primitives sets.

    Terminals:
    - 5 inputs (the robot's), 2 ephemeral constants

    Functions:
    - +, -. *, /, neg, max, min

    Optional primitives:
    - if-else, <, >, sin, cos, eph. (-pi, pi)
    """

    pset = gp.PrimitiveSet('MAIN', 5)  # 5 sensor inputs
    
    # Rename arguments to match the robots' sensors
    pset.renameArguments(ARG0='enemy_distance')
    pset.renameArguments(ARG1='enemy_direction')
    pset.renameArguments(ARG2='health')
    pset.renameArguments(ARG3='ammo')
    pset.renameArguments(ARG4='wall_distance')
    
    # Functions
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)

    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    
    def protected_div(a, b):
        try:
            return a / b
        except ZeroDivisionError:
            return 1
    pset.addPrimitive(protected_div, 2)

    # Ephemeral constants
    pset.addEphemeralConstant('rand_int', functools.partial(random.randint, 0, 10))
    pset.addEphemeralConstant('rand_float', functools.partial(random.uniform, -1, 1))

    # Optional primitives
    def if_then_else(condition, out1, out2):
        return out1 if condition else out2
    if ifelse:
        pset.addPrimitive(if_then_else, 3)

    def greater_than(a, b):
        return 1.0 if a > b else 0.0
    def less_than(a, b):
        return 1.0 if a < b else 0.0
    if logicals:
        pset.addPrimitive(greater_than, 2)
        pset.addPrimitive(less_than, 2)

    if angle_primitives:
        pset.addEphemeralConstant('rand_angle', functools.partial(random.uniform, -3.14, 3.14))
        pset.addPrimitive(math.sin, 1)
        pset.addPrimitive(math.cos, 1)
    
    return pset




def evaluate_individuals(pop1, pop2, toolbox, arena_size=200, max_steps=100):
    """
    Evaluates fitness by pitting individuals from pop1 against individuals from pop2, where
    each robot executes its GP-evolved strategy using its sensors.

    At every time step:
    - Sensor values are updated.
    - Output from GP tree is computed.
    - Output is mapped to one of six possible actions.
    - Robots act (move, shoot, etc.).
    - If a robot's health <= 0, the match ends.

    Fitness is increased by 1 if an individual defeats its opponent, where defeat is considered valid only if
    the opponent dies and the winner survives.
    If both robots survive or both die, no change to fitness.
    """

    # Fitness initialization for new population
    for ind1 in pop1:
        if not ind1.fitness.valid:
            ind1.fitness.values = (0,)
    
    for ind2 in pop2:
        if not ind2.fitness.valid:
            ind2.fitness.values = (0,)
    
    # Competition loop
    for ind1 in pop1:
        for ind2 in pop2:

            # Create robots
            robot1 = Robot(random.uniform(50, arena_size-50), 
                          random.uniform(50, arena_size-50))
            robot2 = Robot(random.uniform(50, arena_size-50), 
                          random.uniform(50, arena_size-50))
            
            # Compile the individual programs
            func1 = toolbox.compile(expr=ind1)
            func2 = toolbox.compile(expr=ind2)
            
            # At each step of the battle:
            for _ in range(max_steps):
                
                # 1) Update sensors
                robot1.update_sensors(robot2, arena_size)
                robot2.update_sensors(robot1, arena_size)
                
                # 2) Get outputs from GP trees
                output1 = func1(
                    robot1.sensors['enemy_distance'],
                    robot1.sensors['enemy_direction'],
                    robot1.sensors['health'],
                    robot1.sensors['ammo'],
                    robot1.sensors['wall_distance']
                )
                output2 = func2(
                    robot2.sensors['enemy_distance'],
                    robot2.sensors['enemy_direction'],
                    robot2.sensors['health'],
                    robot2.sensors['ammo'],
                    robot2.sensors['wall_distance']
                )
                
                # 3) Map output to actions
                action1 = select_action(output1)
                action2 = select_action(output2)
                
                # 4) Execute actions
                robot1.execute_action(action1, robot2, arena_size)
                robot2.execute_action(action2, robot1, arena_size)
                
                # 5) Check if battle is over
                if robot1.health <= 0 or robot2.health <= 0:
                    break
            
            # Fitness update based on battle outcome
            if robot1.health > 0 and robot2.health <= 0:
                ind1.fitness.values = (ind1.fitness.values[0] + 1,)
            elif robot2.health > 0 and robot1.health <= 0:
                ind2.fitness.values = (ind2.fitness.values[0] + 1,)
            



def select_action(output):
    """
    Converts numeric output of the GP tree to one of 6 discrete actions:

    `move_forward`, `turn_left`, `turn_right`, `shoot`, `reload`, `do_nothing`
    """
 
    action_num = int(abs(output)) % 6
    
    actions = [
        'move_forward',
        'turn_left',
        'turn_right',
        'shoot',
        'reload',
        'do_nothing'
    ]
    
    return actions[action_num]




def setup_evolution(pset):
    """
    Configures the evolution parameters with:
    - Fitness and individual types (creator).
    - Population creation tools.
    - Crossover (one-point).
    - Mutation (uniform with subtree replacement).
    - Selection (tournament).
    """

    if not hasattr(creator, 'FitnessMax'):    # To avoid redefining the class each time a new pair of populations is instantiated, which brings up warnings
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()

    # Initialization with ramped half-and-half
    toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)
    
    # One-point crossover
    toolbox.register('mate', gp.cxOnePoint)

    # Uniform mutation w/ subtree replacement (between 0 and 2)
    toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Tournament selection
    toolbox.register('select', tools.selTournament, tournsize=3)
    
    return toolbox




def coevolution(pop_size=30, generations=20, ifelse=False, logicals=False, angle_primitives=False):
    """
    Runs the co-evolutionary algorithm.

    - Create two populations (pop1, pop2) with randomly initialized individuals.
    - Evaluate all individuals using evaluate_individuals().
    - For each generation:
        - Select, clone, and apply crossover/mutation.
        - Re-evaluate the new individuals.
        - Replace the current population.
        - Log average and max fitness for both populations.

    Additional parameters
    - ifelse (default False): adds conditional operator to primitives
    - logicals (default False): adds "<" and ">" operators to primitives
    - angle_primitives (default False): adds sin, cos operators and ephemeral constant (-pi, pi)
    """

    pset = create_primitive_set(ifelse, logicals, angle_primitives)
    toolbox = setup_evolution(pset)
    
    # Populations with initial evaluation
    pop1 = toolbox.population(n=pop_size)
    pop2 = toolbox.population(n=pop_size)
    evaluate_individuals(pop1, pop2, toolbox)

    # For logging/visualizing
    avg_fits1, max_fits1 = [], []
    avg_fits2, max_fits2 = [], []
    
    # Evolutionary loop
    for gen in tqdm(range(generations), unit=' generation'):

        # Select parents with tournament selection (same population size)
        offspring1 = toolbox.select(pop1, len(pop1))
        offspring2 = toolbox.select(pop2, len(pop2))
        
        # Create a deepcopy of the parents to perform crossover and mutation on
        offspring1 = [toolbox.clone(ind) for ind in offspring1]
        offspring2 = [toolbox.clone(ind) for ind in offspring2]
        
        # Crossover with 50% probability between pairs of ordered individuals (e.g. 0&1, 2&3, 4&5 etc.)
        for child1, child2 in zip(offspring1[::2], offspring1[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for child1, child2 in zip(offspring2[::2], offspring2[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation (uniform subtree replacement) with 10% probability
        for mutant in offspring1:
            if random.random() < 0.1:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        for mutant in offspring2:
            if random.random() < 0.1:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the new individuals and replace the old population
        evaluate_individuals(offspring1, offspring2, toolbox)
        pop1[:] = offspring1
        pop2[:] = offspring2
        
        # Fitness
        fits1 = [ind.fitness.values[0] for ind in pop1]
        fits2 = [ind.fitness.values[0] for ind in pop2]

        avg_fits1.append(sum(fits1) / len(fits1))
        max_fits1.append(max(fits1))

        avg_fits2.append(sum(fits2) / len(fits2))
        max_fits2.append(max(fits2))
    
    return pop1, pop2, avg_fits1, max_fits1, avg_fits2, max_fits2




def draw_tree(individual, filename='gp_tree', format='png'):
    """
    Draw and save a GP tree using Graphviz (default format = .png)
    """

    dot = Digraph(format=format)
    
    nodes, edges, labels = gp.graph(individual)
    
    for node_id, label in labels.items():
        dot.node(str(node_id), str(label))
    
    for src, dst in edges:
        dot.edge(str(src), str(dst))
    
    dot.render(filename, cleanup=True)
    print(f'Tree saved as {filename}.{format}')