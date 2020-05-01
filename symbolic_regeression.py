import pandas as pd

# Read in data
data = pd.read_csv(
    "auto-mpg.data", delim_whitespace=True, na_values="?", header=None,
    names=["mpg", "cylinders", "displacement", "horsepower", "weight",
           "acceleration", "model year", "origin", "car name", ])
# Drop string feature
data.pop("car name")
# Replace N/A values in horsepower
data["horsepower"].fillna(data["horsepower"].median(), inplace=True)
# Separate target feature
target = data.pop("mpg")

import operator

val1 = {"feature_name": "horsepower"}
val2 = {"feature_name": "cylinders"}
val3 = {"feature_name": "weight"}
node1 = {
    "func": operator.add,
    "children": [val1, val2],
    "format_str": "({} + {})",
}
program = {
    "func": operator.mul,
    "children": [node1, val3],
    "format_str": "({} * {})",
}

def render_prog(node):
    if "children" not in node:
        return node["feature_name"]
    return node["format_str"].format(*[render_prog(c) for c in node["children"]])

print(render_prog(program))

def evaluate(node, row):
    if "children" not in node:
        return row[node["feature_name"]]
    return node["func"](*[evaluate(c, row) for c in node["children"]])

print(evaluate(program, data.iloc[0]))

def safe_div(a, b):
    return a / b if b else a

operations = (
    {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
    {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
    {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
    {"func": safe_div, "arg_count": 2, "format_str": "({} / {})"},
    {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
)

from random import randint, random, seed

seed(0)

def random_prog(depth):
    # favor adding function nodes near the tree root and
    # leaf nodes as depth increases
    if randint(0, 10) >= depth * 2:
        op = operations[randint(0, len(operations) - 1)]
        return {
            "func": op["func"],
            "children": [random_prog(depth + 1) for _ in range(op["arg_count"])],
            "format_str": op["format_str"],
        }
    else:
        return {"feature_name": data.columns[randint(0, data.shape[1] - 1)]}


POP_SIZE = 300
population = [random_prog(0) for _ in range(POP_SIZE)]

print(render_prog(population[0]))

def select_random_node(selected, parent, depth):
    if "children" not in selected:
        return parent
    # favor nodes near the root
    if randint(0, 10) < 2*depth:
        return selected
    child_count = len(selected["children"])
    return select_random_node(
        selected["children"][randint(0, child_count - 1)],
        selected, depth+1)

print(render_prog(select_random_node(program, None, 0)))

from copy import deepcopy

def do_mutate(selected):
    offspring = deepcopy(selected)
    mutate_point = select_random_node(offspring, None, 0)
    child_count = len(mutate_point["children"])
    mutate_point["children"][randint(0, child_count - 1)] = random_prog(0)
    return offspring


print(render_prog(do_mutate(program)))

def do_xover(selected1, selected2):
    offspring = deepcopy(selected1)
    xover_point1 = select_random_node(offspring, None, 0)
    xover_point2 = select_random_node(selected2, None, 0)
    child_count = len(xover_point1["children"])
    xover_point1["children"][randint(0, child_count - 1)] = xover_point2
    return offspring


print(render_prog(do_xover(population[0], population[1])))

TOURNAMENT_SIZE = 3

def get_random_parent(population, fitness):
    # randomly select population members for the tournament
    tournament_members = [
        randint(0, POP_SIZE - 1) for _ in range(TOURNAMENT_SIZE)]
    # select tournament member with best fitness
    member_fitness = [(fitness[i], population[i]) for i in tournament_members]
    return min(member_fitness, key=lambda x: x[0])[1]

XOVER_PCT = 0.7

def get_offspring(population, fitness):
    parent1 = get_random_parent(population, fitness)
    if random() > XOVER_PCT:
        parent2 = get_random_parent(population, fitness)
        return do_xover(parent1, parent2)
    else:
        return do_mutate(parent1)

REG_STRENGTH = 0.5

def node_count(x):
    if "children" not in x:
        return 1
    return sum([node_count(c) for c in x["children"]])


def compute_fitness(program, prediction):
    mse = ((pd.Series(prediction) - target) ** 2).mean()
    penalty = node_count(program) ** REG_STRENGTH
    return mse * penalty

MAX_GENERATIONS = 10

global_best = float("inf")
for gen in range(MAX_GENERATIONS):
    fitness = []
    for prog in population:
        prediction = [
            evaluate(prog, row) for _, row in data.iterrows()]
        score = compute_fitness(prog, prediction)
        fitness.append(score)
        if score < global_best:
            global_best = score
            best_pred = prediction
            best_prog = prog
    print(
        "Generation: %d\nBest Score: %.2f\nMedian score: %.2f\nBest program: %s\n"
        % (
            gen,
            global_best,
            pd.Series(fitness).median(),
            render_prog(best_prog),
        )
    )
    population = [
        get_offspring(population, fitness)
        for _ in range(POP_SIZE)]

print("Best score: %f" % global_best)
print("Best program: %s" % render_prog(best_prog))
output = {"target": target, "pred": best_pred}
pd.DataFrame(output).to_csv("best_pred.csv")
