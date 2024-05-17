import math
from random import Random
import random
from time import time
from inspyred import ec, benchmarks
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Static variables
gridsize = 100
height =10
width = 10

elevations_string = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290,
    120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
    130, 150, 170, 190, 210, 230, 250, 270, 290, 310,
    140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
    150, 170, 190, 210, 230, 250, 270, 290, 310, 330,
    160, 180, 200, 220, 240, 260, 280, 300, 320, 340,
    170, 190, 210, 230, 250, 270, 290, 310, 330, 350,
    180, 200, 220, 240, 260, 280, 300, 320, 340, 360,
    190, 210, 230, 250, 270, 290, 310, 330, 350, 370]

class CityLayout(benchmarks.Benchmark):
  def __init__(self, elevations_string):
        benchmarks.Benchmark.__init__(self, len(elevations_string))
        #self.init_layout = init_layout
        self.elevations_string = elevations_string
        self.bounder = ec.DiscreteBounder([0, 1])
        self.maximize = True
        self.best_all = None
        self.best_feasible = None
  def generator(self, random, args):
        chars = ['R', 'C', 'S', 'G']
        layout = [random.choice(chars) for _ in range(gridsize)]
        return layout

  def evaluator(self, candidates, args):
        fitness = []
        for candidate in candidates:
            fitness.append(CityLayout.calculate_fitness(candidate, self.elevations_string))

        return fitness

  def find_groups(grid):
    def dfs(i, j, group):
      if 0 <= i < height and 0 <= j < width and visited[i][j] == False and grid_matrix[i][j] in ['R']:
        visited[i][j] = True
        group.append((i, j))
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          dfs(i + di, j + dj, group)
      elif 0 <= i < height and 0 <= j < width and grid_matrix[i][j] == 'S':
        group.append(True)


    grid_matrix = [grid[i:i+width] for i in range(0, len(grid), width)]
    visited = [[False] * width for _ in range(height)]
    groups = []

    for i in range(height):
      for j in range(width):
        if visited[i][j] == False and grid_matrix[i][j] in ['R']:
          group = []
          dfs(i, j, group)
          if group:
            groups.append(group)

    return groups

  def find_streets_connected(grid): #searches for groups of R and C
    def dfs(i, j, group):
      if 0 <= i < height and 0 <= j < width and visited[i][j] == False and grid_matrix[i][j] in ['S']:
        visited[i][j] = True
        group.append((i, j))
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
          dfs(i + di, j + dj, group)


    grid_matrix = [grid[i:i+width] for i in range(0, len(grid), width)]
    visited = [[False] * width for _ in range(height)]
    groups = []

    for i in range(height):
      for j in range(width):
        if visited[i][j] == False and grid_matrix[i][j] in ['S']:
          group = []
          dfs(i, j, group)
          if group:
            groups.append(group)

    return groups


  def has_nearby_green(matrix, row, col, max_distance=3):
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(max(row - max_distance, 0), min(row + max_distance + 1, rows)):
        for j in range(max(col - max_distance, 0), min(col + max_distance + 1, cols)):
            if matrix[i][j] == 'G':
                return True

    return False

  def search_for_nearby_green(matrix):
    counter = 0;
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 'R':
                if CityLayout.has_nearby_green(matrix, i, j, 3):
                    counter = counter + 1

    return counter


  def commercial_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      commercial_count = 0
      for i in range(height):
          for j in range(width):
            # Current tile
            tile = layout_matrix[i][j]
            if tile == 'C':
              commercial_count = commercial_count + 1
      commercial_weight = 1;
      if commercial_count < width * height * 0.1:
        commercial_weight = (commercial_count / gridsize) * 5 + 0.5
      elif commercial_count > width * height * 0.2:
        commercial_weight = (commercial_count / gridsize) * (-0.625) + 1.125
      return commercial_weight

  def green_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      green_count = 0
      for i in range(height):
          for j in range(width):
            tile = layout_matrix[i][j]
            if tile == 'G':
              green_count = green_count + 1
      green_weight = 1;
      if green_count < width * height * 0.15:
        green_weight = (green_count / gridsize) * 3.33 + 0.5
      elif green_count > width * height * 0.25:
        green_weight = (green_count / gridsize) * (-0.66) + 1.16

      return green_weight

  def street_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      street_count = 0
      for i in range(height):
          for j in range(width):
            tile = layout_matrix[i][j]
            if tile == 'S':
              street_count = street_count + 1
      street_weight = 1;
      if street_count < width * height * 0.2:
        street_weight = (street_count / gridsize) * 2.5 + 0.5
      elif street_count > width * height * 0.3:
        street_weight = (street_count / gridsize) * (-0.714) + 1.214

      return street_weight

  def res_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      res_count = 0
      for i in range(height):
          for j in range(width):
            tile = layout_matrix[i][j]
            if tile == 'R':
              res_count = res_count + 1
      res_weight = 1;
      if res_count < width * height * 0.25:
        res_weight = (res_count / gridsize) * 2 + 0.5
      elif res_count > width * height * 0.4:
        green_weight = (res_count / gridsize) * (-0.833) + 1.33

      return res_weight

  def res_clusters_weight(layout):
      r_count = 0
      for i in layout:
        if i == 'R':
          r_count += 1
      res_weight = 1
      groups = CityLayout.find_groups(layout)

      groups_ok = 0
      sum = 0
      for i, group in enumerate(groups, 1):
        filtered_list = [item for item in group if item != True]
        num_elements = len(filtered_list)

        if num_elements >= 2 and num_elements <= 10:
          groups_ok += 1

      if groups_ok > len(groups) / 2:
        res_weight *= groups_ok / len(groups)
      else:
        res_weight = 0.5

      return res_weight

  def street_adjacency_weight(layout):
      groups = CityLayout.find_groups(layout)
      street_adj_weight = 1
      count = 0
      for group in groups:
        if True in group:
          count += 1
      if len(groups) != 0:
       percentage = count / len(groups)
      else:
        percentage = 0
      street_adj_weight = 0.5 * percentage + 0.5

      return street_adj_weight

  def nearby_green_weight(layout):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      counter = CityLayout.search_for_nearby_green(layout_matrix)
      r_count = 0
      for row in layout_matrix:
        for tile in row:
            if tile == 'R':
                r_count += 1
      if r_count != 0:
        nearby_green_weight = counter / r_count
      else:
        nearby_green_weight = 0.5

      return nearby_green_weight

  def street_connectivity_weight(layout):
      groups = CityLayout.find_streets_connected(layout)
      street_connectivity = 0
      for g in groups:
        if len(g) != 0:
          street_connectivity += 1/(len(g)**2)

      s_count = 0
      for i in layout:
        if i == 'S':
          s_count += 1

      if s_count != 1:
        return street_connectivity*(s_count**2)/(1-s_count**3) - s_count**3/(1-s_count**3)
      else:
        return 0.5

  def elev_weight_normal(layout, elev_grid):
      elevations_matrix = [elev_grid[i:i+width] for i in range(0, len(elev_grid), width)]
      layout_matrix = [layout[i:i+width] for i in range(0, len(layout), width)]
      highest_point = 0
      lowest_point = 1500

      for row in elevations_matrix:
        for value in row:
          if value > highest_point:
            highest_point = value
          elif value < lowest_point:
            lowest_point = value
      elev_weight = 1
      for i in range(width):
        for j in range(height):
            tyle = layout_matrix[i][j]
            if tyle == 'R' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight += elevations_matrix[i][j] * (1.5 / ((-2)*highest_point - lowest_point)) + 0.5 + highest_point*(1.5 / (2 * highest_point + lowest_point))
            elif tyle == 'C' and elevations_matrix[i][j] > (highest_point - lowest_point) / 5: #385
              elev_weight += elevations_matrix[i][j] * (2.5 / ((-4)*highest_point - lowest_point)) + 0.5 + highest_point*(2.5 / (4 * highest_point + lowest_point))
            elif tyle == 'G' and elevations_matrix[i][j] > (highest_point - lowest_point) / 3: #642
              elev_weight += elevations_matrix[i][j] * (1.5 / (2 * highest_point + lowest_point)) + 1 - 0.5 * (highest_point - lowest_point) / (2 * highest_point + lowest_point)
            else:
              elev_weight += 1
      return elev_weight/gridsize

  def calculate_fitness(layout, elev_grid):
      layout_matrix = [layout[i:i+width] for i in range(0, gridsize, width)]
      elevations_matrix = [elev_grid[i:i+width] for i in range(0, len(elev_grid), width)]
      fitness = 0

      fitness = fitness + 10 * CityLayout.elev_weight_normal(layout, elev_grid)

      fitness = fitness + 10 * CityLayout.commercial_weight(layout)

      fitness = fitness + 10 * CityLayout.green_weight(layout)

      fitness = fitness + 10 * CityLayout.res_weight(layout)

      fitness = fitness + 10 * CityLayout.street_weight(layout)

      fitness = fitness + 10 * CityLayout.res_clusters_weight(layout)

      fitness = fitness + 10 * CityLayout.street_adjacency_weight(layout)

      fitness = fitness + 10 * CityLayout.nearby_green_weight(layout)

      fitness = fitness + 10 * CityLayout.street_connectivity_weight(layout)

      return fitness

def distance(x, y):
  count = 0
  for item1, item2 in zip(x, y):
    if item1 != item2:
      count += 1

  return count

def test(problem, chosen_evaluator, selector, variator1, variator2, replacer, terminator, nr_gen, args):
  seed = time()
  prng = Random()
  prng.seed(seed)
  def history_observer(population, num_generations, num_evaluations, args):
        best = max(population).candidate
        solution_history.append(best)
        commercial_weight_history.append(CityLayout.commercial_weight(best))
        green_weight_history.append(CityLayout.green_weight(best))
        res_weight_history.append(CityLayout.res_weight(best))
        street_weight_history.append(CityLayout.street_weight(best))
        res_clusters_weight_history.append(CityLayout.res_clusters_weight(best))
        street_adjacency_history.append(CityLayout.street_adjacency_weight(best))
        nearby_green_weight_history.append(CityLayout.nearby_green_weight(best))
        street_connectivity_history.append(CityLayout.street_connectivity_weight(best))
        elev_weight_overall_history.append(CityLayout.elev_weight_normal(best, elevations_string))

  def diversity(population):
    value_map = {'R': 0, 'S': 1, 'C': 2, 'G': 3}
    numerical_population = [[value_map[val] for val in individual.candidate] for individual in population]

    return np.array([i for i in numerical_population]).std(axis=0).mean()

  def fitness_diversity_observer(population, num_generations, num_evaluations, args):
    best = max(population).fitness
    div = diversity(population)

    best_fitness_historic.append(best)
    diversity_historic.append(div)

  best_fitness_historic = []
  diversity_historic = []

  commercial_weight_history = []
  green_weight_history = []
  res_weight_history = []
  street_weight_history = []
  res_clusters_weight_history = []
  street_adjacency_history = []
  nearby_green_weight_history = []
  street_connectivity_history = []
  elev_weight_overall_history = []
  solution_history = []

  ga = ec.EvolutionaryComputation(prng)
  ga.selector = selector
  ga.variator = [variator1, variator2]
  ga.terminator = terminator
  ga.replacer = replacer
  ga.observer = [history_observer, fitness_diversity_observer,]

  final_pop = ga.evolve(generator=problem.generator,
                          evaluator=chosen_evaluator,
                          pop_size=200,
                          bounder = ['R', 'S', 'G', 'C'],
                          maximize=problem.maximize,
                          max_evaluations=1,
                          max_generations=nr_gen,
                          num_elites=1,
                          distance_function = distance,
                          **args)


  best = max(ga.population)

  color_map = {'C': 'blue', 'R': 'brown', 'S': 'grey', 'G': 'green'}
  color_array = [color_map[char] for char in best.candidate]

  fig, ax = plt.subplots(figsize=(10, 10))
  for i in range(width):
    for j in range(height):
        index = i * width + j  # Calculate the index in the flat list
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_array[index]))

  ax.set_xlim(0, width)
  ax.set_ylim(0, height)
  ax.invert_yaxis()  # Correct the display orientation
  ax.axis('off')  # Hide grid lines and numbers
  plt.show()

  return final_pop, solution_history, commercial_weight_history, green_weight_history, res_weight_history, street_weight_history, res_clusters_weight_history, street_adjacency_history, nearby_green_weight_history, street_connectivity_history, elev_weight_overall_history, best_fitness_historic, diversity_historic


problem = CityLayout(elevations_string)

final_pop, solution_history, commercial_weight_history, green_weight_history, res_weight_history, street_weight_history, res_clusters_weight_history, street_adjacency_history, nearby_green_history, street_connectivity_history, elev_weight_overall_history, best_fitness_historic, diversity_historic = test(problem, problem.evaluator,ec.selectors.uniform_selection, ec.variators.n_point_crossover, ec.variators.random_reset_mutation, ec.replacers.crowding_replacement, ec.terminators.no_improvement_termination, 200, {'tournament_size': 3, 'num_points_crossver': width, 'num_selected': 200})

final_pop2, solution_history2, commercial_weight_history2, green_weight_history2, res_weight_history2, street_weight_history2, res_clusters_weight_history2, street_adjacency_history2, nearby_green_history2, street_connectivity_history2, elev_weight_overall_history2, best_fitness_historic2, diversity_historic2 = test(problem, problem.evaluator,ec.selectors.tournament_selection, ec.variators.n_point_crossover, ec.variators.random_reset_mutation, ec.replacers.crowding_replacement, ec.terminators.no_improvement_termination, 200, {'tournament_size': 3, 'num_points_crossver': width, 'num_selected': 200})

final_pop3, solution_history3, commercial_weight_history3, green_weight_history3, res_weight_history3, street_weight_history3, res_clusters_weight_history3, street_adjacency_history3, nearby_green_history3, street_connectivity_history3, elev_weight_overall_history3, best_fitness_historic3, diversity_historic3 = test(problem, problem.evaluator,ec.selectors.fitness_proportionate_selection, ec.variators.n_point_crossover, ec.variators.random_reset_mutation, ec.replacers.crowding_replacement, ec.terminators.no_improvement_termination, 200, {'tournament_size': 3, 'num_points_crossver': width, 'num_selected': 200})

final_pop4, solution_history4, commercial_weight_history4, green_weight_history4, res_weight_history4, street_weight_history4, res_clusters_weight_history4, street_adjacency_history4, nearby_green_history4, street_connectivity_history4, elev_weight_overall_history4, best_fitness_historic4, diversity_historic4 = test(problem, problem.evaluator,ec.selectors.default_selection, ec.variators.n_point_crossover, ec.variators.random_reset_mutation, ec.replacers.crowding_replacement, ec.terminators.no_improvement_termination, 200, {'tournament_size': 3, 'num_points_crossver': width, 'num_selected': 200})

final_pop5, solution_history5, commercial_weight_history5, green_weight_history5, res_weight_history5, street_weight_history5, res_clusters_weight_history5, street_adjacency_history5, nearby_green_history5, street_connectivity_history5, elev_weight_overall_history5, best_fitness_historic5, diversity_historic5 = test(problem, problem.evaluator,ec.selectors.truncation_selection, ec.variators.n_point_crossover, ec.variators.random_reset_mutation, ec.replacers.crowding_replacement, ec.terminators.no_improvement_termination, 200, {'tournament_size': 3, 'num_points_crossver': width, 'num_selected': 200})


# Example elevation data (replace with your actual elevation array)
elevations_string = [445, 242, 273, 280, 224, 239, 484,
465, 424, 382, 430, 363, 227, 230,
471, 364, 433, 227, 234, 402, 331,
311, 404, 255, 495, 373, 368, 414,
360, 219, 261, 239, 332, 466, 484,
398, 488, 229, 416, 396, 219, 347,
430, 409, 452, 418, 485, 239, 438]

elevations = np.reshape(elevations_string, (height, width))

# Plot the elevation matrix as a heatmap
plt.imshow(elevations, cmap='terrain', interpolation='nearest')
plt.colorbar(label='Elevation')
plt.title('Elevation Map')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()