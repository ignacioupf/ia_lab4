# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveExpectimaxAgent', second='DefensiveExpectimaxAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

#
# Agente abstracto
# (Usa algoritmo Expectimax)
#
class ExpectimaxAgent(CaptureAgent):

    #
    # Inicilización
    #

    # Constuctor
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # Posición inicial del agente.
        self.start = None

        # Profundidad del algoritmo Expectimax
        self.depth = 3

        # Opción que indica que el algoritmo Expectimax no considera
        # que el agente MAX pueda estar estar parado.
        # (Puede variar con la estrategia a seguir.)
        self.skip_stop = False

    # Se ejecuta al inicio de cada partida.
    def register_initial_state(self, game_state):
        # Posición inicial del agente
        self.start = game_state.get_agent_position(self.index)

        CaptureAgent.register_initial_state(self, game_state)

        # Usamos distancias de maze y no distancias Manhattan.
        # (Se comenta porque ya ha sido llamado en la superclase.)
        # self.distancer.get_maze_distances()

        # Índices de los oponentes
        self.opponent_indices = self.get_opponents(game_state)

    #
    # Acción elegida
    #

    def choose_action(self, game_state):
        # Seleccionamos estrategia a seguir.
        # (Influye en la función de evaluación).
        self.select_strategy(game_state)
    
        # Llamamos al algoritmo Expectimax
        return self.expectimax(game_state)

    def select_strategy(self, game_state):
        self.analize_state(game_state)
        self.default_preferences(game_state)
        self.select_preferences(game_state)
        self.default_goals(game_state)
        self.select_goals(game_state)

    def analize_state(self, game_state):
        self.set_boundary(game_state)
        self.set_subboundary(game_state)
        self.set_patrol_focuses(game_state)

    def default_preferences(self, game_state):
        self.skip_stop = False
        self.extra_distance = 1
        self.radius = 5

    def select_preferences(self, game_state):
        pass

    def default_goals(self, game_state):
        self.goals = {}
        
    def select_goals(self, game_state):
        pass

    #
    # Algoritmo Expectimax
    #

    def expectimax(self, game_state):
        value, action = self.max_node(game_state, 1)
        return action
    
    # Devuelve valor y acción del nodo MAX.
    def max_node(self, game_state, depth):
        # Si es un nodo terminal, devolver función de evaluación.
        if self.is_terminal_node(game_state) or depth == self.depth:
            return self.evaluation_function(game_state), Directions.STOP
        
        # Acciones posibles
        actions = game_state.get_legal_actions(self.index)
        if self.skip_stop:
            actions.remove(Directions.STOP)

        # Valores de los hijos
        values = []
        for action in actions:
            successor = game_state.generate_successor(self.index, action)
            # successor = self.get_successor(game_state, action) ###################
            value = self.chance_node(successor, 0, depth)
            values.append(value)
        
        # Devolver mejor valor y acción asocidada.
        return self.best_pair(values, actions)

    # Devuelve valor del nodo CHANCE.
    def chance_node(self, game_state, opponent, depth):
        if self.is_terminal_node(game_state):
            return self.evaluation_function(game_state)
        
        # Datos de los oponentes
        opponent_index = self.opponent_indices[opponent]
        next_opponent = (opponent + 1) % len(self.opponent_indices)

        # Si el oponente no es observable, hacemos un bypass.
        if self.is_not_observable(game_state, opponent_index):
            successor = game_state
            value = self.child_of_chance_node(successor, next_opponent, depth)
            return value

        # Acciones posibles
        actions = game_state.get_legal_actions(opponent_index)

        # Valores de los hijos.
        values = []
        for action in actions:
            successor = game_state.generate_successor(opponent_index, action)
            value = self.child_of_chance_node(successor, next_opponent, depth)
            values.append(value)

        # Devolvemos esperanza de los valores.
        return self.expectation(values)

    # Devuelve valor del hijo del nodo CHANCE.
    def child_of_chance_node(self, successor, next_opponent, depth):
        if next_opponent == 0:
            value, _ = self.max_node(successor, depth + 1)
        else:
            value = self.chance_node(successor, next_opponent, depth)
        return value

    # Comprueba si el nodo es TERMINAL.
    def is_terminal_node(self, game_state):
        return game_state.is_over()

    # Función de evaluación de un nodo TERMINAL.
    def evaluation_function(self, game_state):
        value = 0
        for name, weight in self.goals.items():
            feature = self.get_feature(game_state, name)
            value += feature * weight
            #print("feature", name, weight, feature)
        return value

    # Comprueba si el oponente es observable.
    def is_not_observable(self, game_state, opponent_index):
        position = game_state.get_agent_state(opponent_index).get_position()
        return position is None

    # Obtener uno de las mejores pares valor-acción
    def best_pair(self, values, actions):
        # Obtener lista de mejores pares valor-acción.
        # (Son varios en caso de empate.)
        max_value = max(values)
        best = []
        for value, action in zip(values, actions):
            if value == max_value:
                best.append((value, action))
        
        # Retornar una de los mejores pares valor-acción.
        # (Los empates se resueven al azar.)
        return random.choice(best)

    # Esperanza de una serie de valores.
    def expectation(self, values):
        return sum(values) / len(values)

    #
    # Distancias
    #

    def distance(self, game_state, destiny):
        agent_position = game_state.get_agent_position(self.index)
        return self.get_maze_distance(agent_position, destiny)

    def distance_to_start(self, game_state):
        return self.distance(game_state, self.start)

    def min_distance(self, game_state, positions, extra=0):
        agent_position = game_state.get_agent_position(self.index)
        distances = []
        for position in positions:
            distance = self.get_maze_distance(agent_position, position)
            distances.append(distance)
        if len(distances) == 0:
            return 0
        return min(distances) + extra

    def min_distance_observable_pacmans(self, game_state):
        agents = [game_state.get_agent_state(i) for i in self.opponent_indices]
        positions = [a.get_position() for a in agents if a.get_position() != None and a.is_pacman == True]
        return self.min_distance(game_state, positions, self.extra_distance)

    def min_distance_observable_ghosts(self, game_state):
        agents = [game_state.get_agent_state(i) for i in self.opponent_indices]
        positions = [a.get_position() for a in agents if a.get_position() != None and a.is_pacman == False]
        return self.min_distance(game_state, positions, self.extra_distance)

    def min_distance_observable_daring_ghosts(self, game_state):
        agents = [game_state.get_agent_state(i) for i in self.opponent_indices]
        positions = [a.get_position() for a in agents if a.get_position() != None and a.is_pacman == False and a.scared_timer <= 0]
        return self.min_distance(game_state, positions, self.extra_distance)

    def min_distance_observable_scared_ghosts(self, game_state):
        agents = [game_state.get_agent_state(i) for i in self.opponent_indices]
        positions = [a.get_position() for a in agents if a.get_position() != None and a.is_pacman == False and a.scared_timer > 0]
        return self.min_distance(game_state, positions, self.extra_distance)


#
#    def min_distance_observables(self, game_state, filtrum):
#        agents = list(filter(filtrum, self.observable_agents(game_state)))
#        positions = list(map(lambda a: a.get_position(), agents))
#        return self.min_distance(game_state, positions, self.extra_distance)
#
#    def min_distance_observable_all(self, game_state):
#        filtrum = lambda a: True
#        return self.min_distance_observables(game_state, filtrum)
#
#    def min_distance_observable_pacmans(self, game_state):
#        filtrum = lambda a: a.is_pacman
#        return self.min_distance_observables(game_state, filtrum)
#
#    def min_distance_observable_ghosts(self, game_state):
#        filtrum = lambda a: a.is_pacman == False
#        return self.min_distance_observables(game_state, filtrum)
#
#    def min_distance_observable_daring_ghosts(self, game_state):
#        filtrum = lambda a: (a.is_pacman == False) and (a.scared_timer <= 0)
#        return self.min_distance_observables(game_state, filtrum)
#
#    def min_distance_observable_scared_ghosts(self, game_state):
#        filtrum = lambda a: (a.is_pacman == False) and (a.scared_timer > 0)
#        return self.min_distance_observables(game_state, filtrum)

    def min_distance_food(self, game_state):
        positions = self.get_food(game_state).as_list()
        return self.min_distance(game_state, positions)

    def min_distance_capsules(self, game_state):
        positions = self.get_capsules(game_state)
        return self.min_distance(game_state, positions)

    def min_distance_cocos(self, game_state):
        positions = self.get_food(game_state).as_list() + self.get_capsules(game_state)
        return self.min_distance(game_state, positions)

    def min_distance_own_capsules(self, game_state):
        positions = self.get_capsules_you_are_defending(game_state)
        return self.min_distance(game_state, positions, 1)

    def min_distance_patrol_focuses(self, game_state):
        positions = self.patrol_focuses
        return self.min_distance(game_state, positions)

    def min_distance_boundary(self, game_state):
        positions = self.boundary
        return self.min_distance(game_state, positions)

    def min_distance_subboundary(self, game_state):
        positions = self.subboundary
        return self.min_distance(game_state, positions)

    #
    # Cantidades
    #

    def num_own_capsules(self, game_state):
        return len(self.get_capsules_you_are_defending(game_state))

    def num_food_carrying(self, game_state):
        return game_state.get_agent_state(self.index).num_carrying

    def num_food(self, game_state):
        return len(self.get_food(game_state).as_list())

    def num_capsules(self, game_state):
        return len(self.get_capsules(game_state))
        
    def num_cocos(self, game_state):
        return len(self.get_food(game_state).as_list() + self.get_capsules(game_state))

    def clap(self, distance):
        if distance > self.radius:
            return 0
        return distance

    #
    # Posiciones
    #

    def set_boundary(self, game_state):
        x = self.get_boundary_x(game_state)
        ys = list(range(game_state.data.layout.height))
        self.boundary = []
        for y in ys:
            if not game_state.has_wall(x, y):
             self.boundary.append((x, y))

    def get_boundary_x(self, game_state):
        x = game_state.data.layout.width // 2
        if self.red:
            x = x - 1
        return x

    def set_subboundary(self, game_state):
        x = self.get_subboundary_x(game_state)
        ys = list(range(game_state.data.layout.height))
        self.subboundary = []
        for y in ys:
            if not game_state.has_wall(x, y):
             self.subboundary.append((x, y))

    def get_subboundary_x(self, game_state):
        x = game_state.data.layout.width // 2 + 1
        if self.red:
            x = x - 3
        return x

    # Establece los focos de patrulla.
    def set_patrol_focuses(self, game_state):
        # Al principio, los focos de patrulla son las cápsulas.
        # Después, los focos de patrulla son las últimas comidas
        # consumidas por los oponentes.
        if self.first_turn(game_state):
            self.patrol_focuses = self.get_capsules_you_are_defending(game_state)
        else:
            positions = self.positions_food_modified(self)
            if len(positions) > 0:
                self.patrol_focuses = positions

    # Averigua si se ha modificado la comida que se defiende.
    def positions_food_modified(self, game_state):
        if self.first_turn(game_state):
            return []
        previous = self.get_food_you_are_defending(self.get_previous_observation()).as_list()
        current = self.get_food_you_are_defending(self.get_current_observation()).as_list()
        return self.list_difference(previous, current)

    # Calcula la diferencia de dos listas.
    def list_difference(self, list_a, list_b):
        list_c = []
        for a in list_a:
            if a not in list_b:
                list_c.append(a)
        return list_c

    #
    # Info oponentes
    #

    def observable_agents(self, game_state):
        agents = self.opponent_agents(game_state)
        filtrum = lambda a: a.get_position() != None
        observable = list(filter(filtrum, agents))
        #print("obs", observable)
        return observable

    def opponent_agents(self, game_state):
        agents = []
        for index in self.opponent_indices:
            agents.append(game_state.get_agent_state(index))
        return agents

    def opponent_timers(self, game_state):
        agents = self.opponent_agents(game_state)
        return list(map(lambda a: a.scared_timer, agents))

    def print_opponents_info(self, game_state):
        for index in self.opponent_indices:
            agent = game_state.get_agent_state(index)
            print(" is_pacman", agent.is_pacman, "position", agent.get_position(), "timer", agent.scared_timer, end="")
        print()

    #
    # Condiciones
    #
    
    def first_turn(self, game_state):
        return self.get_previous_observation() is None

    def is_fed(self, game_state):
        return self.num_food_carrying(game_state) >= self.capacity

    def is_scared(self, game_state):
        agent = game_state.get_agent_state(self.index)
        return agent.scared_timer > 0

    def are_some_opponents_scared(self, game_state):
        return max(self.opponent_timers(game_state)) > 0

    def are_all_opponents_scared(self, game_state):
        return min(self.opponent_timers(games_state)) > 0

    def has_collided(self, game_state):
        return 0
        
    def is_at_home(self, game_state):
        agent = game_state.get_agent_state(self.index)
        return not agent.is_pacman

    #
    # Objetivos
    #
    
    def get_feature(self, game_state, name):

        if name == "eat_capsules":
            return -(self.min_distance_capsules(game_state) + 10000 * self.num_capsules(game_state))

        if name == "eat_cocos":
            return -(self.min_distance_cocos(game_state) + 100 * self.num_food(game_state) + 10000 * self.num_capsules(game_state))
            #return -(self.min_distance_cocos(game_state) + 100 * self.num_cocos(game_state))

        if name == "eat_food":
            return -(self.min_distance_food(game_state) + 100 * self.num_food(game_state))
        
        if name == "flee_daring_ghosts":
            return self.clap(self.min_distance_observable_daring_ghosts(game_state))

        if name == "flee_ghosts":
            return self.clap(self.min_distance_observable_ghosts(game_state))

        if name == "flee_pacmans":
            return self.clap(self.min_distance_observable_pacmans(game_state))

        if name == "go_boundary":
            return -self.min_distance_boundary(game_state)

        if name == "go_start":
            return -self.distance_to_start(game_state)

        if name == "go_subboundary":
            return -self.min_distance_subboundary(game_state)

        if name == "hunt_ghosts":
            return -(self.min_distance_observable_ghosts(game_state) + 1000 * self.has_collided(game_state))

        if name == "hunt_scared_ghosts":
            return -self.min_distance_observable_scared_ghosts(game_state)

        if name == "hunt_pacmans":
            return -self.min_distance_observable_pacmans(game_state)

        if name == "patrol":
            return -self.min_distance_patrol_focuses(game_state)

        if name == "protect":
            return -self.min_distance_own_capsules(game_state)

        if name == "stay_at_home":
            return 1 if self.is_at_home(game_state) else -1

        print("Erroneus feature:", name)
        return 0


#
# Agente ofensivo
#
class OffensiveExpectimaxAgent(ExpectimaxAgent):

    def select_preferences(self, game_state):
        self.skip_stop = True
        self.extra_distance = 1
        self.radius = 5
        self.capacity = 2

    def select_goals(self, game_state):
        if not self.is_fed(game_state):
            self.goals = {"eat_cocos": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}
        else:
            self.goals = {"go_subboundary": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}


    def select_goals_v0(self, game_state):
        if self.are_all_opponents_scared(game_state):
            if not self.is_fed(game_state):
                #self.goals = {"eat_cocos": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}
                self.goals = {"eat_cocos": 1, "hunt_scared_ghosts": 100}
                #self.goals = {"eat_capsules": 1, "flee_ghosts": 100}
            else:
                #self.goals = {"go_boundary": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}
                self.goals = {"go_subboundary": 1, "hunt_scared_ghosts": 100}
 
        else:
            if not self.is_fed(game_state):
                #self.goals = {"eat_cocos": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}
                self.goals = {"eat_cocos": 1, "flee_daring_ghosts": 100}
                #self.goals = {"eat_capsules": 1, "flee_ghosts": 100}
            else:
                #self.goals = {"go_boundary": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}
                self.goals = {"go_subboundary": 1, "flee_daring_ghosts": 100}
        #print(self.goals)

#
# Agente defensivo
#
class DefensiveExpectimaxAgent(ExpectimaxAgent):

    def select_preferences(self, game_state):
        self.skip_stop = True
        self.extra_distance = 1
        self.radius = 5
        self.capacity = 0

    def select_goals(self, game_state):
        if not self.is_scared(game_state):
            self.goals = {"patrol": 1, "hunt_pacmans": 100, "stay_at_home": 10000}
        else:
            self.goals = {"patrol": 1, "flee_pacmans": 100, "stay_at_home": 10000}

#
