# my_team.py
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


# my_team.py
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

import time

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveExpectiminimaxAgent', second='DefensiveExpectiminimaxAgent', num_training=0):
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
# (Usa algoritmos EXPECTIMINIMAX, MINIMAX y A*)
#
class ExpectiminimaxAgent(CaptureAgent):

    #
    # Inicilización
    #

    # Constuctor
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # Posición inicial de agente.
        self.start = None

        # Profundidad del algoritmo EXPECTIMINIMAX
        self.depth = 2

        # Opción que indica que el algoritmo EXPECTIMINIMAX no considera
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
        
        # Guarida de los oponentes
        self.opponent_start = game_state.get_agent_position(self.opponent_indices[0])

        # Frontera
        self.set_boundary(game_state)
        self.set_subboundary(game_state)

    #
    # Acción elegida
    #

    # Elegir acción.
    def choose_action(self, game_state):
        start_time = time.time()
    
        # Seleccionamos estrategia a seguir.
        # (Influye en la función de evaluación).
        self.select_strategy(game_state)
    
        # Llamamos al algoritmo EXPECTIMINIMAX.
        action = self.expectiminimax(game_state)

        # Comprobamos que no hayamos excedidor el tiempo límite.
        self.check_time(start_time)
        
        return action

    def check_time(self, start_time):
        end_time = time.time()
        duration = end_time - start_time

        #if duration >= 0.3:
        #    print("duration", duration, self.index, self.role)

        if duration >= 0.9:
            self.depth = 1

    # Seleccionar estrategia a seguir
    def select_strategy(self, game_state):
        # Establecer preferencias
        self.default_preferences(game_state)
        self.select_preferences(game_state)
        
        # Analizar estado del juego
        self.analize_state(game_state)
        
        # Establecer objetivos
        self.default_goals(game_state)
        self.select_goals(game_state)

    # Preferencias por defecto.
    def default_preferences(self, game_state):
        # Verdadero si el agente MAX no para nunca.
        self.skip_stop = False
        # Agresividad de los oponentes:
        #    1.0 -> Oponentes agresivos -> Se usa algorimo MINIMAX
        #    0.0 -> Openentes aleatorios -> Se usa algortimo EXPECTIMINIMAX
        #    Entre 0.0 y 1.0 -> Oponentes semiagresivos -> Se usa una mezcla de MINIMAX y EXPECTIMINIMAX
        self.opponent_aggressivity = 0.5
        # Verdadero si el foco de patrulla inicial es la comida interesante (la más cercana para el oponente).
        # Falso si el foco de patrulla inicial son la posiciones de las cápsulas.
        self.patrol_interesting = True
        # Verdadero si se calculan las rutas que no tienen obstáculos.
        self.calculate_unobstructive = False
        # Distancia extra que se suma a la distancia de los oponentes.
        self.extra_distance = 0
        # Distancia de laberinto a partir de la cual no se consideran los oponentes.
        self.far_distance = 5
        # Cantidad máxima de comida a cargar.
        self.capacity = 0

    # Preferencias de la subclase.
    # (Método a sobreescribir por la subclase.)
    def select_preferences(self, game_state):
        pass

    # Analizar estado.
    def analize_state(self, game_state):
        # Estabelcer los focos de patrulla.
        self.set_patrol_focuses(game_state)
        # Establecer que cocos tienen una ruta sin obstáculos (oponentes observables).
        self.set_unobstructive_cocos(game_state)

    # Objetivos por defecto.
    def default_goals(self, game_state):
        # No hay objetivos por defecto.
        self.goals = {}
    
    # Objetivos de la subcalse.
    # (Método a sobreescribir por la subclase.)
    def select_goals(self, game_state):
        pass

    #
    # Algoritmo EXPECTIMINIMAX
    #

    # Se llama al algoritmo EXPECTIMINIMAX y se devuelve la mejor acción.
    def expectiminimax(self, game_state):
        value, action = self.max_node(game_state, 0)
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

        # Valores de los hijos
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
    # La esperanza puede dar más peso al valor mínimo:
    #   *  Importancia 0.0 -> EXPECTIMINIMAX
    #   *  Importancia 1.0 -> MINIMAX
    #   *  Importancia entre 0.0 y 1.0 -> Mezcla de EXPECTIMINIMAX y MINIMAX
    def expectation(self, values):
        average = sum(values) / len(values)
        minimum = min(values)
        importance = self.opponent_aggressivity
        return importance * minimum + (1.0 - importance) * average

    #
    # Algoritmo A*
    #
    
    # Algortimo A*.
    # Busca la ruta más corta de origen a diana que no tenga obstáculos.
    def a_star(self, game_state, origin, target, obstacles=[]):
        # Nodo del algoritmo A*
        class AStarNode:
            # Construtor: posición, padre del nodo, coste del último paso.
            def __init__(self, position, parent, cost):
                x = int(position[0])
                y = int(position[1])
                self.position = (x, y)
                self.parent = parent
                self.cost = cost if parent is None else cost + parent.cost

            # Obtener ruta
            def get_path(self):
                path = []
                current_node = self
                while current_node.parent is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                path.reverse()
                return path

            # Comprobación de igualdad entre nodos.
            # Dos nodos son iguales si tienen la misma posición.
            def __eq__(self, other) -> bool:
                if (type(other) is AStarNode):
                    return self.position == other.position
                return False

            # Hash del nodo.
            # Se calcula el hash basado solamente en la posición.
            def __hash__(self) -> int:
                return hash(self.position)

        # Acciones posibles
        actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Estructuras de datos
        expanded_nodes = set()
        frontier = util.PriorityQueue()
        
        # Nodo origen
        origin_node = AStarNode(origin, None, 0)
        origin_priority = 0 + self.get_maze_distance(origin, target)
        frontier.push(origin_node, origin_priority)
        
        # Mientras haya nodos en la frontera...
        while not frontier.is_empty():
            # Expandir nodo.
            node = frontier.pop()
            expanded_nodes.add(node)
            
            # Comprobar si hemos llegado al final.
            if node.position == target:
                return node.get_path()
        
            # Por cada acción...
            for action in actions:
                # Obtener posición del nodo hijo
                x = node.position[0] + action[0]
                y = node.position[1] + action[1]
                successor = (x, y)
                
                # Si la posción es una pared o un óbstáculo, ignorar.
                if game_state.has_wall(x, y) or successor in obstacles:
                    continue
                
                # Nodo hijo
                child_node = AStarNode(successor, node, 1)
                
                # Añadir nodo hijo a la frontera, si no se ha visitado.
                if child_node not in expanded_nodes and child_node not in frontier.heap:
                    child_priority = child_node.cost + self.get_maze_distance(successor, target)
                    frontier.update(child_node, child_priority)

        # Si no se llega al nodo diana, devolvemos ruta vacía.
        return []

    #
    # Distancias
    #

    # Calcular distancia entre este agente y una diana.
    def distance(self, game_state, target):
        agent_position = game_state.get_agent_position(self.index)
        return self.get_maze_distance(agent_position, target)

    # Distancia a la guarida
    def distance_to_start(self, game_state):
        return self.distance(game_state, self.start)

    # Calcular la distancia mínima entre este agente y un lista de posiciones
    def min_distance(self, game_state, positions, extra=0):
        agent_position = game_state.get_agent_position(self.index)
        distances = []
        for position in positions:
            distance = self.get_maze_distance(agent_position, position)
            distances.append(distance)
        if len(distances) == 0:
            return 0
        return min(distances) + extra

    # Mínima distancia a oponentes observables
    def min_distance_observables(self, game_state, filtrum):
        agents = list(filter(filtrum, self.observable_agents(game_state)))
        positions = list(map(lambda a: a.get_position(), agents))
        return self.min_distance(game_state, positions, self.extra_distance)

    # Mínima distancia a pacmans observables
    def min_distance_observable_pacmans(self, game_state):
        filtrum = lambda a: a.is_pacman == True
        return self.min_distance_observables(game_state, filtrum)

    # Mínima distancia a fantasmas observables
    def min_distance_observable_ghosts(self, game_state):
        filtrum = lambda a: a.is_pacman == False
        return self.min_distance_observables(game_state, filtrum)

    # Mínima distancia a fantasmas observables no asustados
    def min_distance_observable_daring_ghosts(self, game_state):
        filtrum = lambda a: (a.is_pacman == False) and (a.scared_timer <= 0)
        return self.min_distance_observables(game_state, filtrum)

    def min_distance_observable_scared_ghosts(self, game_state):
        filtrum = lambda a: (a.is_pacman == False) and (a.scared_timer > 0)
        return self.min_distance_observables(game_state, filtrum)

    # Mínima distancia a la comida
    def min_distance_food(self, game_state):
        positions = self.get_food(game_state).as_list()
        return self.min_distance(game_state, positions)

    # Mínima distancia a las cápsulas
    def min_distance_capsules(self, game_state):
        positions = self.get_capsules(game_state)
        return self.min_distance(game_state, positions)

    # Mínima distancia a los cocos (comida + cápsulas)
    def min_distance_cocos(self, game_state):
        positions = self.get_food(game_state).as_list() + self.get_capsules(game_state)
        return self.min_distance(game_state, positions)

    # Mínima distancia a los cocos (comida + cápsulas)
    # que tiene una ruta sin obstáculos (oponentes observables)
    def min_distance_unobstructive_cocos(self, game_state):
        positions = self.unobstructive_cocos
        return self.min_distance(game_state, positions)

    # Mínima distancia a las cápsulas propias
    def min_distance_own_capsules(self, game_state):
        positions = self.get_capsules_you_are_defending(game_state)
        return self.min_distance(game_state, positions, 1)

    # Mínima distancia a los focos de patrulla
    def min_distance_patrol_focuses(self, game_state):
        positions = self.patrol_focuses
        return self.min_distance(game_state, positions, 1)

    # Mínima distancia a la frontera
    def min_distance_boundary(self, game_state):
        positions = self.boundary
        return self.min_distance(game_state, positions)

    # Mínima distancia a la subfrontera
    def min_distance_subboundary(self, game_state):
        positions = self.subboundary
        return self.min_distance(game_state, positions)

    # Sólo considerar distanicas cercanas.
    def ignore_far(self, distance):
        if distance > self.far_distance:
            return 0
        return distance

    #
    # Cantidades
    #

    # Cantidad de cápsulas
    def num_own_capsules(self, game_state):
        return len(self.get_capsules_you_are_defending(game_state))

    # Cantidad de comida acarreada
    def num_food_carrying(self, game_state):
        return game_state.get_agent_state(self.index).num_carrying

    # Cantidad de comida
    def num_food(self, game_state):
        return len(self.get_food(game_state).as_list())

    # Cantidad de cápsulas
    def num_capsules(self, game_state):
        return len(self.get_capsules(game_state))
    
    # Cantidad de cocos (comida + cápsulas)
    def num_cocos(self, game_state):
        return len(self.get_food(game_state).as_list() + self.get_capsules(game_state))

    #
    # Posiciones
    #

    # Establecer frontera.
    def set_boundary(self, game_state):
        x = self.get_boundary_x(game_state)
        ys = list(range(game_state.data.layout.height))
        self.boundary = []
        for y in ys:
            if not game_state.has_wall(x, y):
             self.boundary.append((x, y))

    # Obtener abcisa de la frontera.
    def get_boundary_x(self, game_state):
        x = game_state.data.layout.width // 2
        if self.red:
            x = x - 1
        return x

    # Establecer subfrontera.
    def set_subboundary(self, game_state):
        x = self.get_subboundary_x(game_state)
        ys = list(range(game_state.data.layout.height))
        self.subboundary = []
        for y in ys:
            if not game_state.has_wall(x, y):
             self.subboundary.append((x, y))

    # Obtener abcisa de la subfrontera.
    def get_subboundary_x(self, game_state):
        x = game_state.data.layout.width // 2 + 1
        if self.red:
            x = x - 3
        return x

    # Establece los focos de patrulla.
    def set_patrol_focuses(self, game_state):
        # Al principio, los focos de patrulla son la comida interesante (la más cercana para el oponente).
        # Después, los focos de patrulla son las últimas comidas consumidas por los oponentes.
        if self.is_first_turn(game_state):
            if self.patrol_interesting == True:
                self.patrol_focuses = self.get_interesting_food(game_state)
            else:
                self.patrol_focuses = self.get_capsules_you_are_defending(game_state)
        else:
            positions = self.positions_food_modified(self)
            if len(positions) > 0:
                self.patrol_focuses = positions

    # Obtiene la comida interesante (la más cercana para el oponente).
    def get_interesting_food(self, game_state):
        food = self.get_food_you_are_defending(game_state).as_list()
        capsules = self.get_capsules_you_are_defending(game_state)
        positions = food + capsules
        distances = []
        for position in positions:
            distance = self.get_maze_distance(self.opponent_start, position)
            distances.append(distance)
        min_distance = min(distances)
        interesting = []
        for index in range(len(positions)):
            if distances[index] == min_distance:
                interesting.append(positions[index])
        return interesting

    # Averiguar si se ha modificado la comida que se defiende.
    def positions_food_modified(self, game_state):
        if self.is_first_turn(game_state):
            return []
        previous = self.get_food_you_are_defending(self.get_previous_observation()).as_list()
        current = self.get_food_you_are_defending(self.get_current_observation()).as_list()
        return self.list_difference(previous, current)

    # Calcular la diferencia de dos listas.
    def list_difference(self, list_a, list_b):
        list_c = []
        for a in list_a:
            if a not in list_b:
                list_c.append(a)
        return list_c

    # Obtener aquellos cocos cuya ruta no está obstruida (es decir no hay oponentes en ella).
    def set_unobstructive_cocos(self, game_state):
        cocos = self.get_food(game_state).as_list() + self.get_capsules(game_state)
        if self.calculate_unobstructive == False:
            self.unobstructive_cocos = cocos
            return
        obstacles = self.observable_positions(game_state)
        origin = game_state.get_agent_state(self.index).get_position()
        self.unobstructive_cocos = []
        for coco in cocos:
            if self.is_reachable(game_state, origin, coco, obstacles):
                self.unobstructive_cocos.append(coco)

    # Verdadero si la ruta hacia la diana no tienen obstáculos.
    def is_reachable(self, game_state, origin, target, obstacles):
            path = self.a_star(game_state, origin, target)
            if len(path) == 0:
                return False
            for obstacle in obstacles:
                if obstacle in path:
                    return False
            return True

    # Distancia de la ruta si no tiene obtáculos
    def reachable_distance(self, game_state, origin, target, obstacles):
            far = 999
            path = self.a_star(game_state, origin, target)
            if len(path) == 0:
                return far
            for obstacle in obstacles:
                if obstacle in path:
                    return far
            return len(path)

    #
    # Info oponentes
    #

    def observable_agents(self, game_state):
        agents = self.opponent_agents(game_state)
        filtrum = lambda a: a.get_position() != None
        observable = list(filter(filtrum, agents))
        return observable

    def opponent_agents(self, game_state):
        agents = []
        for index in self.opponent_indices:
            agents.append(game_state.get_agent_state(index))
        return agents

    def observable_positions(self, game_state):
        agents = self.observable_agents(game_state)
        positions = list(map(lambda a: a.get_position(), agents))
        return positions

    def opponent_timers(self, game_state):
        agents = self.opponent_agents(game_state)
        return list(map(lambda a: a.scared_timer, agents))

#    def print_opponents_info(self, game_state):
#        for index in self.opponent_indices:
#            agent = game_state.get_agent_state(index)
#            print(" is_pacman", agent.is_pacman, "position", agent.get_position(), "timer", agent.scared_timer, end="")
#        print()

    #
    # Condiciones
    #
    
    # Verdadero si es el primer turno de la partida
    def is_first_turn(self, game_state):
        return self.get_previous_observation() is None

    # Verdadero si el agente ha llegado a su capacidad de acarrear comida
    def is_fed(self, game_state):
        return self.num_food_carrying(game_state) >= self.capacity

    # Verdadero si el agente está asustado.
    def is_scared(self, game_state):
        agent = game_state.get_agent_state(self.index)
        return agent.scared_timer > 0

    # Verdadero si alguno de los oponentes está asustado.
    def are_some_opponents_scared(self, game_state):
        return max(self.opponent_timers(game_state)) > 0

    # Verdadero si todos los oponentes están assustados.
    def are_all_opponents_scared(self, game_state):
        return min(self.opponent_timers(games_state)) > 0

    # Verdadero si el agente está en casa (hemicampo propio)
    def is_at_home(self, game_state):
        agent = game_state.get_agent_state(self.index)
        return not agent.is_pacman

    #
    # Objetivos
    #
    
    def feature_eat_cocos(self, game_state):
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        cocos = food + capsules
        return -(self.min_distance(game_state, cocos) + 100 * len(food) + 100000 * len(capsules))
        #return -(self.min_distance_cocos(game_state) + 100 * self.num_food(game_state) + 100000 * self.num_capsules(game_state))
    
    # Obtener característica para la función de evaluación según objetivo.
    def get_feature(self, game_state, name):
    
        # Características ofensivas:
    
        if name == "eat_cocos":
            return self.feature_eat_cocos(game_state)
            
        if name == "flee_daring_ghosts":
            return self.ignore_far(self.min_distance_observable_daring_ghosts(game_state))

        if name == "go_subboundary":
            return -self.min_distance_subboundary(game_state)

        if name == "hunt_scared_ghosts":
            return -self.min_distance_observable_scared_ghosts(game_state)

        # Características defensivas:

        if name == "patrol":
            return -self.min_distance_patrol_focuses(game_state)

        if name == "flee_pacmans":
            return self.ignore_far(self.min_distance_observable_pacmans(game_state))

        if name == "hunt_pacmans":
            return -self.min_distance_observable_pacmans(game_state)

        if name == "stay_at_home":
            return 1 if self.is_at_home(game_state) else -1

        # Características descartadas:

        if name == "eat_capsules":
            return -(self.min_distance_capsules(game_state) + 10000 * self.num_capsules(game_state))

        if name == "eat_food":
            return -(self.min_distance_food(game_state) + 100 * self.num_food(game_state))

        if name == "eat_unobstructive_cocos":
            return -(self.min_distance_unobstructive_cocos(game_state) + 100 * self.num_food(game_state) + 100000 * self.num_capsules(game_state))

        if name == "flee_ghosts":
            return self.ignore_far(self.min_distance_observable_ghosts(game_state))

        if name == "go_boundary":
            return -self.min_distance_boundary(game_state)

        if name == "go_start":
            return -self.distance_to_start(game_state)

        if name == "hunt_ghosts":
            return -self.min_distance_observable_ghosts(game_state)

        if name == "protect":
            return -self.min_distance_own_capsules(game_state)

        #print("Erroneus feature:", name)
        return 0


#
# Agente ofensivo
#
class OffensiveExpectiminimaxAgent(ExpectiminimaxAgent):

    def select_preferences(self, game_state):
        self.role = "offensive"
        # El agente no considera parar.
        # (Se reduce un 35% el número de nodos).
        self.skip_stop = True
        # Se considera que los oponentes son semiagresivos.
        # (Se usa una mezcla de MINIMAX y EXPECTIMINIMAX.)
        self.opponent_aggressivity = 0.5
        # Se patrullan inicialmente la comida interesante (la más cercana para el oponente).
        self.patrol_interesting = True
        # No se calculan las rutas sin obtáculos.
        # (Tarda demasiado en computar.)
        self.calculate_unobstructive = False
        # Se añade una distancia extra a los oponentes.
        self.extra_distance = 1
        # La distancia máxima de laberinto a la que se reacciona al huir es 6.
        self.far_distance = 6
        # Se recolecta comida de 1 en 1.
        self.capacity = 1

    def select_goals(self, game_state):
        if not self.is_fed(game_state):
            # Si no ha recolectado suficiente comida:
            #   *  comer cocos (comdida + cápsulas)
            #   *  huir de fantasmas no asustados cercanos
            #   *  cazar fantasmas asustados obsevables
            self.goals = {"eat_cocos": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}
        else:
            # Si ha recolectado suficiente comida:
            #   *  volver al punto más cercano de casa
            #   *  huir de fantasmas no asustados cercanos
            #   *  cazar fantasmas asustados obsevables
            self.goals = {"go_subboundary": 1, "flee_daring_ghosts": 1000, "hunt_scared_ghosts": 100}


#
# Agente defensivo
#
class DefensiveExpectiminimaxAgent(ExpectiminimaxAgent):

    # Preferencias
    def select_preferences(self, game_state):
        self.role = "defensive"
        # El agente no considera parar.
        # (Se reduce un 35% el número de nodos).
        self.skip_stop = True
        # Se considera que los oponentes son semiagresivos.
        # (Se usa una mezcla de MINIMAX y EXPECTIMINIMAX.)
        self.opponent_aggressivity = 0.5
        # Se patrullan inicialmente la comida interesante (la más cercana para el oponente).
        self.patrol_interesting = True
        # No se calculan las rutas sin obtáculos.
        # (Tarda demasiado en computar.)
        self.calculate_unobstructive = False
        # Se añade una distancia extra a los oponentes.
        self.extra_distance = 1
        # La distancia máxima de laberinto a la que se reacciona al huir es 6.
        self.far_distance = 6
        # No se recolecta comida.
        self.capacity = 0

    # Objetivos
    def select_goals(self, game_state):
        if not self.is_scared(game_state):
            # Si no está asusado:
            #   *  patrullar (ir la comida más cercana para el openente o a la última comida modificada)
            #   *  cazar pacmans observables
            #   *  permanecer en casa
            self.goals = {"patrol": 1, "hunt_pacmans": 100, "stay_at_home": 1000000}
        else:
            # Si está asusado:
            #   *  patrullar (ir la comida más cercana para el openente o a la última comida modificada)
            #   *  huir de pacmans cercanos
            #   *  permanecer en casa
            self.goals = {"patrol": 1, "flee_pacmans": 100, "stay_at_home": 1000000}

#
