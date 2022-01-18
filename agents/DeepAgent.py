from mesa import Agent
import torch
import random
import numpy as np
from collections import deque
from neuralNetwork import LinearQNet, QTrainer

MAX_MEMORY = 500
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class DeepAgent(Agent):
    def __init__(self, unique_id, pos, direction, model, fov, max_path_length, agent_type):
        super().__init__(pos, model)
        self.unique_id = unique_id
        self.pos = pos
        self.direction = direction

        self.epsilon = 0.80  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.net = LinearQNet(17, 256, 3)
        self.net.load(self.unique_id)
        self.trainer = QTrainer(self.net, lr=LEARNING_RATE, gamma=self.gamma)

        self.lightpath = set()
        self.ordered_lightpath = [self.pos]
        self.others_lightpaths = set()
        self.boundries = [(-1, n) for n in range(0, 12)] + [(12, n) for n in range(0, 12)] + \
                         [(n, -1) for n in range(0, 12)] + [(n, 12) for n in range(0, 12)]
        self.direction = direction
        self.agent_type = agent_type
        self.score = 0
        self.fov = fov
        self.max_path_length = max_path_length

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_state(self, position, direction):
        point_l = (position[0] - 1, position[1])
        point_r = (position[0] + 1, position[1])
        point_u = (position[0], position[1] - 1)
        point_d = (position[0], position[1] + 1)

        dir_l = direction == 'W'
        dir_r = direction == 'E'
        dir_u = direction == 'N'
        dir_d = direction == 'S'

        op_left = False
        op_right = False
        op_up = False
        op_down = False

        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                if agent.pos[0] < self.pos[0]:
                    op_left = True
                if agent.pos[0] > self.pos[0]:
                    op_right = True
                if agent.pos[1] < self.pos[1]:
                    op_up = True
                if agent.pos[1] > self.pos[1]:
                    op_down = True

        boundry_left = self.pos[0]
        boundry_right = 11 - self.pos[0]
        boundry_up = 11 - self.pos[1]
        boundry_down = self.pos[1]

        state = [
            # Head
            self.pos[0],
            self.pos[1],

            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Distance to boundries
            boundry_left,
            boundry_right,
            boundry_up,
            boundry_down,

            # Opponent location
            op_left,
            op_right,
            op_up,
            op_down

        ]
        return np.array(state, dtype=int)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0, 0, 0]
        if np.random.choice([True, False], p=[1 - self.epsilon, self.epsilon]):
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.net(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        else:
            move = random.randint(0, 2)
            final_move[move] = 1

        return final_move

    def observation(self):
        fov_grid = [(self.pos[0], self.pos[1] + n) for n in range(-self.fov, self.fov + 1) if
                    self.pos[1] + n >= 0 and self.pos[1] + n <= 25] + \
                   [(self.pos[0] + n, self.pos[1]) for n in range(-self.fov, self.fov + 1) if
                    self.pos[0] + n >= 0 and self.pos[0] + n <= 25]
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                for point in agent.lightpath:
                    if point in fov_grid:
                        self.others_lightpaths.add(point)
                if agent.pos in fov_grid:
                    self.others_lightpaths.add(agent.pos)
        # for agent in self.model.schedule.agents:
        #     if agent.unique_id != self.unique_id:
        #         for point in agent.lightpath:
        #             self.others_lightpaths.add(point)
        #             self.others_lightpaths.add(agent.pos)

    def eat_your_tail(self):
        if len(self.ordered_lightpath) > self.max_path_length:
            to_delete = self.ordered_lightpath[0]
            self.lightpath.remove(to_delete)
            self.ordered_lightpath = self.ordered_lightpath[1:]
            self.model.grid._remove_agent(to_delete,
                                          self.model.grid[to_delete[0], to_delete[1]][0])

            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id:
                    if to_delete in agent.others_lightpaths:
                        agent.others_lightpaths.remove(to_delete)

    def is_collision(self, point):
        if point in self.lightpath:
            return 1
        elif point in self.others_lightpaths:
            return 2
        elif point in self.boundries:
            return 3
        else:
            return 0

    def get_pos_from_move(self, move):
        if self.direction == 'N':
            if move == [1, 0, 0]:
                new_pos = (self.pos[0], self.pos[1] + 1)
                new_direction = 'N'
            elif move == [0, 1, 0]:
                new_pos = (self.pos[0] + 1, self.pos[1])
                new_direction = 'E'
            elif move == [0, 0, 1]:
                new_pos = (self.pos[0] - 1, self.pos[1])
                new_direction = 'W'

        elif self.direction == 'S':
            if move == [1, 0, 0]:
                new_pos = (self.pos[0], self.pos[1] - 1)
                new_direction = 'S'
            elif move == [0, 1, 0]:
                new_pos = (self.pos[0] - 1, self.pos[1])
                new_direction = 'W'
            elif move == [0, 0, 1]:
                new_pos = (self.pos[0] + 1, self.pos[1])
                new_direction = 'E'

        elif self.direction == 'W':
            if move == [1, 0, 0]:
                new_pos = (self.pos[0] - 1, self.pos[1])
                new_direction = 'W'
            elif move == [0, 1, 0]:
                new_pos = (self.pos[0], self.pos[1] + 1)
                new_direction = 'N'
            elif move == [0, 0, 1]:
                new_pos = (self.pos[0], self.pos[1] - 1)
                new_direction = 'S'

        elif self.direction == 'E':
            if move == [1, 0, 0]:
                new_pos = (self.pos[0] + 1, self.pos[1])
                new_direction = 'E'
            elif move == [0, 1, 0]:
                new_pos = (self.pos[0], self.pos[1] - 1)
                new_direction = 'S'
            elif move == [0, 0, 1]:
                new_pos = (self.pos[0], self.pos[1] + 1)
                new_direction = 'N'
        return new_pos, new_direction

    def step(self):
        self.lightpath.add(self.pos)
        self.observation()
        state_old = self.get_state(self.pos, self.direction)
        final_move = self.get_action(state_old)
        # [1, 0, 0] - front
        # [0, 1, 0] - right
        # [0, 0, 1] - left
        new_pos, new_direction = self.get_pos_from_move(final_move)
        if new_pos in self.lightpath or new_pos in self.boundries or new_pos in self.others_lightpaths:
            state_new = self.get_state(tuple(new_pos), new_direction)
            self.remember(state_old, final_move, -25, state_new, False)
            temp_epsilon = self.epsilon
            self.epsilon = 1
            temp_pos = new_pos
            while temp_pos == new_pos:
                final_move = self.get_action(state_old)
                temp_pos, new_direction = self.get_pos_from_move(final_move)
            if temp_pos in self.lightpath or new_pos in self.boundries or temp_pos in self.others_lightpaths:
                state_new = self.get_state(tuple(temp_pos), new_direction)
                self.remember(state_old, final_move, -25, state_new, False)
                temp_pos2 = temp_pos
                while temp_pos2 == temp_pos or temp_pos2 == new_pos:
                    final_move = self.get_action(state_old)
                    temp_pos2, new_direction = self.get_pos_from_move(final_move)
                new_pos = temp_pos2
            else:
                new_pos = temp_pos
            self.epsilon = temp_epsilon
        state_new = self.get_state(tuple(new_pos), new_direction)

        if new_pos in self.boundries or not self.model.grid.is_cell_empty(new_pos):
            self.model.scores.append(self.score)
            self.death()
            self.model.schedule.remove(self)
            self.remember(state_old, final_move, -25, state_new, False)
            self.train_long_memory()
            self.net.save(self.unique_id)


        else:
            self.model.grid.place_agent(self, tuple(new_pos))
            self.direction = new_direction
            self.pos = tuple(new_pos)
            self.ordered_lightpath.append(self.pos)
            self.remember(state_old, final_move, 10, state_new, False)
            self.score += 1
            self.eat_your_tail()

    def death(self):
        for coords in self.lightpath:
            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id:
                    if coords in agent.others_lightpaths:
                        agent.others_lightpaths.remove(coords)
            self.model.grid._remove_agent(coords, self.model.grid[coords[0], coords[1]][0])
