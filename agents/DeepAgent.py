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
    def __init__(self, unique_id, pos, direction, model, agent_type):
        super().__init__(pos, model)
        self.unique_id = unique_id
        self.pos = pos
        self.direction = direction

        self.epsilon = 0.80  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.net = LinearQNet(11, 256, 3)
        self.net.load()
        self.trainer = QTrainer(self.net, lr=LEARNING_RATE, gamma=self.gamma)

        self.lightpath = set()
        self.others_lightpaths = set()
        self.boundries = [(-1, n) for n in range(0, 26)] + [(26, n) for n in range(0, 26)] + \
                         [(n, -1) for n in range(0, 26)] + [(n, 26) for n in range(0, 26)]
        self.direction = direction
        self.agent_type = agent_type
        self.n_games = 1
        self.score = 0

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

        state = [
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
        for agent in self.model.schedule.agents:
            if agent.unique_id != self.unique_id:
                for point in agent.lightpath:
                    self.others_lightpaths.add(point)
                    self.others_lightpaths.add(agent.pos)

    def find_empty(self, point, dir):
        point_l = (point[0] - 1, point[1])
        point_r = (point[0] + 1, point[1])
        point_u = (point[0], point[1] - 1)
        point_d = (point[0], point[1] + 1)

        if dir == 'N':
            if self.is_collision(point_r):
                if self.is_collision(point_l):
                    if self.is_collision(point_u):
                        return (-1, -1), 'X'
                    return point_u, 'N'
                return point_l, 'W'
            return point_r, 'E'

        elif dir == 'S':
            if self.is_collision(point_l):
                if self.is_collision(point_r):
                    if self.is_collision(point_d):
                        return (-1, -1), 'X'
                    return point_d, 'S'
                return point_r, 'E'
            return point_l, 'W'

        elif dir == 'E':
            if self.is_collision(point_u):
                if self.is_collision(point_r):
                    if self.is_collision(point_d):
                        return (-1, -1), 'X'
                    return point_d, 'S'
                return point_r, 'E'
            return point_u, 'N'

        elif dir == 'W':
            if self.is_collision(point_u):
                if self.is_collision(point_l):
                    if self.is_collision(point_d):
                        return (-1, -1), 'X'
                    return point_d, 'S'
                return point_l, 'W'
            return point_u, 'N'
        else:
            return (-1, -1), 'X'

    def is_collision(self, point):
        if point in self.lightpath:
            return 1
        elif point in self.others_lightpaths:
            return 2
        elif point in self.boundries:
            return 3
        else:
            return 0

    def step(self):
        self.lightpath.add(self.pos)
        self.observation()
        state_old = self.get_state(self.pos, self.direction)
        new_pos = self.pos
        new_direction = self.direction
        final_move = self.get_action(state_old)
        # [1, 0, 0] - front
        # [0, 1, 0] - right
        # [0, 0, 1] - left
        if self.direction == 'N':
            if final_move == [1, 0, 0]:
                new_pos = (self.pos[0], self.pos[1] + 1)
                new_direction = 'N'
            elif final_move == [0, 1, 0]:
                new_pos = (self.pos[0] + 1, self.pos[1])
                new_direction = 'E'
            elif final_move == [0, 0, 1]:
                new_pos = (self.pos[0] - 1, self.pos[1])
                new_direction = 'W'

        elif self.direction == 'S':
            if final_move == [1, 0, 0]:
                new_pos = (self.pos[0], self.pos[1] - 1)
                new_direction = 'S'
            elif final_move == [0, 1, 0]:
                new_pos = (self.pos[0] - 1, self.pos[1])
                new_direction = 'W'
            elif final_move == [0, 0, 1]:
                new_pos = (self.pos[0] + 1, self.pos[1])
                new_direction = 'E'

        elif self.direction == 'W':
            if final_move == [1, 0, 0]:
                new_pos = (self.pos[0] - 1, self.pos[1])
                new_direction = 'W'
            elif final_move == [0, 1, 0]:
                new_pos = (self.pos[0], self.pos[1] + 1)
                new_direction = 'N'
            elif final_move == [0, 0, 1]:
                new_pos = (self.pos[0], self.pos[1] - 1)
                new_direction = 'S'

        elif self.direction == 'E':
            if final_move == [1, 0, 0]:
                new_pos = (self.pos[0] + 1, self.pos[1])
                new_direction = 'E'
            elif final_move == [0, 1, 0]:
                new_pos = (self.pos[0], self.pos[1] - 1)
                new_direction = 'S'
            elif final_move == [0, 0, 1]:
                new_pos = (self.pos[0], self.pos[1] + 1)
                new_direction = 'N'

        state_new = self.get_state(tuple(new_pos), new_direction)
        if new_pos in self.lightpath or new_pos in self.boundries:
            self.remember(state_old, final_move, -1000, state_new, False)
            new_pos, new_direction = self.find_empty(new_pos, new_direction)
            if new_pos == (-1, -1):
                self.death()
                self.model.schedule.remove(self)
                self.remember(state_old, final_move, -1000, state_new, False)
                self.train_long_memory()
                self.net.save()

        elif new_pos in self.boundries or not self.model.grid.is_cell_empty(new_pos):
            self.death()
            self.model.schedule.remove(self)
            self.remember(state_old, final_move, -1000, state_new, False)
            self.train_long_memory()
            self.net.save()

        else:
            self.model.grid.place_agent(self, tuple(new_pos))
            self.direction = new_direction
            self.pos = tuple(new_pos)
            self.remember(state_old, final_move, 10, state_new, False)
            self.score += 1

    def death(self):
        for coords in self.lightpath:
            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id:
                    if coords in agent.others_lightpaths:
                        agent.others_lightpaths.remove(coords)
            self.model.grid._remove_agent(coords, self.model.grid[coords[0], coords[1]][0])
