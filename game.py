import random

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents.DeepAgent import DeepAgent
from agents.LightcycleAgent import LightcycleAgent
from agents.RandomAgent import RandomAgent


def getStartingPosition(startingPositions, isRandom):
    if isRandom:
        coords = (random.randrange(0, 11), random.randrange(0, 11))
        while coords in startingPositions:
            coords = (random.randrange(0, 11), random.randrange(0, 11))
        return coords
    else:
        options = [(1, 6), (10, 6), (6, 1), (6, 10), (1, 3), (10, 8), (1, 8), (10, 3), (3, 1), (8, 10), (8, 1),
                   (3, 10)]
        return next(x for x in options if x not in startingPositions)


def getStartingDirection(position, isRandom):
    if isRandom:
        return random.choice(['N', 'S', 'W', 'E'])
    if max(11 - position[0], position[0]) > max(11 - position[1], position[1]):
        if 11 - position[0] > position[0]:
            return 'E'
        else:
            return 'W'
    else:
        if 11 - position[1] > position[1]:
            return 'N'
        else:
            return 'S'


class TronModel(Model):
    def __init__(self, n_agents, fov, max_path_length, isStartingPositionRandom):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(12, 12, torus=False)
        self.startingPositions = []
        self.n_agents = n_agents
        self.fov = fov
        self.max_path_length = max_path_length
        self.isStartingPositionRandom = isStartingPositionRandom
        for i in range(n_agents):
            self.startingPositions.append(getStartingPosition(self.startingPositions, isStartingPositionRandom))
            if i == 0 or i == 2:
                a = DeepAgent(i,  self.startingPositions[-1],
                              getStartingDirection(self.startingPositions[-1], isStartingPositionRandom),
                              self, fov, max_path_length, 3)
            elif i == 1 or i == 3:
                a = LightcycleAgent(i,  self.startingPositions[-1],
                                    getStartingDirection(self.startingPositions[-1], isStartingPositionRandom),
                                    self, fov, max_path_length, 3)
            else:
                a = RandomAgent(i,  self.startingPositions[-1],
                                getStartingDirection(self.startingPositions[-1], isStartingPositionRandom),
                                self, fov, max_path_length, 3)

            self.schedule.add(a)
            self.grid.place_agent(a, self.startingPositions[-1])
        self.n_games = 0
        self.record = 0
        self.epsilon = 0.8
        self.scores = []

    def step(self):
        self.schedule.step()
        if self.schedule.get_agent_count() < 2:
            if self.schedule.get_agent_count() == 1:
                winner = type(self.schedule.agents[0])
                if winner.__name__ == 'DeepAgent':
                    state_old, final_move, reward, state_new, done = self.schedule.agents[0].memory[-1]
                    self.schedule.agents[0].memory[-1] = (state_old, final_move, 100, state_new, done)
                    self.schedule.agents[0].train_long_memory()
                    self.schedule.agents[0].net.save(self.schedule.agents[0].unique_id)
                    self.scores.append(self.schedule.agents[0].score)
                    self.epsilon *= 0.999
                print('Game:', len(self.scores), 'Winner:', winner.__name__, self.schedule.agents[0].unique_id, 'NN Score:', self.scores[-1],
                      'NN Average:', round(sum(self.scores) / len(self.scores), 2), 'Epsilon:', round(self.epsilon, 3))
            else:
                print('Game: ', len(self.scores), ' Draw')
            scores = self.scores
            epsilon = self.epsilon
            self.__init__(self.n_agents, self.fov, self.max_path_length, self.isStartingPositionRandom)
            self.scores = scores
            self.epsilon = epsilon
            for i in range(len(self.schedule.agents)):
                if type(self.schedule.agents[i]) == DeepAgent:
                    self.schedule.agents[i].epsilon = self.epsilon


if __name__ == '__main__':
    number_of_agents = 2
    model = TronModel(number_of_agents, 10, 100, True)
    while True:
        model.step()
