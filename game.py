from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents.DeepAgent import DeepAgent
from agents.LightcycleAgent import LightcycleAgent
from agents.RandomAgent import RandomAgent


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B


def getStartingPosition(startingPositions):
    options = [(2, 13), (23, 13), (13, 2), (13, 23), (2, 6), (23, 20), (2, 20), (23, 6), (6, 2), (20, 23), (20, 2),
               (6, 23)]
    return next(x for x in options if x not in startingPositions)


def getStartingDirection(position):
    if max(26 - position[0], position[0]) > max(26 - position[1], position[1]):
        if 26 - position[0] > position[0]:
            return 'E'
        else:
            return 'W'
    else:
        if 26 - position[1] > position[1]:
            return 'N'
        else:
            return 'S'


class TronModel(Model):
    def __init__(self, n_agents):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(26, 26, torus=False)
        self.startingPositions = []
        self.n_agents = n_agents
        for i in range(n_agents):
            self.startingPositions.append(getStartingPosition(self.startingPositions))
            if i == 0:
                a = DeepAgent(i, self.startingPositions[-1], getStartingDirection(self.startingPositions[-1]), self, 3)
            elif i == 2 or i == 3:
                a = RandomAgent(i, self.startingPositions[-1], getStartingDirection(self.startingPositions[-1]),
                                self, 20, 625, 3)
            else:
                a = LightcycleAgent(i, self.startingPositions[-1], getStartingDirection(self.startingPositions[-1]),
                                    self, 20, 625, 3)

            self.schedule.add(a)
            self.grid.place_agent(a, self.startingPositions[-1])
        self.n_games = 0
        self.record = 0
        self.epsilon = 0.8

    def step(self):
        temp_games = self.n_games
        self.schedule.step()
        record = self.record
        epsilon = self.epsilon
        for agent in self.schedule.agents:
            if type(agent) == DeepAgent:
                epsilon = agent.epsilon
                if record < agent.score:
                    epsilon *= 0.9
                    record = agent.score
                    self.record = record
                    self.epsilon = epsilon

        if self.schedule.get_agent_count() < 2:
            if self.schedule.get_agent_count() == 1:
                winner = type(self.schedule.agents[0])
                if type(winner) == DeepAgent:
                    state_old, final_move, reward, state_new, done = self.schedule.agents[0].memory[-1]
                    self.schedule.agents[0].memory[-1] = (state_old, final_move, 100, state_new, done)
                    self.schedule.agents[0].train_long_memory()
                    self.schedule.agents[0].net.save()
                    epsilon *= 0.9
                    self.epsilon = epsilon
                print('Game: ', self.n_games, 'Winner: ', winner, 'NN Record: ', record, 'Epsilon: ', self.epsilon)
            else:
                print('Game: ', self.n_games, ' Draw')

            self.__init__(self.n_agents)
            self.n_games = temp_games + 1
            self.record = record
            self.epsilon = epsilon
            for i in range(len(self.schedule.agents)):
                self.schedule.agents[i].n_games = self.n_games
                if type(self.schedule.agents[i]) == DeepAgent:
                    self.schedule.agents[i].epsilon = self.epsilon


if __name__ == '__main__':
    number_of_agents = 2
    model = TronModel(number_of_agents)
    while True:
        model.step()
