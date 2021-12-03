import random

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agent import LightcycleAgent


def getStartingPosition(startingPositions, isRandom):
    if isRandom:
        coords = (random.randrange(0, 25), random.randrange(0, 25))
        while coords in startingPositions:
            coords = (random.randrange(0, 25), random.randrange(0, 25))
        return coords
    else:
        options = [(2, 13), (23, 13), (13, 2), (13, 23), (2, 6), (23, 20), (2, 20), (23, 6), (6, 2), (20, 23), (20, 2),
                   (6, 23)]
        return next(x for x in options if x not in startingPositions)


def getStartingDirection(position, isRandom):
    if isRandom:
        return random.choice(['N', 'S', 'W', 'E'])
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
    def __init__(self, n_agents, max_path_length, fov, isStartingPositionRandom, teams):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(26, 26, torus=False)
        self.startingPositions = []

        for i in range(n_agents):
            self.startingPositions.append(getStartingPosition(self.startingPositions, isStartingPositionRandom))
            if teams:
                a = LightcycleAgent(i, self.startingPositions[-1],
                                    getStartingDirection(self.startingPositions[-1], isStartingPositionRandom), self, fov,
                                    max_path_length, i % 2)
            else:
                a = LightcycleAgent(i, self.startingPositions[-1],
                                    getStartingDirection(self.startingPositions[-1], isStartingPositionRandom), self, fov,
                                    max_path_length, 3)

            self.schedule.add(a)
            self.grid.place_agent(a, self.startingPositions[-1])

    def step(self):
        self.schedule.step()


if __name__ == '__main__':
    model = TronModel(5, 3, 3, False)
    model.step()
