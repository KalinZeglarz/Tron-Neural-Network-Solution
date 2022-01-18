import random
from mesa import Agent


class RandomAgent(Agent):

    def __init__(self, unique_id, pos, direction, model, fov, max_path_length, agent_type):
        super().__init__(pos, model)
        self.unique_id = unique_id
        self.pos = pos
        self.boundries = [(-1, n) for n in range(0, 26)] + [(26, n) for n in range(0, 26)] + \
                         [(n, -1) for n in range(0, 26)] + [(n, 26) for n in range(0, 26)]
        self.direction = direction
        self.agent_type = agent_type
        self.n_games = 0
        self.lightpath = set()
        self.others_lightpaths = set()
        for point in self.boundries:
            self.others_lightpaths.add(point)

    def move(self, fillings):
        new_direction = random.choice([*fillings])
        new_pos = list(self.pos)
        if new_direction == 'N':
            new_pos[1] += 1
        elif new_direction == 'S':
            new_pos[1] -= 1
        elif new_direction == 'W':
            new_pos[0] -= 1
        elif new_direction == 'E':
            new_pos[0] += 1
        if tuple(new_pos) in self.boundries or not self.model.grid.is_cell_empty(tuple(new_pos)):
            self.death()
            self.model.schedule.remove(self)
        else:
            self.model.grid.place_agent(self, tuple(new_pos))
            self.direction = new_direction
            self.pos = tuple(new_pos)

    def step(self):
        self.lightpath.add(self.pos)
        if self.direction == 'N':
            fillings = ['W', 'N', 'E']
        elif self.direction == 'S':
            fillings = ['W', 'S', 'E']
        elif self.direction == 'W':
            fillings = ['N', 'W', 'S']
        else:
            fillings = ['S', 'E', 'N']
        self.move(fillings)
        # self.model.grid.place_agent(self, tuple(self.pos))

    def death(self):

        for coords in self.lightpath:
            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id:
                    if coords in agent.others_lightpaths:
                        agent.others_lightpaths.remove(coords)
            self.model.grid._remove_agent(coords, self.model.grid[coords[0], coords[1]][0])
