import random
from mesa import Agent


class RandomAgent(Agent):

    def __init__(self, unique_id, pos, direction, model, fov, max_path_length, agent_type):
        super().__init__(pos, model)
        self.unique_id = unique_id
        self.pos = pos
        self.boundries = [(-1, n) for n in range(0, 12)] + [(12, n) for n in range(0, 12)] + \
                         [(n, -1) for n in range(0, 12)] + [(n, 12) for n in range(0, 12)]
        self.direction = direction
        self.agent_type = agent_type
        self.n_games = 0
        self.lightpath = set()
        self.ordered_lightpath = [self.pos]
        self.others_lightpaths = set()
        for point in self.boundries:
            self.others_lightpaths.add(point)
        self.fov = fov
        self.max_path_length = max_path_length


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
            self.ordered_lightpath.append(self.pos)
            self.eat_your_tail()

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

    def death(self):

        for coords in self.lightpath:
            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id:
                    if coords in agent.others_lightpaths:
                        agent.others_lightpaths.remove(coords)
            self.model.grid._remove_agent(coords, self.model.grid[coords[0], coords[1]][0])
