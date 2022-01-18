import random
from mesa import Agent


class LightcycleAgent(Agent):
    """
    Lightcycle driver agent
    """

    def __init__(self, unique_id, pos, direction, model, fov, max_path_length, agent_type):
        """
        Create a new Lightcycle agent.

        Args:
           x, y: Agent initial location.
        """
        super().__init__(pos, model)
        self.unique_id = unique_id
        self.pos = pos
        self.lightpath = set()
        self.others_lightpaths = set()
        self.boundries = [(-1, n) for n in range(0, 26)] + [(26, n) for n in range(0, 26)] + \
                         [(n, -1) for n in range(0, 26)] + [(n, 26) for n in range(0, 26)]
        self.direction = direction
        self.first_move = True
        self.fov = fov
        self.max_path_length = max_path_length
        self.ordered_lightpath = [self.pos]
        self.agent_type = agent_type
        self.n_games = 0

    def move(self, fillings):
        new_direction = ''
        new_pos = self.pos
        while len(fillings) > 0:

            if self.first_move:
                new_direction = random.choice(list(fillings.keys()))
                self.first_move = False
            else:
                new_direction = min(fillings, key=fillings.get)
            new_pos = list(self.pos)
            if new_direction == 'N':
                new_pos[1] += 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            elif new_direction == 'S':
                new_pos[1] -= 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            elif new_direction == 'W':
                new_pos[0] -= 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

            elif new_direction == 'E':
                new_pos[0] += 1
                if tuple(new_pos) in self.lightpath or tuple(new_pos) in self.boundries or tuple(
                        new_pos) in self.others_lightpaths:
                    del fillings[new_direction]
                else:
                    break

        if len(fillings) < 1 or not self.model.grid.is_cell_empty(new_pos):
            self.death()
            self.model.schedule.remove(self)
        else:
            self.model.grid.place_agent(self, tuple(new_pos))
            self.direction = new_direction
            self.pos = tuple(new_pos)
            self.ordered_lightpath.append(self.pos)
            self.eat_your_tail()

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

    def step(self):
        self.lightpath.add(self.pos)
        self.observation()

        all_paths = set.union(self.lightpath, self.others_lightpaths)
        if self.direction == 'N':

            left = len([n for n in all_paths if n[0] < self.pos[0]])
            front = len([n for n in all_paths if n[1] > self.pos[1]])
            right = len([n for n in all_paths if n[0] > self.pos[0]])
            fillings = {'W': left, 'N': front, 'E': right}

        elif self.direction == 'S':
            left = len([n for n in all_paths if n[0] > self.pos[0]])
            front = len([n for n in all_paths if n[1] < self.pos[1]])
            right = len([n for n in all_paths if n[0] < self.pos[0]])
            fillings = {'W': right, 'S': front, 'E': left}

        elif self.direction == 'W':
            left = len([n for n in all_paths if n[1] < self.pos[1]])
            front = len([n for n in all_paths if n[0] < self.pos[0]])
            right = len([n for n in all_paths if n[1] > self.pos[1]])
            fillings = {'N': right, 'W': front, 'S': left}

        else:
            left = len([n for n in all_paths if n[1] > self.pos[1]])
            front = len([n for n in all_paths if n[0] > self.pos[0]])
            right = len([n for n in all_paths if n[1] < self.pos[1]])
            fillings = {'S': right, 'E': front, 'N': left}
        self.move(fillings)

    def death(self):

        for coords in self.lightpath:
            for agent in self.model.schedule.agents:
                if agent.unique_id != self.unique_id:
                    if coords in agent.others_lightpaths:
                        agent.others_lightpaths.remove(coords)
            self.model.grid._remove_agent(coords, self.model.grid[coords[0], coords[1]][0])
