import pygame
import numpy as np
import time
import random
import math
from warnings import warn
clock = pygame.time.Clock()

class Entity:
    all_entities = {}
    radius = 4
    infection_radius = radius + 7
    def __init__(self, x, y, model, name):
        self.x = x
        self.y = y
        self.r = 4
        self.c = (0, 255, 0)
        self.name = name
        self.model = model  # The model inside which it will be simulated
        self.infected_by = set()  # The entities by which it is infected
        self.removed = False
        self.infectious = False
        self.others_infected = 0
        self.all_entities[name] = self
        self.jiggle_rate = 1
        self.JIGGLE_STEP = 2
        self.SPEED = 4
        self.SOCIAL_DISTANCING = self.infection_radius
        self.REPELING_FORCE = self.infection_radius
        self.social_distancer = True  # Wheather this entity obeys social distancing or not
        self.quarantined = False
        self.moving = False  # wheather its moving to central location or not
        self.infection_time = 0  # its time of infection
        self.disinfection_time = None  # its time of disinfection
        self.nearby = {}  # { Entity nearby -> time spent inside its infection radius }

    @property
    def infected(self):
        return True if len(self.infected_by) > 0 else False

    @property
    def color(self):
        if self.removed:
            # self.c = (255, 107, 0)
            self.c = (128, 128, 128)
        elif len(self.infected_by) == 0:
            self.c = (0, 255, 0)
        elif self.moving:
            self.c = (160, 100, 30)
        else:
            self.c = (255, 0, 0)
        return self.c

    @classmethod
    def set_radius(cls, value):
        cls.radius = value

    @classmethod
    def set_infection_radius(cls, value):
        cls.infection_radius = cls.radius + value

    @property
    def infected_period(self):
        if self.removed is True:
            return round(self.disinfection_time - self.infection_time, 2)
        return round(time.time() - self.infection_time, 2) if self.infection_time != 0 else 0

    @property
    def status(self):
        if self.infected:
            return 'i'
        elif self.removed:
            return 'r'
        else:
            return 's'

    def jiggle(self, by=2):
        """This function randomly increases/decreases the x, y value of the entity, but not pushing it out of the borders"""
        for _ in range(by*2):
            r1 = random.randrange(-self.JIGGLE_STEP*1, self.JIGGLE_STEP*1+1)
            r2 = random.randrange(-self.JIGGLE_STEP, self.JIGGLE_STEP+1)

            if not self.quarantined:
                bl, br, bt, bb = self.model.bl, self.model.br, self.model.bt, self.model.bb
            else:
                bl, br, bt, bb = self.model.qbl, self.model.qbr, self.model.qbt, self.model.qbb

            # Border collision detection
            if self.x + r1*self.jiggle_rate - self.radius <= bl or self.x + r1*self.jiggle_rate + self.radius >= br:
                r1 = -r1*1.5
            if self.y + r2*self.jiggle_rate - self.radius <= bt or self.y + r2*self.jiggle_rate + self.radius >= bb:
                r2 = -r2*1.5

            self.x += r1*self.jiggle_rate
            self.y += r2*self.jiggle_rate

    def move(self):
        """Similar to the jiggle function, the only difference is that it uses a random angle instead"""

        angle = np.random.randint(0, 361)
        x = math.cos(angle)
        y = math.sin(angle)

        if not self.quarantined:
            bl, br, bt, bb = self.model.bl, self.model.br, self.model.bt, self.model.bb
        else:
            bl, br, bt, bb = self.model.qbl, self.model.qbr, self.model.qbt, self.model.qbb

        if self.x + x*self.SPEED - self.radius <= bl or self.x + x*self.SPEED + self.radius >= br:
            x = -x*1.5
        if self.y + y*self.SPEED - self.radius <= bt or self.y + y*self.SPEED + self.radius >= bb:
            y = -y*1.5

        self.x += x*self.SPEED
        self.y += y*self.SPEED

    @staticmethod
    def dist(p1: tuple or list, p2: tuple or list) -> float:
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def repel(self):
        """This function DOES NOT WORK.
        The intention was to repel nearby entities if the entities do social distancing
        """
        warn('This function(repel) does not work properly')
        if len(self.nearby) > 0:
            e_other = list(self.nearby.keys())[0]
            distance = self.dist((self.x, self.y), (e_other.x, e_other.y))
            print(distance, 'distance', self.SOCIAL_DISTANCING)
            if self.dist((self.x, self.y), (e_other.x, e_other.y)) <= self.infection_radius:
                factorx = self.infection_radius - abs(e_other.x - self.x)
                factory = self.infection_radius - abs(e_other.y - self.y)
                print(factorx, factory, 'factors', self.x, e_other.x, self.y, e_other.y, distance, self.infection_radius)
            else:
                factorx = 0
                factory = 0

            if e_other.x >= self.x:
                factorx = -factorx

            if e_other.y >= self.y:
                factory = -factory


            if self.x + factorx - self.radius > self.model.bl or self.x + factorx + self.radius < self.model.br:
                self.x += factorx

            if self.y + factory - self.radius > self.model.bt or self.y + factory + self.radius < self.model.bb:
                self.y += factory

    def __repr__(self):
        return f"Entity( {self.x}, {self.y}, {self.r}, {self.name} )"


class SIR:
    def __init__(self, win_size, bg=(0, 0, 40), name='SIR', padding=15):
        self.INFECTION_PROBABILITY = [0.2, 0.8]
        self.MOVING_PROBABILITY = [0.2, 0.8]
        self.QUARANTINE_PROBABILITY = 0.8
        self.DEATH_PCT = 0.03
        self.INFECTION_TIME = 0
        self.JIGGLE_FACTOR = 1
        self.JIGGLE_DISTRIBUTION = 1.0
        self.QUARANTINED_AFTER = 0
        self.QUARANTINE_LIMIT = 500
        self.GRAPH_QUEUE_LENGTH = 250
        self.MAX_GRAPH_HEIGHT = 240
        self.HEALTH_CAPACITY = 90  # Number of patients it can treat at a time
        self.START_QUARANTINE = 60000

        self.win_size = np.array(win_size)
        self.bg = bg
        self.name = name
        self.padding = padding

        self.entities = []
        self.currently_infected = 0
        self.total_infected = 0
        self.infected_entities = set()

        # Simulation rectangle borders
        self.bl = self.padding
        self.br = None
        self.bt = self.padding
        self.bb = None
        self.graph_queue = []

        # Quarantine rectangle borders
        self.quarantine_square_size = 70
        self.qbl, self.qbt = self.win_size - self.quarantine_square_size - self.padding
        self.qbr, self.qbb = self.win_size - self.padding
        self.quarantined_entities = 0
        self.total_quarantined = 0

        self.red = (255, 0, 0)
        self.black = (0, 0, 0)
        self.green = (0, 255, 0)
        self.golden = (242, 157, 1)
        self.pinky = (221, 80, 47)

        self.start_time = time.time()
        self.unit = 0.2
        self.moving_entities = []

        self.flags = []
        
        self.window = pygame.display.set_mode(self.win_size)
        pygame.display.set_caption(name)
        pygame.font.init()

    @property
    def infectious_entities(self):
        return [x for x in self.entities if x.infectious]

    def create_rect(self, color=(0, 0, 255), border=2):
        """Creates and draws the main simulation borders on the screen"""

        pxloc = (self.padding, self.padding)
        pxsize = (self.win_size[0] - self.padding*2 - 150, self.win_size[1] / 2)

        self.br, self.bb = pxsize

        pygame.draw.rect(self.window, color, [pxloc[0], pxloc[1], pxsize[0], pxsize[1]], border)

    @staticmethod
    def dist(p1: tuple or list, p2: tuple or list) -> float:
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def make_entities(self, quantity):
        """Makes entities in a uniform distribution in the main simulation area"""

        xs = np.random.uniform(self.bl + 5, self.br, quantity).astype(np.int32)
        ys = np.random.uniform(self.bt + 5, self.bb, quantity).astype(np.int32)
        self.entities = np.array([Entity(xs[i], ys[i], self, str(i)) for i in range(quantity)])

    def random_wonder(self):
        """Applies the jiggle funtion to each entity"""
        for e in self.entities:
            e.jiggle(self.JIGGLE_FACTOR)
            # print(e.jiggle_rate if e.quarantined else '', e.quarantined if e.quarantined else '')

    def set_jiggle(self):
        """Sets the jiggle rate of random entities of size according to the CONSTANT, lowering it"""
        size = int(self.JIGGLE_DISTRIBUTION*len(self.entities))
        # print(size)
        entities = self.entities.copy()
        random.shuffle(entities)
        entities = entities[:size]
        for e in entities:
            e.jiggle_rate = 0.1

    def random_wonder_2(self):
        """Applies the move function to each entity"""
        for e in self.entities:
            e.move()

    def proximity_counter(self):
        """For every entity, calculates if any other entity is in its infection radius, if so calculates the its time spent inside.
        This function wiil be used to infect other entities based on some time constraint."""
        for e in self.entities:
            if e.infectious is True:
                for e_other in self.entities:
                    if self.dist((e.x, e.y), (e_other.x, e_other.y)) <= Entity.infection_radius:
                        if e_other in e.nearby.keys():
                            e.nearby[e_other] = time.time() - self.start_time
                        else:
                            e.nearby[e_other] = 0

                    # Remove entities that are no longer in infection_radius and once were
                    # if e_other in e.nearby.keys() and self.dist((e.x, e.y), (e_other.x, e_other.y)) > Entity.infection_radius:
                    #     e.nearby.pop(e_other)

    def infection_counter(self):
        """Finds all the nearby entities for every entity, calculates the time spent in its infection radius, if its
        above a certain value and the other entity is not yet infected, infects it."""
        for e in self.entities:

            if e.infectious is True:

                for e_other in self.entities:

                    if e_other.name != e.name and e_other.removed is False and e_other in e.nearby:

                        if self.dist((e.x, e.y), (e_other.x, e_other.y)) < e.infection_radius:
                            if e_other in e.nearby.keys():
                                if e.nearby[e_other] >= self.unit:  # Run this only if the entity has spent a day with the infected entity
                                    if np.random.choice([0, 1], p=self.INFECTION_PROBABILITY) == 1:
                                        if not e_other.infected:
                                            self.currently_infected += 1
                                            self.total_infected += 1
                                            e.others_infected += 1
                                            e_other.infected_by.add(e)
                                            e_other.infectious = True
                                            e_other.infection_time = time.time()
                                            self.infected_entities.add(e_other)

                    # else:
                    #     if e in e_other.infected_by:
                    #         e_other.infected_by.remove(e)

    def social_distancing(self):
        """This function DOES NOT WORK.
        because of the repel function not working
        """
        warn('This function(repel) does not work properly')
        for e in self.entities:
            e.repel()
            e.jiggle_rate = 0

    def set_social_distancing_pct(self, to):
        """Used to set a certain distribution of the population to do social distancing"""
        k = len(self.entities) - int(to*len(self.entities))
        for e in random.choices(self.entities, k=k):
            e.social_distancer = False

    @staticmethod
    def infect(entity):
        """To manually infect an entity. Used at the start of the simulation"""
        entity.infectious = True
    
    def disinfect(self, quarantine=False):
        """When the infection period exceeds infection time this function is responsible for
        changing the entitiy's status from 'i' to 'r'.
        If the quarantine system is enabled, this function is also responsible for quarantining entities if they are to be quarantined. the conditions are:
        its infection period has exceeded quarantined after time, and its chosen in the random weighted selection"""
        to_remove = []

        if quarantine:
            to_quarantine = []
            if self.quarantined_entities < self.QUARANTINE_LIMIT:
                for inf_entity in self.infected_entities:
                    if inf_entity.infected_period >= self.QUARANTINED_AFTER and inf_entity.removed is False and inf_entity.quarantined is False:
                        choice = np.random.choice([0, 1], p=self.QUARANTINE_PROBABILITY)
                        if choice == 1:
                            to_quarantine.append(inf_entity)

            for e in to_quarantine:
                # self.infected_entities.remove(e)
                e.infectious = False
                # e.removed = True
                # self.currently_infected -= 1
                e.quarantined = True
                self.quarantined_entities += 1
                self.total_quarantined += 1
                e.x = random.randrange(self.qbl, self.qbr)
                e.y = random.randrange(self.qbt, self.qbb)
                e.jiggle_rate = 0.1

        for inf_entity in self.infected_entities:
            if inf_entity.infected_period >= self.INFECTION_TIME:
                to_remove.append(inf_entity)
                inf_entity.infected_by.clear()
                inf_entity.disinfection_time = time.time()
                inf_entity.removed = True
                inf_entity.infectious = False
                if inf_entity.quarantined:
                    inf_entity.quarantined = False
                    self.quarantined_entities -= 1
                    inf_entity.x = random.randrange(self.bl, self.br)
                    inf_entity.y = random.randrange(self.bt, self.bb)
                    inf_entity.jiggle_rate = 1

        for e in to_remove:
            self.infected_entities.remove(e)
            self.currently_infected -= 1

    def scale(self, y):
        """Scales the input to match the graph"""
        current_max_value = len(self.entities)
        pct = y / current_max_value
        scaled = pct * self.MAX_GRAPH_HEIGHT
        return scaled

    def graph(self, health_line=False, return_x=False):
        """Responsible for drawing the SIR graph by the use of graph queue which is handled by another function.
        It can also draw the health line on the graph and draw flags. the flags drawing do have a little bug in it though."""
        graph_step = 0
        w = 2
        y = self.win_size[1] - 20

        pygame.draw.rect(self.window, self.pinky,
                         [self.padding, y - self.MAX_GRAPH_HEIGHT, self.GRAPH_QUEUE_LENGTH*w, self.MAX_GRAPH_HEIGHT], 2)

        # Graph of the infectious
        for i, j, k in self.graph_queue:
            x = self.padding + graph_step * w
            pygame.draw.rect(self.window, (0, 255, 0), [x, y, w, -self.scale(k)-self.scale(i)+1])
            pygame.draw.rect(self.window, self.red, [x, y, w, -self.scale(i)+1])
            pygame.draw.rect(self.window, (128, 128, 128), [x, y - self.MAX_GRAPH_HEIGHT+1, w, self.scale(j)])

            graph_step += 1

        if health_line:
            self.write_text('health care system capacity', 16, self.golden,
                            (self.padding + self.GRAPH_QUEUE_LENGTH / 2, y - self.scale(self.HEALTH_CAPACITY) - 10))

            pygame.draw.line(self.window, self.golden, (self.padding, y - self.scale(self.HEALTH_CAPACITY)),
                             [w*self.GRAPH_QUEUE_LENGTH + self.padding, y - self.scale(self.HEALTH_CAPACITY)], 2)
        self.write_text(str(len(self.entities)), 20, self.pinky,
                        (self.GRAPH_QUEUE_LENGTH*w + self.padding + 5, y - self.MAX_GRAPH_HEIGHT + 5))

        self.write_text(str(0), 20, self.pinky,
                        (self.GRAPH_QUEUE_LENGTH * w + self.padding + 5, y - 10))

        #  Flags drawing
        for flag in self.flags:
            pygame.draw.line(self.window, self.pinky, (flag[1], y), (flag[1], y - 50), 3)
            self.write_text(flag[0], 16, self.golden, (flag[1], y - 50))

        if return_x:
            return self.padding + graph_step * w

    def graph_queue_handler(self):
        """This function gathers and puts all the values needed to daw the SIR graph in the graph queue."""
        ql = len(self.graph_queue)
        removed_entities_length = len([x for x in self.entities if x.removed])
        susceptible_entities_length = len([x for x in self.entities if x.removed is False and x.infectious is False])

        if ql < self.GRAPH_QUEUE_LENGTH:
            self.graph_queue.append([len(self.infectious_entities), removed_entities_length, susceptible_entities_length])
        elif ql >= self.GRAPH_QUEUE_LENGTH:
            self.graph_queue = self.graph_queue[ql - self.GRAPH_QUEUE_LENGTH + 1:]
            self.graph_queue.append([len(self.infectious_entities), removed_entities_length, susceptible_entities_length])

    def quarantine_handler(self):
        """This function just creates and draws the quarantine box and calls the disinfect function with quarantine=True.
        This is why either this function or disinfect will be called."""
        self.disinfect(quarantine=True)
        pygame.draw.rect(self.window, (0, 0, 255), [self.qbl, self.qbt,
                                                    self.quarantine_square_size, self.quarantine_square_size], 3)

    def R(self):
        """This function calculates the effective-reproductive-number (R) and returns it to be used by other functions."""
        others_infected = [x.others_infected for x in self.entities if x.infectious and x.infected_period > 0]
        infection_times = [x.infected_period for x in self.entities if x.infectious and x.infected_period > 0]
        time_remaining = [self.INFECTION_TIME - x for x in infection_times]
        infected_per_second = [x / y for x, y in zip(others_infected, infection_times)]
        estimated_infections = [z + x * y for x, y, z in zip(time_remaining, infected_per_second, others_infected)]
        if len(others_infected) > 0:
            return round(np.mean(estimated_infections), 2)
        else:
            return 0.0

    def central_location(self, quantity=1):
        """This function DOES NOT WORK.
        The intention was that entities will travel to a central location."""
        warn('This function(central_location) does not work properly')
        size = 50
        central_rect = [self.br / 2 - size/2, self.bb / 2 - size/2, size, size]
        pygame.draw.rect(self.window, (0, 0, 255), central_rect, 2)

        step = (int(self.start_time) - int(time.time())) % 2

        if step == 0.0 and len(self.moving_entities) < 5:
            for _ in range(quantity):
                e = random.choice([x for x in self.entities if x.quarantined is False])
                if e.quarantined is False:
                    choice = np.random.choice([0, 1], p=self.MOVING_PROBABILITY)
                    if choice == 1:
                        directions = [(x, y) for x, y in zip(np.random.normal(self.br/2, size=21),
                                                             np.random.normal(self.br/2, size=21))]
                        final_direction = directions[10]
                        e.jiggle_rate = 0
                        e.x += final_direction[0]
                        e.y += final_direction[1]
                        e.moving = True
        else:
            for e in self.moving_entities:
                e.moving = False
                e.jiggle_rate = 1
            self.moving_entities.clear()
    
    def write_text(self, text: str, size: int, color: tuple, loc: tuple):
        """This function writes text into the screen"""
        font = pygame.font.SysFont("Aerial", size)
        text = font.render(text, True, color)
        self.window.blit(text, loc)

    def render_entities(self, subsample=-1):
        """Renders all or a subsample of the entities to the screen, subsample option is availible to optimize the system."""
        if subsample == -1:
            for e in self.entities:
                # print(e.nearby)
                pygame.draw.ellipse(self.window, e.color, (e.x, e.y, Entity.radius, Entity.radius))
                # pygame.draw.circle(self.window, self.golden, (int(e.x), int(e.y)), Entity(0, 0, None, '').infection_radius, 1)
        else:
            for e in random.choices(self.entities, k=subsample):
                pygame.draw.ellipse(self.window, e.color, (e.x, e.y, Entity.radius, Entity.radius))

    def stats(self):
        """Writes all the statistics to the screen"""
        pop = len(self.entities)
        # casualties = int(self.DEATH_PCT*self.total_infected)
        infected = self.currently_infected
        total_infected = self.total_infected
        quarantined = len([x for x in self.entities if x.quarantined])
        r_val = self.R()


        if r_val > 1:
            status = 'Epidemic'
        elif r_val == 1:
            status = 'Endemic'
        else:
            status = 'On decline'

        texts = [
            f'Population: {pop}',
            f'Infected: {infected}',
            f'Total Infected: {total_infected}',
            f'In Quarantine: {quarantined}',
            f'R: {r_val}'
        ]

        sizes = [16 for _ in range(len(texts))]

        for i, t, s in zip(range(len(texts)), texts, sizes):
            self.write_text(t, s, self.golden, (self.bl + 100 * i, self.bb + 30))

        self.write_text(f'Spread status: {status}', 16, self.golden, (self.bl + 100 * len(texts) - 60, self.bb + 30))
        self.write_text(f'Total Quarantined: {self.total_quarantined}', 16, self.golden, (self.bl + 100 * len(texts) + 100, self.bb + 30))

    def set_hyperparameters(self, infection_time,  quarantined_after,
                            infection_prob, quarantine_prob, jiggle_factor,
                            health_care_cap, quarantine_limit, social_distancing_factor,
                            entity_radius=4, infection_radius=5, start_quarantine=60000, jiggle_distribution=1.0):
        """Sets all the Hyperparameters of the system. This function must be executed for the simulation to work."""
        self.INFECTION_TIME = round(self.unit * infection_time, 2)
        self.QUARANTINED_AFTER = round(self.unit * quarantined_after, 2)
        self.QUARANTINE_PROBABILITY = quarantine_prob
        self.INFECTION_PROBABILITY = infection_prob
        self.JIGGLE_FACTOR = jiggle_factor
        self.QUARANTINE_LIMIT = quarantine_limit
        Entity.set_radius(entity_radius)
        Entity.set_infection_radius(infection_radius)
        self.HEALTH_CAPACITY = health_care_cap
        self.set_social_distancing_pct(social_distancing_factor)
        self.START_QUARANTINE = start_quarantine
        self.JIGGLE_DISTRIBUTION = jiggle_distribution

    def display_hyperparameters(self):
        """This function simply writes all or some  of the hyperparametrs to the screen"""
        disease_status = 'Eradicated' if self.currently_infected == 0 else 'Present'
        texts = ['Hyperparameters',
                 f'1 day = {self.unit} seconds',
                 f'Infection prob. = {self.INFECTION_PROBABILITY[1]}',
                 f'Quarantined after {int(self.QUARANTINED_AFTER / self.unit) + 1} days',
                 f'Infection period is {int(self.INFECTION_TIME / self.unit)} days',
                 f"Infection radius = {Entity.infection_radius-Entity.radius} pixels",
                 f'Quarantined prob. is {self.QUARANTINE_PROBABILITY[1]}',
                 f'Time: {int(time.time() - self.start_time)} secs',
                 f'Disease status: {disease_status}'
                 ]

        sizes = [24] + [16 for _ in range(len(texts) - 1)]

        for i, t, s in zip(range(len(texts)), texts, sizes):
            self.write_text(t, s, self.golden, (self.br + 30, self.padding + 25*i))

    def event_loop(self):
        """This is where The simulation actually starts and ends, The event loop"""
        [self.infect(Entity.all_entities[f'{i}']) for i in range(3)]
        one_time = [0]
        self.start_time = time.time()
        self.set_jiggle()
        while True:
            clock.tick(60)
            time_now = time.time()
            self.window.fill(self.bg)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print('-'*20)
                    print('SIMULATION ENDED')
                    print('-'*20)
                    pygame.quit()
                    quit()

            self.create_rect()
            self.random_wonder_2()
            # self.social_distancing()
            self.proximity_counter()
            self.infection_counter()
            self.render_entities(subsample=-1)
            x = self.graph(health_line=False, return_x=True)
            self.graph_queue_handler()

            if time_now - self.start_time >= self.START_QUARANTINE and one_time[0] == 0:
                self.quarantine_handler()
                self.flags.append(('Quarantine enabled', x))
                one_time[0] = 1
            else:
                if one_time[0] == 1:
                    self.quarantine_handler()
                else:
                    self.disinfect()

            self.display_hyperparameters()
            self.stats()
            pygame.display.update()
            

if __name__ == '__main__':

    model = SIR((750, 600))
    model.create_rect()
    model.make_entities(300)
    model.set_hyperparameters(infection_time=14,
                              quarantined_after=3,
                              infection_prob=[0.7, 0.3],
                              quarantine_prob=[0.9, 0.1],
                              jiggle_factor=8,
                              health_care_cap=50,
                              quarantine_limit=1000,
                              social_distancing_factor=1.0,
                              entity_radius=6,
                              infection_radius=16,
                              start_quarantine=3,
                              jiggle_distribution=0.0)

    print('-'*20)
    print('STARTING SIMULATION')
    print('-'*20)
    print("Entities:", len(model.entities))

    model.event_loop()







