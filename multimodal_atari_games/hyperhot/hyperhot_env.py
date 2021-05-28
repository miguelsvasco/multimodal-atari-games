import gym
from gym import error, spaces, utils
import numpy as np
import random
import copy
import pygame
import time
from enum import Enum

note_frequency_dic = {
    'do': 261.63,
    'do#': 277.18,
    're': 293.66,
    're#': 311.13,
    'mi': 329.63,
    'fa': 349.23,
    'fa#': 369.99,
    'sol': 392.00,
    'sol#': 415.30,
    'la': 440.00,
    'la#': 466.16,
    'si': 493.88
}


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def square_wave_compute_value(i, period, amplitude):
    if i % period < period / 2.0:
        return amplitude
    else:
        return -amplitude


def sine_wave_compute_value(i, amplitude, frequency, sample_rate):
    return amplitude * np.sin(2.0 * np.pi * frequency * i / sample_rate)


def get_compound_wave(scaling_factor, wave_lst, observer_pos, observer_vel,
                      source_pos, source_vel, sample_length):

    # First generate the waves and added them together
    compound_wave = np.zeros(shape=sample_length)
    for i in range(len(wave_lst)):
        wave_smps = wave_lst[i].generate_wave_at_observer(
            scaling_factor=scaling_factor,
            observer_pos=observer_pos,
            observer_vel=observer_vel,
            source_pos=source_pos[i],
            source_vel=source_vel[i],
            sample_length=sample_length)
        compound_wave = compound_wave + wave_smps

    # Normalize compound wave to max 1 if amplitude is greater than 1
    if compound_wave.max() > 1.0 and compound_wave.min() < -1.0:
        compound_wave = rescale_linear(
            compound_wave, new_min=-1.0, new_max=1.0)
    elif compound_wave.max() > 1.0:
        compound_wave = rescale_linear(
            compound_wave, new_min=compound_wave.min(), new_max=1.0)
    elif compound_wave.min() < -1.0:
        compound_wave = rescale_linear(
            compound_wave, new_min=-1.0, new_max=compound_wave.max())

    # Generate samples
    compound_wave_samples = np.array(compound_wave * 32767).astype(np.int16)

    # Debug, play sound
    return compound_wave_samples


def modified_doppler_effect(freq, obs_pos, obs_vel, obs_speed, src_pos,
                            src_vel, src_speed, sound_vel):
    # Normalize velocity vectors to find their directions (zero values
    # have no direction).

    if not np.all(src_vel == 0):
        src_vel = src_vel / np.linalg.norm(src_vel)
    if not np.all(obs_vel == 0):
        obs_vel = obs_vel / np.linalg.norm(obs_vel)

    src_to_obs = obs_pos - src_pos
    obs_to_src = src_pos - obs_pos
    if not np.all(src_to_obs == 0):
        src_to_obs = src_to_obs / np.linalg.norm(src_to_obs)
    if not np.all(obs_to_src == 0):
        obs_to_src = obs_to_src / np.linalg.norm(obs_to_src)

    src_radial_vel = src_speed * src_vel.dot(src_to_obs)
    obs_radial_vel = obs_speed * obs_vel.dot(obs_to_src)

    fp = ((sound_vel + obs_radial_vel) / (sound_vel - src_radial_vel)) * freq

    return fp


class GameSound():
    def __init__(self,
                 note,
                 wave_type='square',
                 sample_rate=44100,
                 amplitude=1.0):
        self.note = note
        self.sample_rate = sample_rate
        self.wave_type = wave_type
        self.og_amplitude = amplitude
        self.amplitude = copy.deepcopy(self.og_amplitude)
        self.gamma = 0.025
        self.sound_vel = 20.
        self.duration = 1. / 60.

        # Frequency
        self.og_frequency = note_frequency_dic[note]
        self.frequency = copy.deepcopy(self.og_frequency)

        # Wave
        self.wave = self.generate_wave(amplitude=None, frequency=None)
        self.wave_idx = 0

    def reset(self):
        self.wave_idx = 0
        self.frequency = copy.deepcopy(self.og_frequency)
        self.amplitude = copy.deepcopy(self.og_amplitude)

    def generate_wave(self, amplitude, frequency, sample_length=None):

        if self.wave_type == 'square':
            return self.build_square_wave(
                amplitude=amplitude,
                frequency=frequency,
                sample_length=sample_length)
        elif self.wave_type == 'sine':
            return self.build_sine_wave(
                amplitude=amplitude,
                frequency=frequency,
                sample_length=sample_length)
        else:
            print("ERROR: Invalid Wave Type")
            return -1

    def compute_amplitude_distance(self, scaling_factor, source_pos,
                                   observer_pos):
        """

        We assume a function of type:

        a(x) = a_0/(\gamma * r^2 + 1)

        where:

        r   = distance from observer to source (we assume center positions)
        a_0 = amplitude
        gamma < 1 = A hyperparameter so that it doesn't dampen really really fast
        + 1 -> So that it doesn't blow up when r=0

        :param observer_pos:
        :return:
        """

        # Distance
        distance = np.sqrt(
            np.power(source_pos[0] - observer_pos[0], 2) + np.power(
                source_pos[1] - observer_pos[1], 2))

        # Amplitude
        if distance > scaling_factor:
            amplitude = self.og_amplitude / (
                np.sqrt(distance / scaling_factor))
        else:
            amplitude = self.og_amplitude

        # Just for checks
        if amplitude < 0.0:
            amplitude = 0.0

        return amplitude

    def compute_amplitude_distance_exp_decay(self, scaling_factor, source_pos,
                                             observer_pos):
        """

        We assume a function of type:

        a(x) = a_0*exp(-\gamma* r)

        where:

        r   = distance from observer to source (we assume center positions)
        a_0 = amplitude

        :param observer_pos:
        :return:
        """

        # Distance
        distance = np.sqrt(
            np.power(source_pos[0] - observer_pos[0], 2) + np.power(
                source_pos[1] - observer_pos[1], 2))

        # Amplitude
        amplitude = self.og_amplitude * np.exp(-self.gamma * distance)

        # print("Computing distance:")
        # print("Source Coordinates = " + str(source_pos[0]) + ", " + str(source_pos[1]))
        # print("Observer Coordinates = " + str(observer_pos[0]) + ", " + str(observer_pos[1]))
        # print("Distance = " + str(distance))
        # print("New amplitude = " + str(amplitude))

        # Just for checks
        if amplitude < 0.0:
            amplitude = 0.0

        return amplitude

    def generate_wave_at_observer(self, scaling_factor, observer_pos,
                                  observer_vel, source_pos, source_vel,
                                  sample_length):

        # First Compute Amplitude dampening with the observer distance
        amplitude = self.compute_amplitude_distance_exp_decay(
            scaling_factor=scaling_factor,
            source_pos=source_pos,
            observer_pos=observer_pos)
        frequency = None

        # # # Second, compute new frequency with Doppler Effect
        # if observer_vel[0] == 0 and observer_vel[1] == 0:
        #     frequency = modified_doppler_effect(self.og_frequency,
        #                                              obs_pos=observer_pos,
        #                                              obs_vel=np.zeros(2),
        #                                              obs_speed=0.0,
        #                                              src_pos=source_pos,
        #                                              src_vel=source_vel,
        #                                              src_speed=np.linalg.norm(source_vel),
        #                                              sound_vel=self.sound_vel)
        # else:
        #     frequency = modified_doppler_effect(self.og_frequency,
        #                                              obs_pos=observer_pos,
        #                                              obs_vel=observer_vel,
        #                                              obs_speed=np.linalg.norm(observer_vel),
        #                                              src_pos=source_pos,
        #                                              src_vel=source_vel,
        #                                              src_speed=np.linalg.norm(source_vel),
        #                                              sound_vel=self.sound_vel)
        # Now generate wave
        self.wave = self.generate_wave(
            amplitude=amplitude,
            frequency=frequency,
            sample_length=sample_length)
        return self.wave

    def build_square_wave(self,
                          amplitude=None,
                          frequency=None,
                          sample_length=None):

        if amplitude is None:
            amplitude = self.og_amplitude

        if frequency is None:
            frequency = self.og_frequency

        period = int(round(self.sample_rate / frequency))

        if sample_length is not None:
            wave = np.array([
                square_wave_compute_value(
                    i=x, period=period, amplitude=amplitude)
                for x in range(self.wave_idx, self.wave_idx + sample_length)
            ])
            self.wave_idx = (self.wave_idx + sample_length) % period
            return wave

        else:
            return np.array([
                square_wave_compute_value(
                    i=x, period=period, amplitude=amplitude)
                for x in range(0, period)
            ])

    def build_sine_wave(self,
                        amplitude=None,
                        frequency=None,
                        sample_length=None):

        if amplitude is None:
            amplitude = self.og_amplitude

        if frequency is None:
            frequency = self.og_frequency

        period = int(round(self.sample_rate / frequency))

        if sample_length is not None:
            wave = np.array([
                sine_wave_compute_value(
                    i=x,
                    amplitude=amplitude,
                    frequency=frequency,
                    sample_rate=self.sample_rate)
                for x in range(self.wave_idx, self.wave_idx + sample_length)
            ])
            self.wave_idx = (self.wave_idx + sample_length) % period
            return wave
        else:
            return np.array([
                sine_wave_compute_value(
                    i=x,
                    amplitude=amplitude,
                    frequency=frequency,
                    sample_rate=self.sample_rate) for x in range(0, period)
            ])


class SoundReceiver(object):
    class Location(Enum):
        BOTTOM = 1,
        LEFT_BOTTOM = 2,
        RIGHT_BOTTOM = 3,
        BOTTOM_SHIP = 4,
        TOP_SHIP = 5,
        CENTER_SHIP = 6,
        LEFT_SHIP = 7,
        RIGHT_SHIP = 8

    def __init__(self, location):
        self.location = location

        # Spatial variables
        self.pos = None
        self.prev_pos = None
        self.vel = None

        self.reset()

    def reset(self):
        self.pos = None
        self.prev_pos = None
        self.vel = np.array([0, 0])

    def update(self, state):
        self.prev_pos = self.pos

        if self.location == SoundReceiver.Location.BOTTOM:
            self.pos = np.array(
                [int(0.5 * state['game_width']), state['game_height']])
            self.vel = np.array([0.0, 0.0])
        elif self.location == SoundReceiver.Location.LEFT_BOTTOM:
            self.pos = np.array([0.0, state['game_height']])
            self.vel = np.array([0.0, 0.0])
        elif self.location == SoundReceiver.Location.RIGHT_BOTTOM:
            self.pos = np.array([state['game_width'], state['game_height']])
            self.vel = np.array([0.0, 0.0])
        elif self.location == SoundReceiver.Location.BOTTOM_SHIP:
            self.pos = np.array(state['player_rect'].midbottom)
            self.vel = state['player_vel']
        elif self.location == SoundReceiver.Location.TOP_SHIP:
            self.pos = np.array(state['player_rect'].midtop)
            self.vel = state['player_vel']
        elif self.location == SoundReceiver.Location.CENTER_SHIP:
            self.pos = np.array(state['player_rect'].center)
            self.vel = state['player_vel']
        elif self.location == SoundReceiver.Location.RIGHT_SHIP:
            self.pos = np.array(state['player_rect'].midright)
            self.vel = state['player_vel']
        elif self.location == SoundReceiver.Location.LEFT_SHIP:
            self.pos = np.array(state['player_rect'].midleft)
            self.vel = state['player_vel']
        else:
            print("Invalid Sound Receiver Location selected")

        return None

    def get_pos(self):
        return self.pos

    def get_vel(self):
        return self.vel


class Ship():
    def __init__(self, y, note, bullet_note):

        # self.image = pygame.image.load("imgs/ball1.png")
        # self.image = pygame.transform.scale(self.image, (20, 20))

        self.rect = pygame.Rect((0, 0, 24, 8))

        # self.rect = self.image.get_rect()

        # put ship bottom, center x
        self.rect.bottom = y - 22
        self.rect.centerx = random.randint(10, 130)

        # Movement variables
        self.move_x = 0
        self.speed = 5

        # Position Variables
        self.prev_pos = None
        self.current_pos = np.array(self.rect.center)
        self.current_vel = np.array([0.0, 0.0])

        # Sound variables
        self.note = note
        self.bullet_note = bullet_note
        self.sound = GameSound(note=note, wave_type='square')

        # Shooting variables
        self.shots = []
        self.shots_count = 0
        self.max_shots = 1

        # Alive variables
        self.is_alive = True

    def action(self, action):

        # Perform Action

        if action == 0:  # NOOP
            self.move_x = 0.0
        elif action == 1:  # LEFT
            self.move_x = -self.speed

        elif action == 2:  # RIGHT
            self.move_x = self.speed

        elif action == 3:  # SHOOT
            if len(self.shots) < self.max_shots:
                self.shots.append(
                    Bullet(
                        self.rect.centerx,
                        self.rect.top,
                        from_player=True,
                        bullet_note=self.bullet_note))
        elif action == 4:  # SHOOT_LEFT
            self.move_x = -self.speed
            if len(self.shots) < self.max_shots:
                self.shots.append(
                    Bullet(
                        self.rect.centerx,
                        self.rect.top,
                        from_player=True,
                        bullet_note=self.bullet_note))
        elif action == 5:  # SHOOT_RIGHT
            self.move_x = self.speed
            if len(self.shots) < self.max_shots:
                self.shots.append(
                    Bullet(
                        self.rect.centerx,
                        self.rect.top,
                        from_player=True,
                        bullet_note=self.bullet_note))

        # Update State
        self.update()

    def update(self):

        # Update position variables
        self.prev_pos = self.current_pos

        # Check if we are inside boundaries
        if self.rect.x < 10 and self.move_x == -self.speed:
            self.move_x = 0
        elif self.rect.x > 130 and self.move_x == self.speed:
            self.move_x = 0

        self.rect.x += self.move_x
        self.current_pos = np.array(self.rect.center)
        if self.prev_pos is not None:
            self.current_vel = self.current_pos - self.prev_pos

        # Delete old shots
        for i in range(len(self.shots) - 1, -1, -1):
            if not self.shots[i].is_alive:
                del self.shots[i]

        # Update shots
        for s in self.shots:
            s.update()

    def won(self, enemy_list):
        if np.sum(enemy_list) == 0:
            return True
        else:
            return False

    def died(self, enemy_list):
        for e in enemy_list:
            for e_s in e.shots:
                if pygame.sprite.collide_circle(self, e_s):
                    self.is_alive = False
                    e_s.is_alive = False
                    return True
        return False

    def miss_enemy(self):

        for s in self.shots:
            if s.rect.y < 0:
                s.is_alive = False
                return True
        return False

    def hit_enemy(self, enemy_list):

        for s in self.shots:
            for e in enemy_list:
                if pygame.sprite.collide_circle(s, e) and e.is_alive:
                    s.is_alive = False
                    e.is_alive = False
                    return True
        return False

    def bullet_collision(self, enemy_list):
        for s in self.shots:
            for e in enemy_list:

                # Check if we hit an enemy shot
                for e_s in e.shots:
                    if pygame.sprite.collide_circle(s, e_s):
                        s.is_alive = False
                        e_s.is_alive = False
                        return True
        return False

    def draw(self, screen):

        # screen.blit(self.rect, self.rect.topleft)
        pygame.draw.rect(screen, [231, 76, 60], self.rect, 0)

        for s in self.shots:
            s.draw(screen)
        return

    def get_sound(self):
        return self.sound

    def get_rect(self):
        return self.rect

    def get_shots(self):
        return self.shots

    def get_current_pos(self):
        return np.array(self.current_pos)

    def get_current_vel(self):
        return np.array(self.current_vel)


#----------------------------------------------------------------------


class Bullet():
    def __init__(self, x, y, from_player, bullet_note):

        self.is_alive = True
        self.from_player = from_player

        self.rect = pygame.Rect((0, 0, 6, 6))
        self.rect.centerx = x
        self.rect.centery = y

        # Position Variables
        self.prev_pos = None
        self.current_pos = np.array(self.rect.center)
        self.current_vel = np.array([0.0, 0.0])
        self.bullet_speed = 6

        # Sound variables
        self.bullet_note = bullet_note
        self.sound = GameSound(note=bullet_note, wave_type='square')

        # Color variable
        if from_player:
            self.color = [46, 204, 113]
        else:
            self.color = [52, 152, 219]
        # self.image = pygame.transform.scale(self.image, (6, 6))

        # self.rect = self.image.get_rect()
        # self.rect.centerx = x
        # self.rect.centery = y

    def update(self):

        # Update previous position variable
        self.prev_pos = self.current_pos

        # Move the rectangule
        if self.from_player:
            self.rect.y -= self.bullet_speed
        else:
            self.rect.y += self.bullet_speed

        # Update position and speed variable
        self.current_pos = np.array(self.rect.center)
        if self.prev_pos is not None:
            self.current_vel = self.current_pos - self.prev_pos

        # Check if it is out of bounds
        if self.rect.y < 0 or self.rect.y > 210:
            self.is_alive = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color,
                           (self.rect.centerx, self.rect.centery), 3)
        # screen.blit(self.image, self.rect.topleft)

    def get_sound(self):
        return self.sound

    def get_rect(self):
        return self.rect

    def get_current_pos(self):
        return np.array(self.current_pos)

    def get_current_vel(self):
        return np.array(self.current_vel)


#----------------------------------------------------------------------


class Enemy():
    def __init__(self,
                 x,
                 y,
                 id,
                 enemy_note,
                 enemy_bullet_note,
                 pacifist_mode,
                 move_right=True):

        # Variables
        self.id = id
        self.pacifist_mode = pacifist_mode

        # Build enemy
        # self.image = pygame.image.load("imgs/ball3.png")
        # self.image = pygame.transform.scale(self.image, (16, 16))
        # self.rect = self.image.get_rect()
        self.rect = pygame.Rect((0, 0, 16, 16))
        self.rect.centerx = x
        self.rect.centery = y

        # Shooting variables
        self.shots = []
        self.shots_count = 0
        self.max_shots = 1

        # Position Variables
        self.prev_pos = None
        self.current_pos = np.array(self.rect.center)
        self.current_vel = np.array([0.0, 0.0])

        # Sound variables
        self.note = enemy_note
        self.bullet_note = enemy_bullet_note
        self.sound = GameSound(note=enemy_note, wave_type='square')

        self.is_alive = True
        self.move_right = move_right

    def update(self):

        # Update previous position variable
        self.prev_pos = self.current_pos

        # Update position
        if self.move_right:
            self.rect.x += 1

        else:
            self.rect.x -= 1

        # Update position and speed variable
        self.current_pos = np.array(self.rect.center)
        if self.prev_pos is not None:
            self.current_vel = self.current_pos - self.prev_pos

        # Delete old shots
        for i in range(len(self.shots) - 1, -1, -1):
            if not self.shots[i].is_alive:
                del self.shots[i]

        # Update shots
        for s in self.shots:
            s.update()

        return

    def change_movement_side(self, move_right):
        self.move_right = move_right
        return

    def draw(self, screen):

        # Draw shots if it exists
        for s in self.shots:
            s.draw(screen)

        if self.is_alive:
            # screen.blit(self.image, self.rect.topleft)
            pygame.draw.circle(screen, [241, 196, 15],
                               (self.rect.centerx, self.rect.centery), 8)

    def shoot(self):
        if len(self.shots) < self.max_shots:
            self.shots.append(
                Bullet(
                    self.rect.centerx,
                    self.rect.bottom,
                    from_player=False,
                    bullet_note=self.bullet_note))

    def is_shooting(self):
        if len(self.shots) == 0:
            return False
        else:
            return True

    def get_shots(self):
        return self.shots

    def get_rect(self):
        return self.rect

    def get_current_pos(self):
        return np.array(self.current_pos)

    def get_current_vel(self):
        return np.array(self.current_vel)

    def get_sound(self):
        return self.sound


import pygame


class HyperhotEnv(gym.Env):
    """A top-down shooting game for OpenAI gym"""

    metadata = {
        'render.modes': ['human', 'console'],
        'video.frames_per_second': 30
    }

    def __init__(self,
                 num_enemies,
                 sound_receivers=[
                     SoundReceiver(SoundReceiver.Location.LEFT_SHIP),
                     SoundReceiver(SoundReceiver.Location.RIGHT_SHIP)
                 ],
                 time_limit=20,
                 play_sound=True,
                 pacifist_mode=False):

        # User Variables
        self.num_enemies = num_enemies
        self.pacifist_mode = pacifist_mode
        self.sound_receivers = sound_receivers
        self.scaling_factor = 10  # 10 px = 1 m
        self.time_limit = time_limit
        self.time_limit_frames = time_limit * 30
        self.time_limit_frame_counter = 0

        # Viewer and Sound Renderer
        if len(sound_receivers) == 1:
            pygame.mixer.pre_init(31400, -16, 1, 64)
        elif len(sound_receivers) == 2:
            pygame.mixer.pre_init(31400, -16, 2, 64)
        else:
            pygame.mixer.pre_init(31400, -16, 1,
                                  64)  # Only supports up to 2 sound channels
        pygame.init()
        self.screen_width, self.screen_height = 160, 210
        self.screen = pygame.display.set_mode((round(self.screen_width), round(
            self.screen_height)))
        self.border_threshold = 5
        pygame.mixer.music.set_volume(0.1)

        # Action Set and Space
        self.action_set_dic = {
            0: 'NOOP',
            1: 'LEFT',
            2: 'RIGHT',
            3: 'SHOOT',
            4: 'SHOOT_LEFT',
            5: 'SHOOT_RIGHT'
        }
        self._action_set = np.array([0, 1, 2, 3, 4, 5])
        self.action_space = spaces.Discrete(len(self._action_set))

        # Observation Variables
        self.sound_set_dic = {
            'ship': 'do',
            'ship_bullet': 'la#',
            'enemy': {
                0: 'do',
                1: 'do',
                2: 'mi',
                3: 'mi'
            },
            'enemy_bullet': 'sol'
        }

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 3),
            dtype=np.uint8)

        # Reward space
        self.rewards = {
            "hit_enemy": 0.0,
            "miss_enemy": 0.,
            "alive": 0.,
            "win": 10.,
            "lost": -1.,
            'times_up': -1.
        }

        # Game Variables
        self.score = 0
        self.clock = None
        self.ship = None
        self.enemies = []
        self.alive_enemies = np.ones(num_enemies)
        self.max_enemy_shots = 4
        self.init_pos_enemy_dic = {
            1: [80],
            2: [80 - 8 - 16, 80 + 8 + 16],
            3: [80 - 8 - 16 - 8 - 16, 80, 80 + 8 + 16 + 8 + 16],
            4: [
                80 - 8 - 16 - 8 - 16, 80 - 8 - 8, 80 + 8 + 8,
                80 + 8 + 16 + 8 + 16
            ]
        }

        # Sound variables
        self.play_sound = play_sound
        self.sound = None
        self.sample_length = 1047

        # User Events
        self.ticks_count = 0
        self.shooting_period = 6
        # self.ENEMY_SHOOTS = pygame.USEREVENT
        # pygame.time.set_timer(self.ENEMY_SHOOTS, 50)

    def step(self, action):

        # Update the clock
        self.ticks_count += 1
        self.time_limit_frame_counter += 1

        # Get action
        action = self._action_set[action]

        # Update ship
        self.ship.action(action)

        # Update enemies
        self.update_enemies()

        # Update sound
        self.update_sound()

        # Update drawings
        self.update_drawings()

        # Get reward
        reward, done = self.get_reward()

        # Get observation
        obs = self._get_observation()
        sound_obs = self._get_sound_observation()

        return (obs, sound_obs), reward, done, {}

    def reset(self, **kwargs):

        # Reset
        self.score = 0
        self.ticks_count = 0
        self.time_limit_frame_counter = 0

        # Create characters
        self.ship = Ship(
            y=self.screen_height,
            note=self.sound_set_dic['ship'],
            bullet_note=self.sound_set_dic['ship_bullet'])
        self.enemies = []

        # Randomize intial position
        if np.random.uniform() >= 0.5:
            move_right = False
        else:
            move_right = True

        for i in range(self.num_enemies):
            self.enemies.append(
                Enemy(
                    x=self.init_pos_enemy_dic[self.num_enemies][i],
                    y=40,
                    id=i,
                    enemy_note=self.sound_set_dic['enemy'][i],
                    enemy_bullet_note=self.sound_set_dic['enemy_bullet'],
                    pacifist_mode=self.pacifist_mode,
                    move_right=move_right))

        # Game Variables
        self.clock = pygame.time.Clock()
        self.alive_enemies = np.ones(self.num_enemies)

        # Update sound
        self.update_sound()

        # Update drawings
        self.update_drawings()

        # Get observation
        obs = self._get_observation()
        sound_obs = self._get_sound_observation()
        return (obs, sound_obs)

    def render(self, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """
        if mode == 'console':
            print(self._get_game_state)
        elif mode == "human":
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode((round(
                        self.screen_width), round(self.screen_height)))

                # self.screen.fill((0, 0, 0))
                #
                # self.ship.draw(self.screen)
                #
                # for e in self.enemies:
                #     e.draw(self.screen)

                # Update clock
                # self.clock.tick(30)

                pygame.display.update()

                # Debug, play sound
                # if self.play_sound:
                #     if len(self.sound_receivers) == 1:  # Mono Channel
                #
                #         channel_m= pygame.mixer.Channel(0)
                #         channel_m.set_volume(0.1, 0.1)
                #         game_sound_m = pygame.mixer.Sound(self.sound[0].astype(np.int16))
                #         channel_m.play(game_sound_m)
                #
                #     elif len(self.sound_receivers) == 2:    # Stereo Channels
                #
                #         channel_l = pygame.mixer.Channel(0)     # Left Channel
                #         channel_l.set_volume(0.1, 0.0)
                #
                #         channel_r = pygame.mixer.Channel(1)     # Right Channel
                #         channel_r.set_volume(0.0, 0.1)
                #         game_sound_l = pygame.mixer.Sound(self.sound[0].astype(np.int16))
                #         game_sound_r = pygame.mixer.Sound(self.sound[1].astype(np.int16))
                #
                #         channel_l.play(game_sound_l)
                #         channel_r.play(game_sound_r)
                #
                #     while pygame.mixer.get_busy():
                #         continue
                return

        elif mode == "human_playing":
            if close:
                pygame.quit()
            else:
                # Update clock
                self.clock.tick(30)

                pygame.display.update()

                # Debug, play sound
                if self.play_sound:
                    if len(self.sound_receivers) == 1:  # Mono Channel

                        channel_m = pygame.mixer.Channel(0)
                        channel_m.set_volume(0.1, 0.1)
                        game_sound_m = pygame.mixer.Sound(
                            self.sound[0].astype(np.int16))
                        channel_m.play(game_sound_m)

                    elif len(self.sound_receivers) == 2:  # Stereo Channels

                        channel_l = pygame.mixer.Channel(0)  # Left Channel
                        channel_l.set_volume(0.1, 0.0)

                        channel_r = pygame.mixer.Channel(1)  # Right Channel
                        channel_r.set_volume(0.0, 0.1)
                        game_sound_l = pygame.mixer.Sound(
                            self.sound[0].astype(np.int16))
                        game_sound_r = pygame.mixer.Sound(
                            self.sound[1].astype(np.int16))

                        channel_l.play(game_sound_l)
                        channel_r.play(game_sound_r)

                    while pygame.mixer.get_busy():
                        continue
                return

        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def update_drawings(self):

        self.screen.fill((0, 0, 0))

        self.ship.draw(self.screen)

        for e in self.enemies:
            e.draw(self.screen)

        return

    def get_reward(self):

        # Init reward
        reward = 0.0
        reset_game = False

        if self.time_limit_frame_counter >= self.time_limit_frames:
            reward += self.rewards['times_up']
            reset_game = True
            return reward, reset_game

        # Check if we died or if we continue playing
        if self.ship.died(enemy_list=self.enemies):
            reward += self.rewards['lost']
            reset_game = True
            return reward, reset_game

        # Check if we finished the game
        elif self.ship.won(enemy_list=self.alive_enemies):
            reward += self.rewards['win']
            reset_game = True
            return reward, reset_game

        # Player is still alive but nothing happened
        else:
            reward += self.rewards['alive']

            # Check for bullet colision
            self.ship.bullet_collision(enemy_list=self.enemies)

            # Check if we hit something
            if self.ship.hit_enemy(enemy_list=self.enemies):
                reward += self.rewards['hit_enemy']

            # Check if player shot miss the enemies
            if self.ship.miss_enemy():
                reward += self.rewards['miss_enemy']

            return reward, reset_game

    def update_enemies(self):

        # Delete dead enemies
        for i in range(len(self.enemies) - 1, -1, -1):
            if not self.enemies[i].is_alive:
                self.alive_enemies[i] = 0

        # Check if all enemies are dead
        if np.sum(self.alive_enemies) == 0:
            return

        # Check if enemy shoots
        # for event in pygame.event.get():
        #     if event.type == self.ENEMY_SHOOTS and not self.pacifist_mode:
        if self.ticks_count % self.shooting_period == 0:

            # Reset ticks_count
            self.ticks_count = 0

            total_shots = 0
            shottable_enemies = []

            for j in range(len(self.enemies)):
                if self.enemies[j].is_shooting():
                    total_shots += len(self.enemies[j].get_shots())

                if self.alive_enemies[j] == 1 and len(
                        self.enemies[j]
                        .get_shots()) < self.enemies[j].max_shots:
                    shottable_enemies.append(j)
                else:
                    continue

            if total_shots < self.max_enemy_shots and len(
                    shottable_enemies) > 0:
                shooting_enemy_idx = np.random.choice(shottable_enemies)
                self.enemies[shooting_enemy_idx].shoot()

        # Check if we need to change motion direction
        alive_enemies = []
        for i in range(len(self.enemies)):
            if self.alive_enemies[i] == 1:
                alive_enemies.append(self.enemies[i])

        e_left, e_right = alive_enemies[0], alive_enemies[-1]

        if e_left.rect.x <= 0 + self.border_threshold:
            for e in self.enemies:
                e.change_movement_side(move_right=True)

        if e_right.rect.x + 16 >= self.screen_width - self.border_threshold:
            for e in self.enemies:
                e.change_movement_side(move_right=False)

        # Update enemies
        for e in self.enemies:
            e.update()

        return

    def update_sound(self):

        self.sound = np.zeros(
            shape=[len(self.sound_receivers), self.sample_length])

        # Get state
        state = self._get_state()

        # For each sound_receiver
        for i in range(len(self.sound_receivers)):

            source_waves = []
            source_poss = []
            source_vels = []

            # Update Sound Receiver
            self.sound_receivers[i].update(state=state)

            # Get observer position and velocity
            obs_pos, obs_vel = self.sound_receivers[
                i].get_pos(), self.sound_receivers[i].get_vel()

            # Generate sound waves
            # For the ship's bullets
            for s in self.ship.get_shots():
                source_waves.append(s.get_sound())
                source_poss.append(s.get_current_pos())
                source_vels.append(s.get_current_vel())

            # For the enemies and their bullets
            for j in range(len(self.enemies)):
                if self.alive_enemies[j] == 1:
                    source_waves.append(self.enemies[j].get_sound())
                    source_poss.append(self.enemies[j].get_current_pos())
                    source_vels.append(self.enemies[j].get_current_vel())

                for en_s in self.enemies[j].get_shots():
                    source_waves.append(en_s.get_sound())
                    source_poss.append(en_s.get_current_pos())
                    source_vels.append(en_s.get_current_vel())

            # Generate Compound Wave
            sound_receiver_wave = get_compound_wave(
                scaling_factor=self.scaling_factor,
                wave_lst=source_waves,
                observer_pos=obs_pos,
                observer_vel=obs_vel,
                source_pos=source_poss,
                source_vel=source_vels,
                sample_length=self.sample_length)

            self.sound[i, :] = sound_receiver_wave

        return

    def _get_observation(self):
        """
        Returns the current game screen in RGB format.
        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).
        """
        return np.fliplr(
            np.rot90(
                pygame.surfarray.array3d(self.screen).astype(np.uint8), 3))

    def _get_sound_observation(self):

        return self.sound

    def _get_state(self):
        """
        Gets a non-visual state representation of the game.
        """

        # Get rectangle of ship's shots
        ship_shots_rect = []
        for s in self.ship.get_shots():
            ship_shots_rect.append(s.get_rect())

        # Get position of enemies (still alive)
        enemy_pos = []
        enemy_vel = []
        enemy_shots_pos = []
        enemy_shots_vel = []
        for i in range(len(self.enemies)):
            if self.alive_enemies[i] == 1:
                enemy_pos.append(self.enemies[i].get_current_pos())
                enemy_vel.append(self.enemies[i].get_current_vel())
            enemy_shots = self.enemies[i].get_shots()
            for shot in enemy_shots:
                enemy_shots_pos.append(shot.get_current_pos())
                enemy_shots_vel.append(shot.get_current_vel())

        state = {
            'game_height': self.screen_height,
            'game_width': self.screen_width,
            "player_rect": self.ship.get_rect(),
            'player_vel': self.ship.get_current_vel(),
            "player_shots": ship_shots_rect,
            "alive_enemies_pos": enemy_pos,
            "alive_enemies_vel": enemy_vel,
            "enemy_shots_pos": enemy_shots_pos,
            "enemy_shots_vel": enemy_shots_vel
        }

        return state


if __name__ == "__main__":
    env = gym.make()
