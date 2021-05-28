from pysine import sine
import numpy as np
import pickle
from enum import Enum
import pyglet
from gym.envs.classic_control import rendering
from gym.envs.classic_control.pendulum import PendulumEnv


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


def inverse_square_law_observer_receiver(obs_pos, src_pos, K=1.0, eps=0.0):
    """
    Computes the inverse square law for an observer receiver pair.
    Follows https://en.wikipedia.org/wiki/Inverse-square_law
    """
    distance = np.linalg.norm(obs_pos - src_pos)
    return K * 1.0 / (distance**2 + eps)


BOTTOM_MARGIN = -2.2
TOP_MARGIN = 2.2
LEFT_MARGIN = 2.2
RIGHT_MARGIN = -2.2


class CustomViewer(rendering.Viewer):
    def __init__(self, width, height, display=None):
        super().__init__(width, height, display=None)
        self.window = pyglet.window.Window(
            width=width, height=height, display=display, vsync=False)


class SoundReceiver(object):
    class Location(Enum):
        LEFT_BOTTOM = 1,
        LEFT_MIDDLE = 2,
        LEFT_TOP = 3,
        RIGHT_TOP = 4,
        RIGHT_MIDDLE = 5,
        RIGHT_BOTTOM = 6,
        MIDDLE_TOP = 7,
        MIDDLE_BOTTOM = 8

    def __init__(self, location):
        self.location = location

        if location == SoundReceiver.Location.LEFT_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, LEFT_MARGIN])
        elif location == SoundReceiver.Location.LEFT_MIDDLE:
            self.pos = np.array([0.0, LEFT_MARGIN])
        elif location == SoundReceiver.Location.LEFT_TOP:
            self.pos = np.array([TOP_MARGIN, LEFT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_TOP:
            self.pos = np.array([TOP_MARGIN, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_MIDDLE:
            self.pos = np.array([0.0, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.MIDDLE_TOP:
            self.pos = np.array([TOP_MARGIN, 0.0])
        elif location == SoundReceiver.Location.MIDDLE_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, 0.0])


class PendulumSound(PendulumEnv):
    """
    Frame:
    - points stored as (height, weight)
    - positive upwards and left
    Angular velocity:
    - positive is ccw
    """

    def __init__(
            self,
            original_frequency=440.,
            sound_vel=20.,
            sound_receivers=[SoundReceiver(SoundReceiver.Location.RIGHT_TOP)],
            debug=False):
        super().__init__()
        self.original_frequency = original_frequency
        self.sound_vel = sound_vel
        self.sound_receivers = sound_receivers
        self._debug = debug

        self.reset()

    def step(self, a):
        observation, reward, done, info = super().step(a)

        x, y, thdot = observation
        abs_src_vel = np.abs(thdot * 1)  # v = w . r
        # compute ccw perpendicular vector. if angular velocity is
        # negative, we reverse it. then multiply by absolute velocity
        src_vel = np.array([-y, x])
        src_vel = (
            src_vel / np.linalg.norm(src_vel)) * np.sign(thdot) * abs_src_vel
        src_pos = np.array([x, y])

        self._frequencies = [
            modified_doppler_effect(
                self.original_frequency,
                obs_pos=rec.pos,
                obs_vel=np.zeros(2),
                obs_speed=0.0,
                src_pos=src_pos,
                src_vel=src_vel,
                src_speed=np.linalg.norm(src_vel),
                sound_vel=self.sound_vel) for rec in self.sound_receivers
        ]
        self._amplitudes = [
            inverse_square_law_observer_receiver(
                obs_pos=rec.pos, src_pos=src_pos)
            for rec in self.sound_receivers
        ]
        sound_observation = list(zip(self._frequencies, self._amplitudes))

        img_observation = self.render(mode='rgb_array')

        if self._debug:
            self._debug_data['pos'].append(src_pos)
            self._debug_data['vel'].append(src_vel)
            self._debug_data['sound'].append(self._frequencies)

        return (img_observation, sound_observation), reward, done, info

    def render(self, mode='human', sound_channel=0, sound_duration=.1):
        if self.viewer is None:
            self.viewer = CustomViewer(100, 100)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)

        # only play sound in human mode
        if self._frequencies[sound_channel] and (mode == 'human'):
            sine(
                frequency=self._frequencies[sound_channel],
                duration=sound_duration)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def reset(self, num_initial_steps=1):
        observation = super().reset()

        if self._debug:
            self._debug_data = {
                'pos': [],
                'vel': [],
                'sound_receivers': [rec.pos for rec in self.sound_receivers],
                'sound': []
            }

        if type(num_initial_steps) is list or type(num_initial_steps) is tuple:
            assert len(num_initial_steps) == 2
            low = num_initial_steps[0]
            high = num_initial_steps[1]
            num_initial_steps = np.random.randint(low, high)
        elif type(num_initial_steps) is int:
            assert num_initial_steps >= 1
        else:
            raise 'Unsupported type for num_initial_steps. Either list/tuple or int'

        for _ in range(num_initial_steps):
            (observation, sound), _, _, _ = self.step(np.array([0.0]))

        return observation, sound

    def close(self, out=None):
        super().close()

        if out:
            with open(out, 'wb') as filehandle:
                pickle.dump(
                    self._debug_data,
                    filehandle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import argparse

    parser = argparse.ArgumentParser(description='PongSound debugger')
    parser.add_argument(
        '--file', type=str, required=True, help='File with debug data')
    args = parser.parse_args()

    COLORS = ['forestgreen', 'cornflowerblue', 'darkorange', 'm']

    # Load data
    debug = pickle.load(open(args.file, 'rb'))
    positions = np.vstack(debug['pos'])
    velocities = np.vstack(debug['vel'])
    sounds = np.vstack(debug['sound'])

    sound_receiver_positions = debug['sound_receivers']
    sound_receiver_positions = np.vstack(sound_receiver_positions)
    n_sound_receivers = sound_receiver_positions.shape[0]

    # Plots
    fig, _ = plt.subplots()

    # - Plot ball data
    ax = plt.subplot('311')
    plt.xlim(LEFT_MARGIN, RIGHT_MARGIN)
    plt.ylim(BOTTOM_MARGIN, TOP_MARGIN)

    # -- Plot ball position
    plt.scatter(positions[:, 1], positions[:, 0], s=3, c='k')
    ball_plot, = plt.plot(positions[0, 1], positions[0, 0], marker='o')

    # -- Plot ball velocity
    vel_arrow = plt.arrow(
        positions[0, 1],
        positions[0, 0],
        velocities[0, 1],
        velocities[0, 0],
        width=4e-2)

    # -- Plot ball to mic line
    src_mic_plots = []
    for sr in range(n_sound_receivers):
        p, = plt.plot([positions[0, 1], sound_receiver_positions[sr, 1]],
                      [positions[0, 0], sound_receiver_positions[sr, 0]],
                      c=COLORS[sr])
        src_mic_plots.append(p)

    time_slider = Slider(
        plt.axes([0.2, 0.05, 0.65, 0.03]),
        'timestep',
        0,
        len(debug['pos']) - 1,
        valinit=0,
        valstep=1)

    # - Plot sound data
    plt.subplot('312')
    sound_marker_plots = []
    for sr in range(n_sound_receivers):
        plt.plot(sounds[:, sr], c=COLORS[sr])
        p, = plt.plot(1, sounds[1, sr], marker='o')
        sound_marker_plots.append(p)

    plt.subplot('313')
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    plt.plot(speeds)
    speed_marker_plot, = plt.plot(0, speeds[0], marker='o')

    def update(_):
        nonlocal vel_arrow
        timestep = int(time_slider.val)

        ball_position = debug['pos'][timestep]
        ball_plot.set_data(ball_position[1], ball_position[0])

        for sr in range(n_sound_receivers):
            src_mic_plots[sr].set_data(
                [positions[timestep, 1], sound_receiver_positions[sr, 1]],
                [positions[timestep, 0], sound_receiver_positions[sr, 0]])

        vel_arrow.remove()
        vel_arrow = ax.arrow(
            positions[timestep, 1],
            positions[timestep, 0],
            velocities[timestep, 1],
            velocities[timestep, 0],
            width=4e-2)
        for sr in range(n_sound_receivers):
            sound_marker_plots[sr].set_data(timestep, sounds[timestep, sr])

        speed_marker_plot.set_data(timestep, speeds[timestep])

        fig.canvas.draw_idle()

    def arrow_key_image_control(event):
        if event.key == 'left':
            time_slider.set_val(max(time_slider.val - 1, time_slider.valmin))
        elif event.key == 'right':
            time_slider.set_val(min(time_slider.val + 1, time_slider.valmax))

        update(0)

    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    time_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
