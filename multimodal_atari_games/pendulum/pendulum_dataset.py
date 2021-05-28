import os
import errno
import pickle
import numpy as np
import argparse
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import multimodal_atari_games.pendulum.pendulum_env as ps


class PendulumSoundDataset(Dataset):
    def __init__(self,
                 root,
                 generate=False,
                 n_samples=10000,
                 n_stack=2,
                 seed=0,
                 original_frequency=440.,
                 sound_velocity=20.,
                 sound_receivers=['TOP_BOARD']):

        self._root = root
        self._n_stack = n_stack

        dataset_filename = _dataset_filename(n_samples, n_stack,
                                             original_frequency,
                                             sound_velocity, sound_receivers)
        if generate:
            _generate(root, n_samples, n_stack, seed, original_frequency,
                      sound_velocity, sound_receivers)

        if not self._check_exists(dataset_filename):
            raise RuntimeError(
                'Dataset not found. You can use generate=True to generate it')

        self._images, self._sounds, self._sound_normalization = torch.load(
            os.path.join(root, dataset_filename))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, sound)
        """
        img, sound = self._images[index], self._sounds[index]
        return img, sound

    def __len__(self):
        return len(self._images)

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self._root, filename))

    def sound_normalization(self):
        return self._sound_normalization

    def statistics(self):
        pendulums = np.zeros((60, 60))

        for img in tqdm(self._images):
            npimg = img.numpy()

            for stack in range(self._n_stack):
                h, w = np.where(npimg[stack, :, :] == 1.0)
                pendulums[h, w] += 1

        pendulums = pendulums / np.max(pendulums)

        import matplotlib.pyplot as plt
        plt.clf()
        plt.ylabel('h')
        plt.xlabel('w')
        plt.imshow(pendulums)
        plt.colorbar()
        plt.show()


def _dataset_filename(n_samples, n_stack, original_frequency, sound_velocity,
                      sound_receivers):
    return '_'.join([
        f'pendulum_ds_samples{n_samples}', f'stack{n_stack}',
        f'freq{original_frequency}', f'soundvel{sound_velocity}',
        f'rec{str(sound_receivers)}.pt'
    ])


def preprocess(observation):
    processed_observation = observation.copy()
    # crop
    processed_observation = processed_observation[20:80, 20:80]
    # downsample
    # processed_observation = processed_observation[::3, ::3, :]
    # remove color
    processed_observation = processed_observation[:, :, 0]

    # make rod white, backround black
    should_be_white_indexes = (processed_observation == 0) | (
        processed_observation == 204)
    should_be_black_indexes = (processed_observation == 255)
    processed_observation[should_be_white_indexes] = 1
    processed_observation[should_be_black_indexes] = 0

    return processed_observation


def _random_action(observation, env):
    return np.random.uniform(env.action_space.low, env.action_space.high)


def _collect_sample(n_stack, last_observation, env):
    samples = []

    while len(samples) != n_stack:
        for i in range(n_stack):
            action = _random_action(last_observation, env)
            (last_observation, sound), _, done, info = env.step(action)

            if done:
                samples = []
                if done:
                    last_observation, sound = env.reset()
                break

            observation = preprocess(last_observation)
            sound = np.array(sound)
            samples.append((observation, sound))

    return samples, last_observation


def _generate(root, n_samples, n_stack, seed, original_frequency,
              sound_velocity, sound_receivers):
    env = ps.PendulumSound(
        original_frequency=original_frequency,
        sound_vel=sound_velocity,
        sound_receivers=[
            ps.SoundReceiver(ps.SoundReceiver.Location[ss])
            for ss in sound_receivers
        ])

    env.seed(seed)
    np.random.seed(seed)

    last_observation = env.reset()
    images = []
    sounds = []
    for i in tqdm(range(n_samples)):
        sample, last_observation = _collect_sample(n_stack, last_observation,
                                                   env)
        sample = np.vstack(sample)

        stacked_images, stacked_sounds = zip(*sample)
        stacked_images = np.stack(stacked_images)
        stacked_sounds = np.stack(stacked_sounds)

        images.append(stacked_images)
        sounds.append(stacked_sounds)

    images = np.stack(images)
    t_images = torch.from_numpy(images).float()

    sounds = np.stack(sounds)

    # normalize frequencies
    max_freq, min_freq = np.max(sounds[:, :, :, 0]), np.min(sounds[:, :, :, 0])
    sounds[:, :, :, 0] = (sounds[:, :, :, 0] - min_freq) / (
        max_freq - min_freq)

    # normalize amplitudes
    max_amp, min_amp = np.max(sounds[:, :, :, 1]), np.min(sounds[:, :, :, 1])
    sounds[:, :, :, 1] = (sounds[:, :, :, 1] - min_amp) / (max_amp - min_amp)
    t_sounds = torch.from_numpy(sounds).float()

    sound_normalization_info = {
        'frequency': (min_freq, max_freq),
        'amplitude': (min_amp, max_amp)
    }

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    with open(
            os.path.join(
                root,
                _dataset_filename(n_samples, n_stack, original_frequency,
                                  sound_velocity, sound_receivers)),
            'wb') as f:
        torch.save((t_images, t_sounds, sound_normalization_info), f)

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description='PendulumSound dataset collector')
    parser.add_argument(
        '--root',
        type=str,
        default='./data',
        help='Dir where to store the dataset')
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of samples to collect')
    parser.add_argument(
        '--n_stack', type=int, default=2, help='Number of images to stack')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument(
        '--sound_receivers',
        choices=[
            'LEFT_BOTTOM', 'LEFT_MIDDLE', 'LEFT_TOP', 'RIGHT_TOP',
            'RIGHT_MIDDLE', 'RIGHT_BOTTOM', 'MIDDLE_TOP', 'MIDDLE_BOTTOM'
        ],
        required=True,
        nargs='*',
        help='The sound receivers')
    parser.add_argument('--original_frequency', type=float, default=440.)
    parser.add_argument('--sound_vel', type=float, default=20.)
    args = parser.parse_args()

    ds = PendulumSoundDataset(
        args.root,
        generate=True,
        n_samples=args.n_samples,
        n_stack=args.n_stack,
        seed=args.seed,
        original_frequency=args.original_frequency,
        sound_velocity=args.sound_vel,
        sound_receivers=args.sound_receivers)

    print('Saved. running stats.')
    ds.statistics()


if __name__ == '__main__':
    main()
