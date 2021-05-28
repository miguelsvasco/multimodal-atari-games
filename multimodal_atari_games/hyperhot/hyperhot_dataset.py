import os
import errno
from collections import deque
import numpy as np
import argparse
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import  multimodal_atari_games.hyperhot.hyperhot_env as hot


class FrameBuffer:
    def __init__(
            self,
            frames_per_state,
            preprocessor=lambda x: x,
    ):
        if frames_per_state <= 0:
            raise RuntimeError('Frames per state should be greater than 0')

        self.frames_per_state = frames_per_state
        self.samples = deque(maxlen=frames_per_state)
        self.preprocessor = preprocessor

    def append(self, sample):
        sample = self.preprocessor(sample)
        if len(self.samples) == 0:
            self.samples.extend(self.frames_per_state * [sample])
        self.samples.append(sample)

    def get_state(self):
        if len(self.samples) == 0:
            return None
        if self.frames_per_state == 1:
            return list(self.samples[0])
        else:
            return list(self.samples)

    def reset(self):
        self.samples.clear()

    def reset_and_append_new(self, sample):
        self.reset()
        self.append(sample)


class HyperhotDataset(Dataset):
    def __init__(self,
                 root,
                 generate=False,
                 n_samples=10000,
                 n_stack=2,
                 seed=0,
                 n_enemies=4,
                 pacifist_mode=False,
                 sound_receivers=['LEFT_SHIP', 'RIGHT_SHIP']):

        self._root = root
        self._n_stack = n_stack

        dataset_filename = _dataset_filename(n_samples, n_stack, n_enemies,
                                             pacifist_mode, sound_receivers)
        if generate:
            _generate(root, n_samples, n_stack, seed, n_enemies, pacifist_mode,
                      sound_receivers)

        if not self._check_exists(dataset_filename):
            raise RuntimeError(
                'Dataset not found. You can use generate=True to generate it')

        self._images, self._sounds, self._sound_limits = torch.load(
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

    @staticmethod
    def sound_normalization():
        return (-32767., 32767.)


def _dataset_filename(n_samples, n_stack, n_enemies, pacifist_mode,
                      sound_receivers):
    return '_'.join([
        f'hyperhot_ds_samples{n_samples}', f'stack{n_stack}',
        f'n_enemies{n_enemies}', f'pacifist_mode{pacifist_mode}',
        f'rec{str(sound_receivers)}.pt'
    ])


def preprocess(observation):

    # downsample
    processed_observation = observation[::2, ::2, :]
    # remove color
    processed_observation = processed_observation[:, :, 0]
    # Grey scale
    processed_observation = processed_observation / 255.
    # Cut top and bottom
    processed_observation = processed_observation[15:95, :]

    # Fixing semi-whites
    should_be_white_indexes = (processed_observation != 0.)
    processed_observation[should_be_white_indexes] = 1

    return processed_observation


def _random_action(observation, env):
    return env.action_space.sample()


def _generate(root, n_samples, n_stack, seed, n_enemies, pacifist_mode,
              sound_receivers):
    env = hot.HyperhotEnv(
        num_enemies=n_enemies,
        pacifist_mode=pacifist_mode,
        sound_receivers=[
            hot.SoundReceiver(hot.SoundReceiver.Location[ss])
            for ss in sound_receivers
        ])

    frame_buffer = FrameBuffer(
        n_stack,
        lambda observation: (preprocess(observation[0]), observation[1]))

    env.seed(seed)
    np.random.seed(seed)

    observation = env.reset()
    frame_buffer.reset_and_append_new(observation)
    images = []
    sounds = []
    for _ in tqdm(range(n_samples)):
        action = _random_action(observation, env)
        observation, _, done, _ = env.step(action)
        frame_buffer.append(observation)
        stacked_observations = frame_buffer.get_state()
        stacked_images, stacked_sounds = zip(*stacked_observations)
        stacked_images = np.stack(stacked_images)
        stacked_sounds = np.stack(stacked_sounds)

        images.append(stacked_images)
        sounds.append(stacked_sounds)

        if done:
            observation = env.reset()
            frame_buffer.reset_and_append_new(observation)

    images = np.stack(images)
    t_images = torch.from_numpy(images).float()

    sounds = np.stack(sounds)
    min_sound, max_sound = HyperhotDataset.sound_normalization()
    sounds = (sounds - min_sound) / (max_sound - min_sound)
    print(f'Sound normalization: ({min_sound}|{max_sound})')
    t_sounds = torch.from_numpy(sounds).float()

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
                _dataset_filename(n_samples, n_stack, n_enemies, pacifist_mode,
                                  sound_receivers)), 'wb') as f:
        torch.save((t_images, t_sounds, (min_sound, max_sound)), f)

    env.close()


def main():
    parser = argparse.ArgumentParser(description='Hyperhot dataset collector')
    parser.add_argument(
        '--root',
        type=str,
        default='./data',
        help='Dir where to store the dataset')
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10000,
        help='Number of samples to collect')
    parser.add_argument(
        '--n_stack', type=int, default=2, help='Number of images to stack')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument(
        '--sound_receivers',
        choices=[
            'BOTTOM', 'LEFT_BOTTOM', 'RIGHT_BOTTOM', 'BOTTOM_SHIP', 'TOP_SHIP',
            'CENTER_SHIP', 'LEFT_SHIP', 'RIGHT_SHIP'
        ],
        required=True,
        nargs='*',
        help='The sound receivers')
    parser.add_argument('--n_enemies', type=int, default=4)
    parser.add_argument(
        '--pacifist_mode',
        action='store_true',
        default=False,
        help='Enables Pacifist Mode')
    args = parser.parse_args()

    hh_ds = HyperhotDataset(
        args.root,
        generate=True,
        n_samples=args.n_samples,
        n_stack=args.n_stack,
        seed=args.seed,
        n_enemies=args.n_enemies,
        pacifist_mode=args.pacifist_mode,
        sound_receivers=args.sound_receivers)

    print("Finished =)")


if __name__ == '__main__':
    main()
