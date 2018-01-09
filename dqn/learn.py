import os
import dill
import zipfile
import tempfile
import traceback
import itertools

import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

import dqn

from baselines.deepq.experiments import train_cartpole


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = dqn.graph.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    import tensorflow as tf
    import tensorflow.contrib.layers as layers
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=tf.nn.tanh)
        out = layers.softmax(out)
        return out


def main():
    print('main')
    with U.make_session(num_cpu=4):
        broker = dqn.env.Broker('http://localhost:5000')
        env = dqn.env.HaliteEnv(broker)

        def make_obs_ph(name):
            import baselines.common.tf_util as U
            return U.BatchInput(env.observation_space.shape, name=name)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = dqn.graph.build_train(
            make_obs_ph=make_obs_ph,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        act = ActWrapper(act, {
            'make_obs_ph': make_obs_ph,
            'q_func': model,
            'num_actions': env.action_space.n,
        })

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=30000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        learning_starts = 1000
        target_network_update_freq = 500
        checkpoint_freq = 20

        episode_rewards = [0.0]
        wins = [False]
        saved_mean_reward = None
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, info = env.step(action)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)
                wins.append(info['win'])

            win_rate = round(np.mean(wins[-100:]), 1)
            is_solved = t > 100 and win_rate >= 99
            if is_solved:
                print('solved')
                break
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > learning_starts:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    actions = np.argmax(actions, axis=1)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t > learning_starts and t % target_network_update_freq == 0:
                    update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and num_episodes % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", mean_100ep_reward)
                logger.record_tabular("mean win rate", win_rate)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if done and (t > learning_starts and num_episodes > 100 and num_episodes % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                               saved_mean_reward, mean_100ep_reward))
                    act.save('dqn_model.pkl')
                    xact = ActWrapper.load('dqn_model.pkl')
                    print('xxx', xact)
                    saved_mean_reward = mean_100ep_reward

    act.save('dqn_model.pkl')


if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()
        if dqn.env.broker_process:
            dqn.env.broker_process.terminate()
