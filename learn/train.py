import pickle
import itertools
import subprocess
import multiprocessing

import requests
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
import gym.spaces
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


import hlt
from learn import common, broker, graph_builder


class Broker:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def ping(self):
        response = requests.get(f'{self.base_url}/ping', )
        assert response.status_code == requests.codes.ok
        return response.content.decode()

    def send_action(self, action):
        response = requests.post(f'{self.base_url}/gym-to-halite', data=pickle.dumps(action), timeout=1)
        assert response.status_code == requests.codes.ok

    def receive_state(self, timeout=2):
        response = requests.get(f'{self.base_url}/halite-to-gym', timeout=timeout)
        assert response.status_code == requests.codes.ok
        return pickle.loads(response.content)

    def reset(self):
        response = requests.get(f'{self.base_url}/reset', timeout=1)
        assert response.status_code == requests.codes.ok

    def kill(self):
        try:
            requests.get(f'{self.base_url}/kill', timeout=.001)
        except requests.ReadTimeout:
            pass


class HaliteEnv(gym.Env):
    def __init__(self, broker):
        self.broker: Broker = broker
        self.halite_command = ['./halite', '-d', '240 160', 'python3 MyLearningBot.py', './run_mlbot.sh']
        self.halite_process = None
        high = 1000 * np.ones((common.PLANET_MAX_NUM,))
        low = -high
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.action_space.n = len(low)

        self.observation_space = gym.spaces.Box(low=-10, high=3000, shape=(common.PLANET_MAX_NUM*common.PER_PLANET_FEATURES,))

        self.turn = 0
        self.total_reward = 0
        self.viewer = None
        self.state = None
        self.last_map = None

    def _reset(self):
        #print('reset')
        self.state = None
        try:
            self.file.close()
        except:
            pass
        self.file = open('stdout-halite.log', 'w')

        try:
            self.broker.kill()
        except:
            pass

        try:
            self.mediator_process.terminate()
        except:
            pass

        self.mediator_process = multiprocessing.Process(target=broker.main)
        self.mediator_process.start()
        while True:
            try:
                self.broker.ping()
                break
            except (requests.ConnectionError, requests.ReadTimeout):
                pass

        self.halite_process = subprocess.Popen(self.halite_command, stdout=self.file)

        self.state, self.last_map = self.broker.receive_state(timeout=100)

        self.turn = 0
        self.total_reward = 0
        return np.array(self.state).ravel()

    def _step(self, action):
        #print('action', action)
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.turn += 1

        try:
            self.broker.send_action(action)
            self.state, map = self.broker.receive_state(timeout=2)
        except requests.ReadTimeout:
            me = self.previous_map.get_me()
            players = self.previous_map.all_players()
            my_ships = len(me.all_ships())
            ennemy_ships = sum(len(player.all_ships()) for player in players if player != me)

            reward = 0
            if my_ships > ennemy_ships:
                print('win', self.total_reward)
                reward = 500
            elif my_ships < ennemy_ships:
                reward = -500

            self.total_reward += reward
            #print('final turn:', self.turn, 'players:', len(players), 'my_ships:', my_ships, 'ennemy_ships:', ennemy_ships, 'reward:', reward)
            #print('episode reward:', self.total_reward, 'won:', reward > 0)
            return np.array(self.state).ravel(), 0, True, {}

        me = map.get_me()
        players = map.all_players()

        reward = 1
        for ship in me.all_ships():
            if ship.docking_status == hlt.entity.Ship.DockingStatus.DOCKED:
                reward += 1
            elif ship.docking_status == hlt.entity.Ship.DockingStatus.DOCKING:
                reward += .5

        self.total_reward += reward
        self.previous_map = map
        ennemy_ships = sum(len(player.all_ships()) for player in players if player != me)
        #print('turn:', self.turn, 'players:', len(players), 'my_ships:', len(me.all_ships()), 'ennemy_ships:', ennemy_ships, 'reward:', reward)
        return np.array(self.state).ravel(), reward, False, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=tf.nn.tanh)
        out = layers.softmax(out)
        return out


def main():
    print('main')
    with U.make_session(8):
        env = HaliteEnv(Broker('http://localhost:5000'))

        make_obs_ph = lambda name: U.BatchInput(env.observation_space.shape, name=name)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = graph_builder.build_train(
            make_obs_ph=make_obs_ph,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': model,
            'num_actions': env.action_space.n,
        }

        act = deepq.simple.ActWrapper(act, act_params)

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = False  #t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                pass
                # Show off the result
                #env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    actions = tf.argmax(actions, axis=1)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()
                    #act.save("halite_model.pkl")
                    #U.save_state('model.pkl')
                    #print('model saved')

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()


if __name__ == '__main__':
    main()
