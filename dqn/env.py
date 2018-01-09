import pickle
import subprocess
import multiprocessing

import requests
import numpy as np
import gym
import gym.spaces

import dqn
import hlt

broker_process = None  # keep global to avoid pickling issues


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
        except (requests.ReadTimeout, requests.ConnectionError):
            pass


class HaliteEnv(gym.Env):
    def __init__(self, broker):
        self.broker: Broker = broker
        self.turn = 0
        self.total_reward = 0
        self.state = None
        self.viewer = None
        self.last_map = None
        self.halite_command = ['./halite', '-d', '240 160', 'python3 MyLearningBot.py', './run_mlbot.sh']
        self.halite_process = None
        self.halite_logfile = None
        high = 1000 * np.ones((dqn.common.PLANET_MAX_NUM,))
        self.action_space = gym.spaces.Box(low=-high, high=high)
        self.action_space.n = len(high)
        # turn + planet properties
        obs_num = 1 + dqn.common.PLANET_MAX_NUM*dqn.common.PER_PLANET_FEATURES
        self.observation_space = gym.spaces.Box(low=-10, high=3000, shape=(obs_num,))

    def _reset(self):
        #print('reset')
        self.turn = 0
        self.total_reward = 0

        self.broker.kill()
        global broker_process
        if broker_process:
            broker_process.terminate()
        broker_process = multiprocessing.Process(target=dqn.broker.main)
        broker_process.start()
        while True:
            try:
                self.broker.ping()
                break
            except (requests.ConnectionError, requests.ReadTimeout):
                pass

        if self.halite_logfile:
            self.halite_logfile.close()
        self.halite_logfile = open('stdout-halite.log', 'w')
        self.halite_process = subprocess.Popen(self.halite_command, stdout=self.halite_logfile)

        self.state, self.last_map = self.broker.receive_state(timeout=100)
        return self.state

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

            win = False
            reward = -1
            if my_ships > ennemy_ships:
                win = True
                reward = 1

            info = {
                'win': win,
                'rewards': self.total_reward,
                'win_reward': reward,
                'turns': self.turn,
                'my_ships': my_ships,
                'ennemy_ships': ennemy_ships,
            }
            self.total_reward += reward
            print('episode:', info)
            return self.state, reward, True, info

        me = map.get_me()
        players = map.all_players()
        my_ships = me.all_ships()

        reward = -.0005

        self.total_reward += reward
        self.previous_map = map
        ennemy_ships = sum(len(player.all_ships()) for player in players if player != me)
        info = {
            'turn': self.turn,
            'reward': reward,
            'total_reward': self.total_reward,
            'my_ships': len(my_ships),
            'ennemy_ships': ennemy_ships,
        }
        return self.state, reward, False, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
