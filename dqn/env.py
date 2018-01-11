import os
import pickle
import random
import subprocess
import datetime as dt
import multiprocessing

import requests
import requests.adapters
import numpy as np
import gym
import gym.spaces

import hlt
import dqn.common
import dqn.broker

broker_process = None  # keep global to avoid pickling issues


class Broker:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        #self.session.mount('http://localhost', requests.adapters.HTTPAdapter(max_retries=10))

    def ping(self):
        response = self.session.get(f'{self.base_url}/ping', )
        assert response.status_code == requests.codes.ok
        return response.content.decode()

    def send_action(self, action):
        response = self.session.post(f'{self.base_url}/gym-to-halite', data=pickle.dumps(action), timeout=1)
        assert response.status_code == requests.codes.ok

    def receive_state(self, timeout=2):
        response = self.session.get(f'{self.base_url}/halite-to-gym', timeout=timeout)
        assert response.status_code == requests.codes.ok
        return pickle.loads(response.content)

    def reset(self):
        response = self.session.get(f'{self.base_url}/reset', timeout=1)
        assert response.status_code == requests.codes.ok

    def kill(self):
        try:
            self.session.get(f'{self.base_url}/kill', timeout=.001)
        except (requests.ReadTimeout, requests.ConnectionError):
            pass


class HaliteEnv(gym.Env):
    def __init__(self, broker):
        self.state = None
        self.viewer = None
        self.total_reward = 0
        high = 1000 * np.ones((dqn.common.PLANET_MAX_NUM,))
        self.action_space = gym.spaces.Box(low=-high, high=high)
        self.action_space.n = len(high)
        # turn + planet properties
        obs_num = 1 + dqn.common.PLANET_MAX_NUM * dqn.common.PER_PLANET_FEATURES
        self.observation_space = gym.spaces.Box(low=-10, high=3000, shape=(obs_num,))

        self.broker: Broker = broker
        self.halite_process = None
        self.halite_logfile = None
        self.last_reset = None

        self.turn = 0
        self.last_map = None

    def _reset(self):
        #print(f'{dt.datetime.now()}: reset')
        self.turn = 0
        self.total_reward = 0
        self.highest_planet_count = 0

        global broker_process
        if not broker_process:
            broker_process = multiprocessing.Process(target=dqn.broker.main)
            broker_process.start()
            while True:
                try:
                    self.broker.ping()
                    break
                except (requests.ConnectionError, requests.ReadTimeout):
                    pass
        self.broker.reset()

        if self.halite_logfile:
            self.halite_logfile.close()
        self.halite_logfile = open('stdout-halite.log', 'w')
        command = self.make_halite_command()
        self.halite_process = subprocess.Popen(command, stdout=self.halite_logfile)

        self.state, self.last_map = self.broker.receive_state(timeout=100)
        self.last_step = dt.datetime.now()
        return self.state

    def _step(self, action):
        #print(f'{dt.datetime.now()}: step (last took {dt.datetime.now() - self.last_step})')
        self.last_step = dt.datetime.now()

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.turn += 1

        try:
            self.broker.send_action(action)
            self.state, map = self.broker.receive_state(timeout=2)
        except (requests.ReadTimeout, requests.ConnectionError):
            me = self.previous_map.get_me()
            players = self.previous_map.all_players()
            my_ships_count = len(me.all_ships())
            enemy_ships_count = sum(len(player.all_ships()) for player in players if player != me)

            win = False
            reward = -1
            if my_ships_count > enemy_ships_count:
                win = True
                reward = 1

            info = {
                'win': win,
                'turns': self.turn,
                #'my_ships': my_ships_count,
                #'enemy_ships': enemy_ships_count,
                'highest_planet_count': self.highest_planet_count,
            }
            self.total_reward += reward
            return self.state, reward, True, info

        map: hlt.game_map.Map
        me = map.get_me()
        players = map.all_players()
        my_planets_count = len([planet for planet in map.all_planets() if planet.owner == me])
        my_ships = me.all_ships()

        reward = -.0005
        if my_planets_count > self.highest_planet_count:
            self.highest_planet_count = my_planets_count
            reward = .001

        self.total_reward += reward
        self.previous_map = map
        enemy_ships_count = sum(len(player.all_ships()) for player in players if player != me)
        info = {
            'turn': self.turn,
            'reward': reward,
            'total_reward': self.total_reward,
            'my_planets': my_planets_count,
            'my_ships': len(my_ships),
            'enemy_ships': enemy_ships_count,
        }
        return self.state, reward, False, info

    def _render(self, mode='human', close=False):
        pass  # just watch replay

    def _close(self):
        if self.halite_logfile:
            self.halite_logfile.close()
        if self.broker:
            self.broker.kill()
        if broker_process:
            broker_process.terminate()

    def make_halite_command(self):
        if os.name == 'nt':
            command = ['.\halite.exe']
        else:
            command = ['./halite']
        width = random.randint(260, 384)
        if width % 2 == 1:
            width += 1
        height = int(width * 2/3)
        command += ['--timeout', '--dimensions', f'{width} {height}']
        command += ['python MyQLearningBot.py', 'python MyMLStarterBot.py']
        if random.randint(0, 1):
            command += ['python MyMLStarterBot.py', 'python MyMLStarterBot.py']
        return command
