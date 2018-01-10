import time
import pickle
import pathlib
import datetime as dt

import requests

import hlt
import dqn.bot


class Broker:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def send_state(self, state):
        response = requests.post(f'{self.base_url}/halite-to-gym', data=pickle.dumps(state), timeout=1)
        assert response.status_code == requests.codes.ok

    def receive_action(self):
        response = requests.get(f'{self.base_url}/gym-to-halite', timeout=1)
        assert response.status_code == requests.codes.ok
        return pickle.loads(response.content)


class QLearningBot(dqn.bot.Bot):
    def __init__(self, name, broker):
        self.broker = broker
        self.turn = 0
        self.game = hlt.Game(name)
        tag = self.game.map.my_id
        self.log_file = pathlib.Path(f'stdout-{name}-{tag}.log')
        with self.log_file.open('w') as fp:
            fp.write('')
        self.log(f'Initialized bot {name} ({tag}) at {dt.datetime.now()}')

    def play(self):
        while True:
            self.turn += 1
            self.log(f"{dt.datetime.now()}: turn {self.turn}")
            map = self.game.update_map()
            self.log(f"{dt.datetime.now()}: map {map}")
            start_time = time.time()

            features = self.produce_features(map)

            self.log(f'{dt.datetime.now()}: sending features')
            self.broker.send_state((features, map))
            self.log(f'{dt.datetime.now()}: done, sending features, waiting action')
            action = self.broker.receive_action()
            self.log(f'{dt.datetime.now()}: received action {action}')

            ships_to_planets_assignment = self.produce_ships_to_planets_assignment(map, action)
            commands = self.produce_commands(map, ships_to_planets_assignment, start_time)
            self.log(f'{dt.datetime.now()}: commands: {commands}')

            self.log(f'{dt.datetime.now()}: sending...')
            self.game.send_command_queue(commands)
            self.log(f'{dt.datetime.now()}: sent commands')

    def log(self, value):
        with self.log_file.open('a') as fp:
            fp.write(value + '\n')
            fp.flush()


def main():
    try:
        broker = Broker(base_url='http://localhost:5000')
        bot = QLearningBot('QLearningBot', broker)
        bot.play()
    except:
        import traceback
        with open('exception-QLearningBot.log', 'w') as fp:
            fp.write(traceback.format_exc())


if __name__ == '__main__':
    main()
