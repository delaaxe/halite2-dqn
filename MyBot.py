import time

import hlt
import dqn.bot
import dqn.learn


class PlayingBot(dqn.bot.Bot):
    def __init__(self, name, model_path):
        self.turn = 0
        self.act = dqn.learn.ActWrapper.load(model_path)
        self.game = hlt.Game(name)
        self.tag = self.game.map.my_id
        #self.log('init super bot')
        #self.log('act loaded')

    def play(self):
        while True:
            self.turn += 1
            self.log(f"turn {self.turn}")
            map = self.game.update_map()
            start_time = time.time()

            features = self.produce_features(map)
            self.log(f"features {features}")
            action = self.act([features]).T
            self.log(f'received action {action}')

            ships_to_planets_assignment = self.produce_ships_to_planets_assignment(map, action)
            commands = self.produce_commands(map, ships_to_planets_assignment, start_time)
            self.log(f'commands: {commands}')

            self.game.send_command_queue(commands)

    def log(self, value):
        with open(f'stdout-QBot-{self.tag}.log', 'a') as fp:
            fp.write(value + '\n')
            fp.flush()


def main():
    try:
        bot = PlayingBot('QBot', 'dqn_model.pkl')
        bot.play()
    except:
        import traceback
        import random
        with open(f'qbot-stack-{random.randint(0, 100)}.txt', 'w') as fp:
            fp.write(traceback.format_exc())


if __name__ == '__main__':
    main()
