import sys
import json

import flask

import logging

logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = flask.Flask(__name__)

state = {
    'episode': 0,
    'halite-to-gym': None,
    'gym-to-halite': None,
}


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'


@app.route('/reset', methods=['GET'])
def reset():
    state['episode'] = state['episode'] + 1
    state['gym-to-halite'] = None
    state['halite-to-gym'] = None
    return ''


@app.route('/inspect', methods=['GET'])
def inspect():
    return json.dumps(state)


@app.route('/halite-to-gym', methods=['GET', 'POST'])
def halite_to_gym():
    if flask.request.method == 'GET':
        episode = state['episode']
        while state['halite-to-gym'] is None:
            if episode != state['episode']:
                return ''
        value = state['halite-to-gym']
        state['halite-to-gym'] = None
        return value
    else:
        # assert state['halite-to-gym'] is None, 'halite-to-gym overwrote value'
        # print(f"state['halite-to-gym'] = {flask.request.data}, type={type(flask.request.data)}")
        state['halite-to-gym'] = flask.request.data
        return ''


@app.route('/gym-to-halite', methods=['GET', 'POST'])
def gym_to_halite():
    if flask.request.method == 'GET':
        episode = state['episode']
        while state['gym-to-halite'] is None:
            if episode != state['episode']:
                return ''
        value = state['gym-to-halite']
        state['gym-to-halite'] = None
        return value
    else:
        # assert state['gym-to-halite'] is None, 'gym-to-halite overwrote value'
        # print(f"state['gym-to-halite'] = {flask.request.data}, type={type(flask.request.data)}")
        state['gym-to-halite'] = flask.request.data
        return ''


@app.route('/kill', methods=['GET'])
def kill():
    sys.exit(0)


def main():
    app.run(debug=False, threaded=True)


if __name__ == '__main__':
    main()
