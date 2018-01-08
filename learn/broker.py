import sys

import flask

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = flask.Flask(__name__)

state = {
    'halite-to-gym': None,
    'gym-to-halite': None,
}


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'


@app.route('/halite-to-gym', methods=['GET', 'POST'])
def halite_to_gym():
    if flask.request.method == 'GET':
        while state['halite-to-gym'] is None:
            pass
        value = state['halite-to-gym']
        state['halite-to-gym'] = None
        return value
    else:
        #assert state['halite-to-gym'] is None, 'halite-to-gym overwrote value'
        #print(f"state['halite-to-gym'] = {flask.request.data}, type={type(flask.request.data)}")
        state['halite-to-gym'] = flask.request.data
        return ''


@app.route('/gym-to-halite', methods=['GET', 'POST'])
def gym_to_halite():
    if flask.request.method == 'GET':
        while state['gym-to-halite'] is None:
            pass
        value = state['gym-to-halite']
        state['gym-to-halite'] = None
        return value
    else:
        #assert state['gym-to-halite'] is None, 'gym-to-halite overwrote value'
        #print(f"state['gym-to-halite'] = {flask.request.data}, type={type(flask.request.data)}")
        state['gym-to-halite'] = flask.request.data
        return ''


@app.route('/reset', methods=['GET'])
def reset():
    state['gym-to-halite'] = None
    state['halite-to-gym'] = None
    return ''


@app.route('/kill', methods=['GET'])
def kill():
    sys.exit(0)


@app.route('/inspect', methods=['GET'])
def inspect():
    import json
    return json.dumps(state)


def main():
    app.run(debug=False, threaded=True)


if __name__ == '__main__':
    main()
