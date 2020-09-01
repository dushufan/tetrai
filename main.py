import pyglet
from model.agent import Agent
from game.view import View
import sys

# define constants
WIDTH, HEIGHT = 300, 400


if __name__ == '__main__':
    screen = View(WIDTH, HEIGHT, 'Tetris')
    board = screen.board

    if len(sys.argv) == 1:
        # no previous model specified so train a new model
        _agent = Agent()
        _agent.run(board)
    else:
        # start training with the previous model
        model = sys.argv[1]
        screen.use_trained_agent(model)

    pyglet.app.run()
