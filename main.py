import pyglet
from model.agent import Agent
from game.view import View
import sys

# define constants
WIDTH, HEIGHT = 300, 400


if __name__ == '__main__':
    screen = View(WIDTH, HEIGHT, 'Tetris')
    board = screen.board

    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            _agent = Agent()
            _agent.run(board)
    elif len(sys.argv) >= 3:
        model = sys.argv[2]
        if sys.argv[1] == 'play':
            screen.use_trained_agent(model)
        if sys.argv[1] == 'train':
            _agent = Agent(model_num=model)
            _agent.run(board)

    pyglet.app.run()
