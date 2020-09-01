from keras.models import Sequential, load_model
from keras.layers import Dense
from .memory import ReplayMemory
from game.board import Board
import os
import numpy as np


class Agent:
    def __init__(self, load_previous=False):
        self.episodes = 300
        self.epsilon = 1
        self.decay = 1 / self.episodes
        self.gamma = 0.95

        self.model = load_model('trained_models/model17.h5')
        self.memory = ReplayMemory()

    def build_model(self):
        """
        The input to the model is a 4-dimensional vector with the encoded game state
        The output of the model is the predicted q-value associated with the game baord
        """
        model = Sequential()
        model.add(Dense(32, input_shape=(4,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def predict(self, board: Board):
        """
        This will return a prediction based on the current state of the game
        The prediction will be one of:
           random action if random val < epsilon
           best action if random val >= epsilon
        """

        possible_boards = board.get_next()
        possible_states = []
        for b in possible_boards:
            possible_states.append(Board.get_game_state(b))

        if len(possible_boards) == 0:
            return -1, np.zeros((11,))
        if np.random.random() < self.epsilon:
            m = np.random.randint(0, len(possible_boards))
            return possible_boards[m], possible_states[m]
        else:
            q_values = []
            for s in possible_states:
                s = np.reshape(s, (1, -1))
                q_values.append(self.model.predict(s)[0])
            m = np.argmax(q_values)
        return possible_boards[m], possible_states[m]

    def train(self):
        mem_size = len(self.memory.memories)
        if mem_size < self.memory.mem_size:
            return

        mems = self.memory.sample()
        punishment = -1

        next_states = [mem[0] for mem in mems]
        q_values = list()
        for state in next_states:
            state = np.reshape(state, (1, -1))
            q_values.append(self.model.predict(state)[0])

        inputs = list()
        targets = list()

        for i in range(len(mems)):
            current_state = mems[i][0]
            reward = mems[i][2]
            game_over = mems[i][3]

            q_prime = reward + self.gamma * q_values[i] if not game_over else punishment

            inputs.append(current_state)
            targets.append(q_prime)

        inputs = np.array(inputs)
        targets = np.array(targets)
        self.model.fit(x=inputs, y=targets, epochs=4, batch_size=64, verbose=0)

    def run(self, board: Board):
        """
        The model performs random actions until there is sufficient data in replay memory
        Then, the model will also train off replay memory after every action
        """

        sub = 0
        max_score = 0
        total = 0

        for e in range(self.episodes):
            print('Running episode', e)
            board.reset_board()
            
            game_over = False
            total_score = 0
            print('Memories', len(self.memory.memories))
            print('Epsilon', self.epsilon)
            while not game_over:
                current_state = Board.get_game_state(board._blocks)
                new_board, new_state = self.predict(board)

                reward = 0
                if new_board == -1:
                    game_over = True
                else:
                    reward, game_over = board.do_move(new_board)

                total_score += reward

                self.memory.remember((current_state, new_state, reward, game_over))

            print('Score', total_score)
            self.train()
            self.epsilon -= self.decay

            total += total_score
            max_score = total_score if total_score > max_score else max_score
            if total_score < 100:
                sub += 1
        
        # generate summary
        print('Max score', max_score)
        print('Avg score', total/self.episodes)
        print('Sub 100', sub)
        self.save()

    def save(self):
        filename = 'trained_models/model'
        ext = '.h5'
        n = 0
        while os.path.exists(filename + str(n) + ext):
            n += 1
        self.model.save(filename + str(n) + ext)
        print('Saved to', filename + str(n) + ext)
        pass

