# Tetris AI with deep Q learning

tetrai is a tetris game created using pyglet. tetrai uses reinforcement learning to train a model that consistently clears rows.

## Model

The model consists of two 32 neuron dense layers and one output neuron that predicts the q-value for a board based on an encoded game state. The game state is represented using a mixture of raw board data (height of each column or just the entire 10 x 20 board) and human engineered features (number of holes, differences in height between adjacent columns, max height etc). Models were trained using different combinations of game state encodings.

The model is trained using reinforcement learning with replay memory and optimized using batch gradient descent.

## Usage

Trained models are stored in a trained_models directory and numbered sequentially (i.e. model0.h5)
Model 17 and 19 were the most successful of the trained models.

```
# train initial model
python main.py train

# train model based off previous model 
# the new model is saved as the next model_number
python main.py train <model_number>

# watch a model play
python main.py play <model_number>
```

