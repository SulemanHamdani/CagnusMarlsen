# Chess Bot Created from Scratch Using Deep Neural Network

This project aims to create a chess bot from scratch using a deep neural network. The bot is trained on a dataset of chess games and uses a neural network to predict the best move in a given position. The goal of the project is to create a bot that can play chess at a high level and possibly even compete against human players.

The project is based on the research paper "Playing Chess with Deep Neural Networks" by Matthew Lai, which can be found at the following link: https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf. This paper provides a detailed description of the neural network architecture and training process used to create a strong chess-playing bot.

## Data

The current dataset used to train the bot contains 1500 games, which have been labeled using the Stockfish chess engine at depth 16. Each game is represented as a sequence of board positions and moves, which are used to train the neural network.

In the future, we plan to expand the dataset to include more games and improve the quality of the labels. We believe that a larger and more diverse dataset will help the bot learn better chess strategies and improve its performance.

## Model

The neural network used to train the bot is based on a deep neural network architecture, which has shown to be effective in many chess engines. The NN takes in a board position as input and outputs a probability distribution over possible moves.

The training process involves using a combination of supervised learning. During supervised learning, the neural network is trained on labeled game data to predict the best move in each position.

## Training

![Example image](https://i.imgur.com/QwSdFBh.png)

## Deployment

We have deployed the bot to the Lichess platform using the Lichess-bot API. This allows the bot to play against human players on the platform and improve its performance through experience.

## Play Against Me
https://lichess.org/@/cagnusmarlsen01

## Future Work

- [ ] Improve the dataset by adding more games
- [ ] Retrain the model using the improved dataset
- [ ] Train a regression model on the improved dataset
- [ ] Add alpha-beta pruning in the model at least 2 moves deep
- [ ] Giraffe Chess Engine 
