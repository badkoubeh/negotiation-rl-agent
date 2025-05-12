# Negotiation RL Agent Using Indication of Interest (IOI) Financial Data

This project uses Reinforcement Learning (RL) to simulate and optimize the negotiation process between buyers and sellers in the private equity market. The objective is to create an intelligent agent that can recommend the best counteroffer or decision (accept, reject, or counteroffer) based on historical Indication of Interest (IOI) data, maximizing the final accepted price while minimizing negotiation time.

## Project Structure

- **negotiation_env.py**: Defines the custom RL environment for the negotiation process, implementing the Gymnasium interface.
- **train_agent.py**: Contains the training loop that trains the RL agent using Stable-Baselines3.
- **inference.py**: This file can be used to load a trained agent and make predictions on new data (not provided yet but can be extended).
- **models**: Contains the saved models after training.
- **logs**: Stores the training logs used by TensorBoard for monitoring the agent's performance.

## Getting Started

Follow these steps to get the project up and running.

### Prerequisites

You will need Python 3.x and the following dependencies:

- `gymnasium` for the environment framework.
- `stable-baselines3` for RL training.
- `torch` for the deep learning framework.
- `tensorboard` for training logs (optional, but recommended).
- `wandb` weight and biases

## Environment Setup

To set up the project environment using Conda, follow these steps:

1. Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

2. Use the provided `environment.yml` file to create the environment:

```bash
conda env create -f environment.yml
```
💡 You can also use the setup.sh bash script if provided:

## Training the Agent
To train the RL agent using the DQN algorithm from Stable-Baselines3:
```bash
python -m train.train_agent
```
This script:

- Loads the NegotiationEnv environment
- Trains a DQN model with defined hyperparameters
- Saves the best-performing and final models to the models/ directory
- Logs training metrics to logs/ for visualization in TensorBoard

## Monitoring with TensorBoard
To monitor the training process:
```bash
tensorboard --logdir=logs
```
Open http://localhost:6006 in your browser to view real-time training curves and metrics.

## Inference
Use a trained agent to make predictions on new IOI negotiation states:
```
from stable_baselines3 import DQN
from env.negotiation_env import NegotiationEnv

model = DQN.load("models/final_model")
env = NegotiationEnv()
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
```
## Reward Strategy

The reward system encourages the agent to:

- Maximize the final agreed price (favoring the seller)
- Minimize the number of negotiation rounds
- Avoid accepting lowball offers or rejecting too early

Reward formula:
```
((price - min_price) / (initial_price - min_price)) * (1 - round_num / max_rounds)
```

## Experiment Tracking with Weights & Biases
To track training metrics and model performance over time, use Weights & Biases:

Install WandB
```bash
pip install wandb
```
## Future Improvements
- Add a buyer agent to simulate two-party, multi-agent negotiation.
- Use real-world IOI datasets or synthetic data from live platforms.

- - Incorporate market sentiment and external signals (news, funding rounds).

- Extend to continuous action space using PPO or SAC for pricing strategy.

- Deploy a FastAPI microservice to serve real-time negotiation recommendations.

- Add evaluation metrics: average deal value, success rate, negotiation duration.

- Implement self-play to co-train buyer and seller agents.