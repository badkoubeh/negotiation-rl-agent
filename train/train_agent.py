import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from simulation_environment.negotiation_env import NegotiationEnv

def main():
    # Load train and test datasets
    train_df = pd.read_parquet("data/processed/ioi_train.parquet")
    test_df = pd.read_parquet("data/processed/ioi_test.parquet")

    # Create environments
    train_env = NegotiationEnv(ioi_df=train_df)
    test_env = NegotiationEnv(ioi_df=test_df)

    # Wrap environments with Monitor for logging
    train_env = Monitor(train_env)
    test_env = Monitor(test_env)

    # Optional: check env compatibility
    check_env(train_env, warn=True)

    # Define evaluation callback
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # Initialize model
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.95,
        tau=1.0,
        train_freq=4,
        target_update_interval=500,
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Train the model
    model.learn(total_timesteps=20000, callback=eval_callback)

    # Save the final model
    model.save("models/final_model")

    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main()
