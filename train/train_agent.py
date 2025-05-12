from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from simulation_environment.negotiation_env import NegotiationEnv

def main():
    # Initialize environment
    env = NegotiationEnv()
    
    # Optional: sanity check on custom env
    check_env(env)

    # Define evaluation environment and callback
    eval_env = NegotiationEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # Define model (you can try other policies like 'MlpPolicy', 'CnnPolicy' etc.)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.95,
        train_freq=4,
        target_update_interval=500,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )

    # Train the model
    model.learn(total_timesteps=20000, callback=eval_callback)

    # Save the final model
    model.save("./models/final_model")

    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
