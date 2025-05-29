import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv(
    "/home/hafezm/Downloads/RL exercise/DLL_25_IL_RL_Exercise/tensorboard_scalars_reinforcement_best.csv"
)
# Check if columns exist
# Plot: Episode Reward
plt.figure(figsize=(10, 5))
plt.plot(df["train/accuracy"], label="train_accuracy", color="blue")
plt.xlabel("Step or Episode")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy Over Time")
plt.grid(True)
plt.legend()
plt.savefig("Best_Reward.png")
plt.show()

# Plot: Eval Episode Reward
eval_data = df[["val/accuracy"]].dropna().reset_index()
plt.figure(figsize=(10, 5))
plt.plot(eval_data.to_numpy()[:, 0], eval_data.to_numpy()[:, 1])
plt.xlabel("Step")
plt.ylabel("Validation Accuracy Reward")
plt.title("Validation Accuracy Over Time")
plt.grid(True)
plt.legend()
plt.savefig("validation_reward_best.png")
plt.show()


# Save the figure
# plt.savefig("rewards_plot.png")
# plt.show()
