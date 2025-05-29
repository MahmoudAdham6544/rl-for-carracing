import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_scalars_to_csv(log_dir, output_csv):
    # Load the TensorBoard event file
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all scalar tags (e.g., train_loss, validation_accuracy, etc.)
    tags = event_acc.Tags()["scalars"]

    # Prepare dictionary with all tag data
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        for event in events:
            step = event.step
            if step not in data:
                data[step] = {}
            data[step][tag] = event.value

    # Write to CSV
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        headers = ["step"] + tags
        writer.writerow(headers)

        for step in sorted(data.keys()):
            row = [step] + [data[step].get(tag, "") for tag in tags]
            writer.writerow(row)

    print(f"Saved scalars to {output_csv}")


# Example usage
log_dir = "/home/hafezm/Downloads/RL exercise/DLL_25_IL_RL_Exercise/tensorboard/Imitation Learning 0-20250525-170133"  # adjust if needed
output_file = "tensorboard_scalars_reinforcement_best.csv"
export_scalars_to_csv(log_dir, output_file)
