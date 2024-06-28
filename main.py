import train
import inference

if __name__ == "__main__":
    # Choose whether to train or run inference
    action = "train"  # Change to "inference" as needed

    if action == "train":
        train.train()
    elif action == "inference":
        inference.inference()