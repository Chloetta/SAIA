import torch
from model import Transformer, ModelArgs
from data import load_data

def inference():
    model_args = ModelArgs(vocab_size=1000)
    model = Transformer(model_args)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    test_data, _ = load_data()
    with torch.no_grad():
        predictions = model(test_data, 0)
        print(predictions)

if __name__ == "__main__":
    inference()