from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
import torch

def training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    vocab_size = 30522  # set this to your vocab size

    x = torch.randint(0, vocab_size, (16, 1024), dtype=torch.long, device=device)
    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=1024, d_model=256, num_layers=2, num_heads=4, d_ff=256, rope_theta=1).to(device)

    print("x shape", x.shape)
    y_pred = model.forward(x)

if __name__ == "__main__":
    training()
    