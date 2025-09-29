from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Faradawn Parser')
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_warmup', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=256)
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    vocab_size = 10000  # set this to your vocab size
    batch_size = 4
    

    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=1024, d_model=256, num_layers=args.num_layers, num_heads=4, d_ff=256, rope_theta=1).to(device)

    optimizer = AdamW(params=model.parameters())

    x = torch.randint(low=0, high=vocab_size, size=(batch_size, args.max_seq_len), dtype=torch.long, device=device)
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, args.max_seq_len), dtype=torch.long, device=device)
    
    for i in range(args.num_epochs):
        logits = model(x)
        print("logits size", logits.size())

        loss = cross_entropy(logits.view(-1, vocab_size), targets)
        loss.backward()
        optimizer.step()
        print("Done training step")

    


    