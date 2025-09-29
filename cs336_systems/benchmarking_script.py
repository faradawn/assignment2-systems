from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import argparse
import timeit
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Faradawn Parser')
    parser.add_argument('--num_warmup', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=256)

    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)

    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    vocab_size = 10000  # set this to your vocab size
    batch_size = 4
    

    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=args.max_seq_len, d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=1).to(device)

    optimizer = AdamW(params=model.parameters())

    x = torch.randint(low=0, high=vocab_size, size=(batch_size, args.max_seq_len), dtype=torch.long, device=device)
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, args.max_seq_len), dtype=torch.long, device=device)

    for i in range(args.num_warmup):
        logits = model(x)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) # inputs: [B*T, vocab_size]. targets: [B*T, ]
        loss.backward()

    optimizer = AdamW(params=model.parameters())
    
    forward_times = []
    backward_times = []
    for i in range(args.num_epochs):
        t0 = timeit.default_timer()
        logits = model(x)
        t1 = timeit.default_timer()
        forward_times.append(t1-t0)

        # TODO: how to zero grad 
        optimizer.zero_grad()
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) # inputs: [B*T, vocab_size]. targets: [B*T, ]
        start = timeit.default_timer()
        loss.backward()
        end = timeit.default_timer()
        backward_times.append(end-start)
        optimizer.step()
        
        # print(f"i {i}, forward time {t1-t0}, backward time {end-start}")

    result = {}
    result["forward_time"] = sum(forward_times) / len(forward_times)
    result["backward_time"] = sum(backward_times) / len(backward_times)
    result["forward_var"] = np.var(forward_times, ddof=1)
    result["backward_var"] = np.var(backward_times, ddof=1)
    print(result)

    


    