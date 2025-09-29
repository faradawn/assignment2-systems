# HW2

## Question 1

a) benchmarking_script.py

b) warmup = 5, num steps = 10, 
{'forward_time': 0.03915175800002544, 'backward_time': 0.05054146880006556, 'forward_var': np.float64(1.632887396883597e-05), 'backward_var': np.float64(0.0001721222424806588)}

Forward is taking 2 times longer than backward. Derivative is dw = dl / da * da / dw, and also da = xxx . which 2 times time computation of foward a = wx + b

Variance is low after warm up.

c) If without warmup, the forward time is higher, because the initial build of cuda graph I suppose.

Namespace(num_warmup=0, num_epochs=10, max_seq_len=256, d_model=768, d_ff=3072, num_layers=12, num_heads=12)
device cuda
{'forward_time': 0.09276861330008615, 'backward_time': 0.05972346790003939, 'forward_var': np.float64(0.028579081714775474), 'backward_var': np.float64(0.002207662404999231)}

Warm up 2

Namespace(num_warmup=2, num_epochs=10, max_seq_len=256, d_model=768, d_ff=3072, num_layers=12, num_heads=12)
device cuda
{'forward_time': 0.039935211200054256, 'backward_time': 0.046788070900038295, 'forward_var': np.float64(2.5362346816779135e-05), 'backward_var': np.float64(9.420173579811798e-05)}

Warmup 3

{'forward_time': 0.039643416500030074, 'backward_time': 0.04830947209998158, 'forward_var': np.float64(2.2877233540053166e-05), 'backward_var': np.float64(0.00014312291323110532)}