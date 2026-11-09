import onnxruntime as ort
from time import perf_counter
from PIL import Image
import numpy as np
# import torch
from torchvision.transforms.v2 import functional as F
from pathlib import Path

def benchmark_onnx_model(onnx_path, num_warmup=10, num_runs=100):
    # hide false error log
    opts = ort.SessionOptions()
    opts.log_severity_level = 3 # log level: 0 = Verbose, 1 = Info, 2 = Warning, 3 = Error, 4 = Fatal (show only >=)
    
    session = ort.InferenceSession(onnx_path, provider_options=['CPUExecutionProvider'], sess_options=opts)
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    fixed_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_shape]   # channel
    
    dummy_input = np.random.randn(*fixed_shape).astype(np.float32)
    
    print(f'[System] Warming-up onnx model for {num_warmup} iterations.')
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy_input})
    print('[System] Finish warm-up.')
    
    print(f'[System] Running {num_runs} benchmark iterations.')
    latencies = []
    throughput_start = perf_counter()   # second
    for _ in range(num_runs):
        latency_start = perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latency_end = perf_counter() - latency_start
        latencies.append(latency_end * 1000)
    throughput_end = perf_counter() - throughput_start
    print('[System] Finish benchmark iteration.')
    
    # calculate metrics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p90_latency = np.percentile(latencies, 90)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    total_img = num_runs if input_shape[0] == 1 else num_runs * input_shape[0]
    throughput = total_img / throughput_end
    
    print("--- Benchmark Results ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Min Latency:     {min_latency:.2f} ms")
    print(f"Max Latency:     {max_latency:.2f} ms")
    print(f"P90 Latency:     {p90_latency:.2f} ms")
    print(f"P95 Latency:     {p95_latency:.2f} ms")
    print(f"P99 Latency:     {p99_latency:.2f} ms")
    print(f"Throughput:      {throughput:.2f} fps")
    
if __name__ == '__main__':
    onnx_path = './test-export/models/convnext_l_2cls.onnx'
    benchmark_onnx_model(onnx_path)