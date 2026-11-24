"""RTX 3090 실습 환경 점검: 버전, 정밀도 지원, 행렬곱 처리량, SDPA 백엔드."""
import time

import torch
import torch.nn.functional as F


def bench_matmul(dtype, n=8192, iters=50):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * n**3 / dt / 1e12


def bench_sdpa(dtype=torch.float16, b=4, h=16, s=4096, d=64, iters=20):
    q, k, v = (torch.randn(b, h, s, d, device="cuda", dtype=dtype) for _ in range(3))
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    return ms, torch.cuda.max_memory_allocated() / 2**30


print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
print("GPU", torch.cuda.get_device_name(0), "| compute capability", torch.cuda.get_device_capability(0))
print("total memory (GiB)", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print("bf16 supported", torch.cuda.is_bf16_supported())
try:
    x = torch.randn(16, 16, device="cuda").to(torch.float8_e4m3fn)
    torch._scaled_mm(x, x.t(), scale_a=torch.tensor(1.0, device="cuda"), scale_b=torch.tensor(1.0, device="cuda"))
    print("fp8 matmul OK")
except Exception as e:  # Ampere에는 FP8 텐서 코어가 없다
    print("fp8 matmul unsupported:", type(e).__name__, str(e).splitlines()[0][:100])

torch.backends.cuda.matmul.allow_tf32 = False
print(f"matmul FP32        {bench_matmul(torch.float32):6.1f} TFLOPS")
torch.backends.cuda.matmul.allow_tf32 = True
print(f"matmul TF32        {bench_matmul(torch.float32):6.1f} TFLOPS")
print(f"matmul FP16        {bench_matmul(torch.float16):6.1f} TFLOPS")
print(f"matmul BF16        {bench_matmul(torch.bfloat16):6.1f} TFLOPS")
ms, gib = bench_sdpa()
print(f"SDPA fp16 (4x16x4096x64) {ms:.2f} ms, peak {gib:.2f} GiB")
