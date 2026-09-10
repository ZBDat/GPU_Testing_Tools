# GPU_Testing_Tools
Benchmarking a GPU's performance for my use case.

## GPU 自动化测试工具

新增 `gpu_testing_tool.py`，用于对 ONNX 模型执行以下场景测试并统一输出 Excel：

1. 单 session 顺序推理（逐图统计耗时）
   - 1b：读取模型的静态 batch 维度后批量推理；尾批使用零图补齐。没有静态 batch 大小大于 1 的模型会记录为不支持。
2. 单 session 任务并发（并发从 2 递增到用户设定上限；每个请求由独立发送线程直接调用同一 ONNX Runtime session）
3. session 并发（2 个真实发送线程 + 多 session，session 数从 2 递增直到 OOM，OOM 后停止扩容并继续后续场景）
4. 固定间隔双发送线程场景
    - 4a：单 session
    - 4b：双 session（每个 session 固定对应一个发送线程）
5. 单图全 session 并发：每张图同时发送至 `--max-session-concurrency` 个独立 session，统计从任一 session 开始处理到全部 session 完成的时间。

每个小场景都记录：
- 场景 1：单图推理耗时（不含图像读取时间）
- 场景 2/3/4：每个请求组从计划提交时刻到该组全部处理完成的耗时
- 峰值显存占用
- 峰值带宽占用（NVML memory utilization）

`--gpu-index` 选择 NVML 采集的 GPU，默认为 `0`。场景 4 按固定提交节拍发送请求，不等待上一请求组完成。

并在每个 Excel sheet 中写入：
- CUDA 版本
- TensorRT 版本
- 模型大小
- 图像大小与 shape

运行前会先执行单图单次推理并在命令行打印结果，用于验证模型可用性。
运行过程中会持续输出 `[STATUS]` 日志，指示启动、各场景开始/完成以及并发子场景进度。

### 依赖

```bash
pip install onnxruntime-gpu onnx tifffile openpyxl pynvml
```

### 使用方式

```bash
python gpu_testing_tool.py \
  --image-dir /path/to/images \
  --model /path/to/model.onnx \
  --output-excel /path/to/result.xlsx \
  --ep cuda \
  --max-task-concurrency 8 \
   --max-session-concurrency 8 \
   --interval-ms 100 \
   --pad
```

### Nsight Systems 连续执行

`nsys_two_sender_four_worker.py` 使用与主工具相同的模型探测、TIFF 预处理和输入验证方式，启动 2 个 sender 与 4 个独立 ONNX Runtime session worker。每轮向每个 worker 提交一个请求，等待该轮结束后继续下一轮，因此每个 worker 最多有一个 in-flight 请求，不会无限堆积任务。

```bash
nsys profile --trace=cuda,nvtx,osrt -o onnx_two_sender_four_worker \
  python nsys_two_sender_four_worker.py \
  --image-dir /path/to/images \
  --model /path/to/model.onnx \
  --ep cuda \
  --duration-seconds 30
```

省略 `--duration-seconds` 可持续运行，并使用 Ctrl+C 停止。建议先用有限时长 trace，检查四个 session 的 CUDA kernel、H2D/D2H copy 与 stream 是否重叠。

### 自动化 Nsight 分析

`analyze_nsys_trace.py` 会以 `cuda-event-trace=false` 采集、导出 SQLite，并生成 CUDA kernel 并发度、CUDA API 阻塞时间和 GPU copy 汇总的 Markdown 报告。它会优先寻找 `PATH` 中的 `nsys`，否则使用本机默认的 Nsight Systems 2025.1.3 路径；可用 `--nsys-path` 覆盖。

采集并分析：

```bash
python analyze_nsys_trace.py profile \
  --image-dir images \
  --model model_trt.onnx \
  --duration-seconds 30 \
  --output two_sender_four_worker_trace
```

PowerShell 可直接执行默认的静态模型 120 秒采集与分析：

```powershell
.\profile_static_model.ps1
```

例如指定 30 秒、TensorRT EP 和不同输出名称：

```powershell
.\profile_static_model.ps1 -DurationSeconds 30 -ExecutionProvider tensorrt -Output trt_static_30s
```

分析已有报告：

```bash
python analyze_nsys_trace.py analyze \
  --report two_sender_four_worker_no_event_trace.nsys-rep
```

### 固定输入尺寸 ONNX 优化

当全部 TIFF 尺寸固定时，可将模型的动态高宽静态化，并由 ONNX Simplifier 折叠 shape/anchor 相关的常量计算。原模型不会被覆盖：

```bash
python make_static_onnx.py \
  --model model_trt.onnx \
  --output model_trt_static_1108x2232.onnx \
  --height 1108 \
  --width 2232
```
