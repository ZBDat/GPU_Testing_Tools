# GPU_Testing_Tools
Benchmarking a GPU's performance for my use case.

## GPU 自动化测试工具

新增 `gpu_testing_tool.py`，用于对 ONNX 模型执行以下场景测试并统一输出 Excel：

1. 单 session 顺序推理（逐图统计耗时）
2. 单 session 任务并发（并发从 2 递增到用户设定上限）
3. session 并发（2 个发送线程 + 多 session，session 数从 2 递增直到 OOM，OOM 后自动恢复并继续后续场景）
4. 固定间隔双发送线程场景
   - 4a：单 session
   - 4b：双 session（每个 session 固定对应一个发送线程）

每个小场景都记录：
- 单图推理耗时（不含图像读取时间）
- 峰值显存占用
- 峰值带宽占用（NVML memory utilization）

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
  --interval-ms 100
```
