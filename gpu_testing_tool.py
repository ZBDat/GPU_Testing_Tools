import argparse
import gc
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import onnx
except ImportError:
    onnx = None

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None

try:
    import pynvml
except ImportError:
    pynvml = None


ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(bool)": np.bool_,
}

RESULT_TIMEOUT_SECONDS = 300.0


@dataclass
class InferenceResult:
    image_name: str
    latency_ms: float


@dataclass
class InferenceOutcome:
    result: Optional[InferenceResult] = None
    error: Optional[BaseException] = None


@dataclass
class ScenarioResult:
    scenario_name: str
    details: str
    per_image: List[InferenceResult]
    peak_memory_mb: float
    peak_bandwidth_percent: float
    task_traces: List["TaskTrace"] = field(default_factory=list)


@dataclass
class TaskTrace:
    request_group: int
    image_name: str
    sender_id: int
    worker_id: int
    sent_ms: float
    processing_started_ms: Optional[float] = None
    processing_completed_ms: Optional[float] = None


class GPUMonitor:
    def __init__(self, gpu_index: int = 0):
        self.enabled = False
        self.backend = "unavailable"
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_memory_bytes = 0
        self.peak_bandwidth_percent = 0.0
        self._handle = None

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.enabled = True
                self.backend = "pynvml"
            except Exception:
                self.enabled = False
        if not self.enabled:
            try:
                self._read_nvidia_smi()
                self.enabled = True
                self.backend = "nvidia-smi"
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                pass

    def start(self):
        if not self.enabled:
            return
        self._stop.clear()
        self.peak_memory_bytes = 0
        self.peak_bandwidth_percent = 0.0
        self._sample()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def close(self):
        if self.enabled and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _collect(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                return
            time.sleep(0.01 if self.backend == "pynvml" else 0.1)

    def _read_nvidia_smi(self) -> Tuple[int, float]:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=memory.used,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [value.strip() for value in result.stdout.strip().split(",")]
        if len(values) != 2:
            raise ValueError(f"Unexpected nvidia-smi output: {result.stdout!r}")
        return int(float(values[0]) * 1024 * 1024), float(values[1])

    def _sample(self):
        if self.backend == "pynvml":
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
            util = float(pynvml.nvmlDeviceGetUtilizationRates(self._handle).memory)
        else:
            mem, util = self._read_nvidia_smi()
        self.peak_memory_bytes = max(self.peak_memory_bytes, mem)
        self.peak_bandwidth_percent = max(self.peak_bandwidth_percent, util)


class SessionWorker(threading.Thread):
    def __init__(self, worker_id: int, session: Any, input_name: str):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.session = session
        self.input_name = input_name
        self.tasks: "queue.Queue[Tuple[Optional[str], Optional[np.ndarray], Optional[queue.Queue], Optional[Callable[[int], None]], Optional[Callable[[int], None]]]]" = queue.Queue()
        self._stop_event = threading.Event()

    def submit(
        self,
        image_name: str,
        arr: np.ndarray,
        result_queue: queue.Queue,
        on_start: Optional[Callable[[int], None]] = None,
        on_complete: Optional[Callable[[int], None]] = None,
    ):
        self.tasks.put((image_name, arr, result_queue, on_start, on_complete))

    def run(self):
        while not self._stop_event.is_set():
            image_name, arr, result_queue, on_start, on_complete = self.tasks.get()
            if image_name is None or arr is None or result_queue is None:
                return
            try:
                if on_start is not None:
                    on_start(self.worker_id)
                start = time.perf_counter()
                self.session.run(None, {self.input_name: arr})
                latency = (time.perf_counter() - start) * 1000.0
                if on_complete is not None:
                    on_complete(self.worker_id)
                result_queue.put(InferenceOutcome(result=InferenceResult(image_name=image_name, latency_ms=latency)))
            except BaseException as exc:
                result_queue.put(InferenceOutcome(error=exc))

    def stop(self):
        self._stop_event.set()
        self.tasks.put((None, None, None, None, None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU benchmark tool based on ONNX Runtime.")
    parser.add_argument("--image-dir", required=True, help="Directory with uint16 tif images")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--output-excel", default="gpu_test_results.xlsx", help="Excel output path")
    parser.add_argument("--ep", choices=["cuda", "tensorrt"], default="cuda", help="Execution provider")
    parser.add_argument("--max-task-concurrency", type=int, default=8, help="Scenario 2 max concurrency")
    parser.add_argument("--max-session-concurrency", type=int, default=8, help="Scenario 3 max sessions")
    parser.add_argument("--interval-ms", type=int, default=100, help="Scenario 4 sender interval")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap of images to use")
    parser.add_argument("--pad", action="store_true", help="Pad image height and width up to multiples of 128")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to monitor")
    args = parser.parse_args()
    if args.max_task_concurrency < 2:
        parser.error("--max-task-concurrency must be at least 2")
    if args.max_session_concurrency < 2:
        parser.error("--max-session-concurrency must be at least 2")
    if args.interval_ms < 0:
        parser.error("--interval-ms must be non-negative")
    if args.max_images < 0:
        parser.error("--max-images must be non-negative")
    if args.gpu_index < 0:
        parser.error("--gpu-index must be non-negative")
    return args


def get_model_info(model_path: str) -> Tuple[str, List[Optional[int]], List[Optional[str]]]:
    if onnx is None:
        return "unknown", [], []
    model = onnx.load(model_path)
    input0 = model.graph.input[0]
    dims: List[Optional[int]] = []
    dim_names: List[Optional[str]] = []
    for d in input0.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append(None)
        dim_names.append(d.dim_param if d.HasField("dim_param") else None)
    return input0.name, dims, dim_names


def resolve_numpy_dtype(ort_type: str) -> np.dtype:
    if ort_type not in ORT_TYPE_TO_NUMPY:
        raise ValueError(f"Unsupported ONNX input dtype: {ort_type}")
    return np.dtype(ORT_TYPE_TO_NUMPY[ort_type])


def _spatial_axis(dim_name: Optional[str]) -> Optional[str]:
    if dim_name is None:
        return None
    normalized = dim_name.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"h", "height", "imageheight", "d0"}:
        return "height"
    if normalized in {"w", "width", "imagewidth", "d1"}:
        return "width"
    return None


def get_batch_axis(target_shape: Sequence[Optional[int]], dim_names: Sequence[Optional[str]]) -> Optional[int]:
    for axis, dim_name in enumerate(dim_names):
        if dim_name is not None and dim_name.strip().lower().replace("_", "") in {"batch", "batchsize", "n"}:
            return axis
    if len(target_shape) == 4 and target_shape[-1] == 1:
        return 0
    return None


def prepare_image_for_model(
    image: np.ndarray,
    target_shape: Sequence[Optional[int]],
    dim_names: Sequence[Optional[str]],
    pad_to_128: bool = False,
) -> np.ndarray:
    """Map a 2D TIFF to named model axes, optionally zero-padding bottom and right edges."""
    if image.ndim != 2:
        raise ValueError(f"Only 2D TIFF images are supported; received shape {image.shape}")
    if len(target_shape) != len(dim_names):
        raise ValueError("Model input shape and dimension-name metadata have different ranks")

    spatial_axes = [_spatial_axis(name) for name in dim_names]
    if spatial_axes.count("height") != 1 or spatial_axes.count("width") != 1:
        # Static ONNX dimensions replace symbolic names. Accept only the
        # unambiguous HWC form used by this tool's single-channel TIFF models.
        if len(target_shape) == 3 and target_shape[2] == 1 and all(name is None for name in dim_names):
            spatial_axes = ["height", "width", None]
        else:
            raise ValueError(
                "Model input must expose exactly one height and one width dimension name; "
                f"received dimension names {list(dim_names)}"
            )
    batch_axis = get_batch_axis(target_shape, dim_names)

    height, width = image.shape
    if pad_to_128:
        padded_height = ((height + 127) // 128) * 128
        padded_width = ((width + 127) // 128) * 128
        if (padded_height, padded_width) != (height, width):
            padded = np.zeros((padded_height, padded_width), dtype=image.dtype)
            padded[:height, :width] = image
            image = padded
            height, width = image.shape
    output_shape = []
    for axis, (target_dim, spatial_axis) in enumerate(zip(target_shape, spatial_axes)):
        actual_dim = height if spatial_axis == "height" else width if spatial_axis == "width" else 1
        if axis != batch_axis and target_dim is not None and target_dim > 0 and target_dim != actual_dim:
            raise ValueError(
                f"Image shape {image.shape} is incompatible with model axis {axis} ({dim_names[axis]!r}): "
                f"expected {target_dim}, received {actual_dim}. Images are never cropped or padded."
            )
        output_shape.append(actual_dim)

    output = np.zeros(output_shape, dtype=image.dtype)
    index = [0] * len(output_shape)
    index[spatial_axes.index("height")] = slice(None)
    index[spatial_axes.index("width")] = slice(None)
    output[tuple(index)] = image
    return output


def assemble_image_batch(
    images: Sequence[np.ndarray],
    batch_axis: int,
    dim_names: Sequence[Optional[str]],
    batch_size: int,
) -> np.ndarray:
    """Pad spatial dimensions within a batch and append zero-valued tail samples."""
    if not images:
        raise ValueError("Cannot assemble an empty image batch")
    if len(images) > batch_size:
        raise ValueError(f"Received {len(images)} images for batch size {batch_size}")

    output_shape = list(images[0].shape)
    output_shape[batch_axis] = batch_size
    spatial_axes = {axis for axis, dim_name in enumerate(dim_names) if _spatial_axis(dim_name) is not None}
    for image in images[1:]:
        if image.ndim != len(output_shape):
            raise ValueError(f"All batch images must have rank {len(output_shape)}, received {image.shape}")
        for axis, size in enumerate(image.shape):
            if axis == batch_axis:
                if size != 1:
                    raise ValueError(f"Each prepared image must have batch size 1, received {image.shape}")
            elif axis in spatial_axes:
                output_shape[axis] = max(output_shape[axis], size)
            elif size != output_shape[axis]:
                raise ValueError(f"Cannot batch images with different non-spatial shapes: {images[0].shape}, {image.shape}")

    batch = np.zeros(output_shape, dtype=images[0].dtype)
    for batch_index, image in enumerate(images):
        destination = [slice(0, size) for size in image.shape]
        destination[batch_axis] = slice(batch_index, batch_index + 1)
        batch[tuple(destination)] = image
    return batch


def load_and_prepare_images(
    image_dir: str,
    target_shape: Sequence[Optional[int]],
    dim_names: Sequence[Optional[str]],
    target_dtype: np.dtype,
    max_images: int,
    pad_to_128: bool = False,
) -> List[Tuple[str, np.ndarray, Tuple[int, ...]]]:
    if tifffile is None:
        raise RuntimeError("tifffile is required to run this tool")

    files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff"))]
    )
    if not files:
        raise RuntimeError("No tif/tiff images found in image-dir")
    if max_images > 0:
        files = files[:max_images]

    prepared = []
    for f in files:
        path = os.path.join(image_dir, f)
        original = np.asarray(tifffile.imread(path))
        original_shape = tuple(original.shape)
        img = prepare_image_for_model(original, target_shape, dim_names, pad_to_128)
        img = img.astype(target_dtype, copy=False)
        prepared.append((f, img, original_shape))
    return prepared


def create_session(model_path: str, ep: str):
    if ort is None:
        raise RuntimeError("onnxruntime is required to run this tool")
    providers = ["CUDAExecutionProvider"]
    if ep == "tensorrt":
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    so = ort.SessionOptions()
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def _get_outcome(result_queue: queue.Queue, context: str) -> InferenceResult:
    try:
        outcome = result_queue.get(timeout=RESULT_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TimeoutError(f"Timed out waiting for inference result: {context}") from exc
    if outcome.error is not None:
        raise RuntimeError(f"Inference failed during {context}: {outcome.error}") from outcome.error
    if outcome.result is None:
        raise RuntimeError(f"Inference worker returned no result during {context}")
    return outcome.result


def run_inference_batch(
    workers: List[SessionWorker],
    assignments: List[Tuple[int, str, np.ndarray]],
) -> List[InferenceResult]:
    result_queue: "queue.Queue[InferenceOutcome]" = queue.Queue()
    for worker_idx, image_name, arr in assignments:
        workers[worker_idx].submit(image_name, arr, result_queue)
    results = [_get_outcome(result_queue, "inference batch") for _ in assignments]
    return results


def run_grouped_inference_batches(
    workers: List[SessionWorker],
    grouped_assignments: List[List[Tuple[int, str, np.ndarray]]],
) -> List[InferenceResult]:
    result_queue: "queue.Queue[InferenceOutcome]" = queue.Queue()
    group_results: List[InferenceResult] = []
    for group_idx, assignments in enumerate(grouped_assignments, start=1):
        start = time.perf_counter()
        for worker_idx, image_name, arr in assignments:
            workers[worker_idx].submit(image_name, arr, result_queue)
        for _ in assignments:
            _get_outcome(result_queue, f"request group {group_idx}")
        elapsed = (time.perf_counter() - start) * 1000.0
        group_results.append(InferenceResult(image_name=f"request_group_{group_idx}", latency_ms=elapsed))
    return group_results


def run_direct_concurrent_groups(
    session: Any,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    concurrency: int,
) -> List[InferenceResult]:
    """Call one ORT session concurrently from independent sender threads."""
    group_results: List[InferenceResult] = []
    for group_idx, group in enumerate(
        (prepared_images[i : i + concurrency] for i in range(0, len(prepared_images), concurrency)), start=1
    ):
        result_queue: "queue.Queue[InferenceOutcome]" = queue.Queue()
        start_gate = threading.Event()
        request_started: List[Optional[float]] = [None]
        request_start_lock = threading.Lock()

        def record_request_start():
            with request_start_lock:
                if request_started[0] is None:
                    request_started[0] = time.perf_counter()

        def send(image_name: str, arr: np.ndarray):
            start_gate.wait()
            try:
                start = time.perf_counter()
                record_request_start()
                session.run(None, {input_name: arr})
                result_queue.put(InferenceOutcome(result=InferenceResult(image_name, (time.perf_counter() - start) * 1000.0)))
            except BaseException as exc:
                result_queue.put(InferenceOutcome(error=exc))

        senders = [threading.Thread(target=send, args=(name, arr)) for name, arr, _ in group]
        for sender in senders:
            sender.start()
        start_gate.set()
        try:
            for _ in group:
                _get_outcome(result_queue, f"concurrent request group {group_idx}")
        finally:
            for sender in senders:
                sender.join(timeout=RESULT_TIMEOUT_SECONDS)
                if sender.is_alive():
                    raise TimeoutError(f"Sender thread did not exit for request group {group_idx}")
        group_results.append(
            InferenceResult(
                image_name=f"request_group_{group_idx}",
                latency_ms=(time.perf_counter() - (request_started[0] or time.perf_counter())) * 1000.0,
            )
        )
    return group_results


def run_worker_concurrent_groups(
    workers: List[SessionWorker],
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    task_concurrency: int,
) -> List[InferenceResult]:
    """Measure each request from the first worker start through its final completion."""
    group_results: List[InferenceResult] = []
    for group_idx, group in enumerate(
        (prepared_images[i : i + task_concurrency] for i in range(0, len(prepared_images), task_concurrency)), start=1
    ):
        result_queue: "queue.Queue[InferenceOutcome]" = queue.Queue()
        request_started: List[Optional[float]] = [None]
        request_start_lock = threading.Lock()

        def record_request_start(_worker_id: int):
            with request_start_lock:
                if request_started[0] is None:
                    request_started[0] = time.perf_counter()

        for image_idx, (image_name, arr, _) in enumerate(group):
            workers[image_idx % len(workers)].submit(image_name, arr, result_queue, record_request_start)
        for _ in group:
            _get_outcome(result_queue, f"concurrent request group {group_idx}")
        if request_started[0] is None:
            raise RuntimeError(f"No worker started concurrent request group {group_idx}")
        group_results.append(
            InferenceResult(
                image_name=f"request_group_{group_idx}",
                latency_ms=(time.perf_counter() - request_started[0]) * 1000.0,
            )
        )
    return group_results


def run_same_image_all_sessions(
    workers: List[SessionWorker],
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
) -> List[InferenceResult]:
    """Run one image on every session and time first processing start through final completion."""
    results: List[InferenceResult] = []
    for image_name, arr, _ in prepared_images:
        result_queue: "queue.Queue[InferenceOutcome]" = queue.Queue()
        request_started: List[Optional[float]] = [None]
        request_start_lock = threading.Lock()

        def record_request_start(_worker_id: int):
            with request_start_lock:
                if request_started[0] is None:
                    request_started[0] = time.perf_counter()

        for worker in workers:
            worker.submit(image_name, arr, result_queue, record_request_start)
        for _ in workers:
            _get_outcome(result_queue, f"all-session inference for {image_name}")
        if request_started[0] is None:
            raise RuntimeError(f"No worker started all-session inference for {image_name}")
        results.append(
            InferenceResult(
                image_name=image_name,
                latency_ms=(time.perf_counter() - request_started[0]) * 1000.0,
            )
        )
    return results


def run_two_sender_jobs(
    workers: List[SessionWorker],
    sender_jobs: List[List[Tuple[int, str, np.ndarray]]],
    interval_ms: Optional[int] = None,
) -> Tuple[List[InferenceResult], List[TaskTrace]]:
    """Submit work from two senders and capture send, start, and completion times."""
    result_queue: "queue.Queue[InferenceOutcome]" = queue.Queue()
    start_gate = threading.Event()
    scheduled_start = [0.0]
    traces: List[TaskTrace] = []
    trace_lock = threading.Lock()

    def send(sender_id: int, jobs: List[Tuple[int, str, np.ndarray]]):
        start_gate.wait()
        for round_idx, (worker_idx, image_name, arr) in enumerate(jobs):
            if interval_ms is not None:
                deadline = scheduled_start[0] + round_idx * interval_ms / 1000.0
                time.sleep(max(0.0, deadline - time.perf_counter()))
            trace = TaskTrace(
                request_group=round_idx + 1,
                image_name=image_name,
                sender_id=sender_id,
                worker_id=worker_idx,
                sent_ms=(time.perf_counter() - scheduled_start[0]) * 1000.0,
            )

            def record_start(actual_worker_id: int, task_trace: TaskTrace = trace):
                task_trace.processing_started_ms = (time.perf_counter() - scheduled_start[0]) * 1000.0
                task_trace.worker_id = actual_worker_id

            def record_complete(actual_worker_id: int, task_trace: TaskTrace = trace):
                task_trace.processing_completed_ms = (time.perf_counter() - scheduled_start[0]) * 1000.0
                task_trace.worker_id = actual_worker_id

            with trace_lock:
                traces.append(trace)
            workers[worker_idx].submit(
                f"request_group_{round_idx + 1}:{image_name}",
                arr,
                result_queue,
                record_start,
                record_complete,
            )

    senders = [threading.Thread(target=send, args=(sender_id, jobs)) for sender_id, jobs in enumerate(sender_jobs)]
    for sender in senders:
        sender.start()
    scheduled_start[0] = time.perf_counter()
    start_gate.set()

    group_end_times: Dict[int, float] = {}
    total_jobs = sum(len(jobs) for jobs in sender_jobs)
    for _ in range(total_jobs):
        result = _get_outcome(result_queue, "two-sender inference")
        group_match = re.match(r"request_group_(\d+):", result.image_name)
        if group_match is None:
            raise RuntimeError(f"Unexpected sender result name: {result.image_name}")
        group_end_times[int(group_match.group(1))] = time.perf_counter()

    for sender in senders:
        sender.join(timeout=RESULT_TIMEOUT_SECONDS)
        if sender.is_alive():
            raise TimeoutError("Sender thread did not exit")

    results = [
        InferenceResult(
            image_name=f"request_group_{round_idx}",
            latency_ms=(end_time - (scheduled_start[0] + (round_idx - 1) * (interval_ms or 0) / 1000.0)) * 1000.0,
        )
        for round_idx, end_time in sorted(group_end_times.items())
    ]
    return results, sorted(traces, key=lambda trace: (trace.request_group, trace.sender_id))


def scenario1_sequential(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    gpu_index: int,
) -> ScenarioResult:
    session = create_session(model_path, ep)
    worker = SessionWorker(0, session, input_name)
    worker.start()
    monitor = GPUMonitor(gpu_index)
    monitor.start()
    try:
        assignments = [(0, name, arr) for name, arr, _ in prepared_images]
        results = run_inference_batch([worker], assignments)
    finally:
        monitor.stop()
        monitor.close()
        worker.stop()
        worker.join(timeout=1.0)
    return ScenarioResult(
        scenario_name="scenario1_sequential",
        details="single_session_sequential_sender",
        per_image=results,
        peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
        peak_bandwidth_percent=monitor.peak_bandwidth_percent,
    )


def scenario1b_batched(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    target_shape: Sequence[Optional[int]],
    dim_names: Sequence[Optional[str]],
    gpu_index: int,
) -> ScenarioResult:
    """Run sequential inference using the static batch size declared by the model input."""
    batch_axis = get_batch_axis(target_shape, dim_names)
    batch_size = target_shape[batch_axis] if batch_axis is not None else None
    if batch_axis is None or batch_size is None or batch_size < 2:
        return ScenarioResult(
            scenario_name="scenario1b_batched",
            details="unsupported: model input has no static batch dimension greater than 1",
            per_image=[],
            peak_memory_mb=0.0,
            peak_bandwidth_percent=0.0,
        )

    session = create_session(model_path, ep)
    monitor = GPUMonitor(gpu_index)
    monitor.start()
    results: List[InferenceResult] = []
    try:
        for batch_index, start in enumerate(range(0, len(prepared_images), batch_size), start=1):
            items = prepared_images[start : start + batch_size]
            batch = assemble_image_batch([item[1] for item in items], batch_axis, dim_names, batch_size)
            started = time.perf_counter()
            session.run(None, {input_name: batch})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            names = ",".join(item[0] for item in items)
            results.append(InferenceResult(f"batch_{batch_index}_images_{len(items)}:{names}", elapsed_ms))
    finally:
        monitor.stop()
        monitor.close()
    return ScenarioResult(
        scenario_name="scenario1b_batched",
        details=f"static_batch_size_{batch_size}; final batch is zero-padded when incomplete",
        per_image=results,
        peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
        peak_bandwidth_percent=monitor.peak_bandwidth_percent,
    )


def scenario2_task_concurrency(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    max_concurrency: int,
    gpu_index: int,
) -> List[ScenarioResult]:
    all_results: List[ScenarioResult] = []
    for concurrency in range(2, max_concurrency + 1):
        print_status(f"scenario2 running concurrency={concurrency}/{max_concurrency}")
        # Use a fresh session for every level so allocator arenas from a lower
        # concurrency test do not inflate or fragment the next level's memory use.
        session = create_session(model_path, ep)
        monitor = GPUMonitor(gpu_index)
        monitor.start()
        try:
            results = run_direct_concurrent_groups(session, input_name, prepared_images, concurrency)
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            print_status(f"scenario2 OOM detected at concurrency={concurrency}, stopping task scaling")
            all_results.append(
                ScenarioResult(
                    scenario_name=f"scenario2_task_concurrency_{concurrency}_oom",
                    details="OOM detected and recovered",
                    per_image=[],
                    peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
                    peak_bandwidth_percent=monitor.peak_bandwidth_percent,
                )
            )
            break
        finally:
            monitor.stop()
            monitor.close()
            del session
            gc.collect()
        all_results.append(
            ScenarioResult(
                scenario_name=f"scenario2_task_concurrency_{concurrency}",
                details=f"single_session_{concurrency}_concurrent_sender_threads",
                per_image=results,
                peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
                peak_bandwidth_percent=monitor.peak_bandwidth_percent,
            )
        )
        print_status(f"scenario2 completed concurrency={concurrency}/{max_concurrency}")
    return all_results


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or "cuda error 2" in msg
        or "bad allocation" in msg
        or "bad_alloc" in msg
        or "cannot allocate memory" in msg
    )


def scenario3_session_concurrency(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    max_sessions: int,
    task_concurrency: int,
    gpu_index: int,
) -> List[ScenarioResult]:
    scenario_results: List[ScenarioResult] = []
    for session_count in range(2, max_sessions + 1):
        print_status(f"scenario3 running sessions={session_count}/{max_sessions}")
        workers: List[SessionWorker] = []
        try:
            for i in range(session_count):
                session = create_session(model_path, ep)
                worker = SessionWorker(i, session, input_name)
                worker.start()
                workers.append(worker)

            monitor = GPUMonitor(gpu_index)
            monitor.start()
            try:
                results = run_worker_concurrent_groups(workers, prepared_images, task_concurrency)
            finally:
                monitor.stop()
                monitor.close()

            scenario_results.append(
                ScenarioResult(
                    scenario_name=f"scenario3_session_concurrency_{session_count}",
                    details=f"task_concurrency_{task_concurrency}_{session_count}_sessions",
                    per_image=results,
                    peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
                    peak_bandwidth_percent=monitor.peak_bandwidth_percent,
                )
            )
            print_status(f"scenario3 completed sessions={session_count}/{max_sessions}")
        except Exception as exc:
            if _is_oom_error(exc):
                print_status(f"scenario3 OOM detected at sessions={session_count}, stopping session scaling")
                scenario_results.append(
                    ScenarioResult(
                        scenario_name=f"scenario3_session_concurrency_{session_count}_oom",
                        details="OOM detected and recovered",
                        per_image=[],
                        peak_memory_mb=0.0,
                        peak_bandwidth_percent=0.0,
                    )
                )
                time.sleep(1.0)
                break
            raise
        finally:
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join(timeout=1.0)
    return scenario_results


def scenario4_fixed_interval(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    interval_ms: int,
    gpu_index: int,
) -> List[ScenarioResult]:
    if len(prepared_images) < 2:
        raise RuntimeError("Scenario 4 requires at least 2 images")

    def run_subscenario(name: str, workers: List[SessionWorker], fixed_map: Optional[Dict[int, int]] = None):
        print_status(f"scenario4 running sub-scenario={name}")
        monitor = GPUMonitor(gpu_index)
        monitor.start()
        try:
            sender_jobs = [[], []]
            for i, (img_name, arr, _) in enumerate(prepared_images):
                sender_id = i % 2
                if fixed_map is not None:
                    worker_idx = fixed_map[sender_id]
                else:
                    worker_idx = i % len(workers)
                sender_jobs[sender_id].append((worker_idx, f"{img_name}#sender{sender_id}", arr))

            results, task_traces = run_two_sender_jobs(workers, sender_jobs, interval_ms)
        finally:
            monitor.stop()
            monitor.close()

        result = ScenarioResult(
            scenario_name=name,
            details=f"fixed_interval_{interval_ms}ms",
            per_image=results,
            peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
            peak_bandwidth_percent=monitor.peak_bandwidth_percent,
            task_traces=task_traces,
        )
        print_status(f"scenario4 completed sub-scenario={name}")
        return result

    single_session = create_session(model_path, ep)
    worker_a = SessionWorker(0, single_session, input_name)
    worker_a.start()

    session_0 = create_session(model_path, ep)
    session_1 = create_session(model_path, ep)
    worker_b0 = SessionWorker(0, session_0, input_name)
    worker_b1 = SessionWorker(1, session_1, input_name)
    worker_b0.start()
    worker_b1.start()

    try:
        r1 = run_subscenario("scenario4a_single_session_fixed_interval", [worker_a])
        r2 = run_subscenario(
            "scenario4b_two_sessions_fixed_sender_mapping",
            [worker_b0, worker_b1],
            fixed_map={0: 0, 1: 1},
        )
    finally:
        for w in [worker_a, worker_b0, worker_b1]:
            w.stop()
        for w in [worker_a, worker_b0, worker_b1]:
            w.join(timeout=1.0)
    return [r1, r2]


def scenario5_all_sessions_same_image(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    session_count: int,
    gpu_index: int,
) -> ScenarioResult:
    """Run every image once per session concurrently, using independent sessions."""
    print_status(f"scenario5 running sessions={session_count}")
    workers: List[SessionWorker] = []
    monitor = GPUMonitor(gpu_index)
    try:
        for worker_id in range(session_count):
            worker = SessionWorker(worker_id, create_session(model_path, ep), input_name)
            worker.start()
            workers.append(worker)
        monitor.start()
        results = run_same_image_all_sessions(workers, prepared_images)
    finally:
        monitor.stop()
        monitor.close()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=1.0)
    print_status(f"scenario5 completed sessions={session_count}")
    return ScenarioResult(
        scenario_name=f"scenario5_same_image_{session_count}_sessions",
        details=f"single_image_sent_to_all_{session_count}_sessions",
        per_image=results,
        peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
        peak_bandwidth_percent=monitor.peak_bandwidth_percent,
    )


def get_environment_info(
    model_path: str,
    first_image_path: str,
    original_image_shape: Tuple[int, ...],
    model_input_shape: Tuple[int, ...],
) -> Dict[str, Any]:
    cuda_version = "unknown"
    trt_version = "unknown"

    try:
        smi = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi.stdout)
        if m:
            cuda_version = m.group(1)
    except Exception:
        pass

    try:
        import tensorrt as trt  # type: ignore

        trt_version = trt.__version__
    except Exception:
        pass

    return {
        "cuda_version": cuda_version,
        "tensorrt_version": trt_version,
        "model_size_bytes": os.path.getsize(model_path),
        "image_size_bytes": os.path.getsize(first_image_path),
        "original_image_shape": str(original_image_shape),
        "model_input_shape": str(model_input_shape),
    }


def write_results_to_excel(
    output_path: str,
    env_info: Dict[str, Any],
    scenario_results: List[ScenarioResult],
):
    if Workbook is None:
        raise RuntimeError("openpyxl is required to run this tool")
    wb = Workbook()
    wb.remove(wb.active)

    for sr in scenario_results:
        sheet_name = sr.scenario_name[:31]
        ws = wb.create_sheet(title=sheet_name)
        trimmed_latencies = [result.latency_ms for result in sr.per_image[1:-1]]
        trimmed_mean = sum(trimmed_latencies) / len(trimmed_latencies) if trimmed_latencies else "N/A"
        trimmed_max = max(trimmed_latencies) if trimmed_latencies else "N/A"
        ws.append(["scenario_name", sr.scenario_name])
        ws.append(["details", sr.details])
        ws.append(["cuda_version", env_info["cuda_version"]])
        ws.append(["tensorrt_version", env_info["tensorrt_version"]])
        ws.append(["model_size_bytes", env_info["model_size_bytes"]])
        ws.append(["image_size_bytes", env_info["image_size_bytes"]])
        ws.append(["original_image_shape", env_info["original_image_shape"]])
        ws.append(["model_input_shape", env_info["model_input_shape"]])
        ws.append(["peak_memory_mb", sr.peak_memory_mb])
        ws.append(["peak_bandwidth_percent", sr.peak_bandwidth_percent])
        ws.append(["time_average_excluding_first_and_last_ms", trimmed_mean])
        ws.append(["time_max_excluding_first_and_last_ms", trimmed_max])
        ws.append([])
        if sr.scenario_name.startswith(("scenario2_", "scenario3_", "scenario4")):
            ws.append(["request_group", "processing_time_ms"])
        else:
            ws.append(["image_name", "latency_ms"])
        for row in sr.per_image:
            ws.append([row.image_name, row.latency_ms])
        if sr.task_traces:
            ws.append([])
            ws.append(
                [
                    "request_group",
                    "image_name",
                    "sender_id",
                    "worker_id",
                    "sent_ms",
                    "processing_started_ms",
                    "processing_completed_ms",
                    "queue_wait_ms",
                    "processing_time_ms",
                ]
            )
            for trace in sr.task_traces:
                queue_wait_ms = None
                processing_time_ms = None
                if trace.processing_started_ms is not None:
                    queue_wait_ms = trace.processing_started_ms - trace.sent_ms
                if trace.processing_completed_ms is not None and trace.processing_started_ms is not None:
                    processing_time_ms = trace.processing_completed_ms - trace.processing_started_ms
                ws.append(
                    [
                        trace.request_group,
                        trace.image_name,
                        trace.sender_id,
                        trace.worker_id,
                        trace.sent_ms,
                        trace.processing_started_ms,
                        trace.processing_completed_ms,
                        queue_wait_ms,
                        processing_time_ms,
                    ]
                )

    wb.save(output_path)


def print_single_inference_verification(session: Any, input_name: str, image_name: str, image_arr: np.ndarray):
    start = time.perf_counter()
    out = session.run(None, {input_name: image_arr})
    elapsed = (time.perf_counter() - start) * 1000.0
    summary = []
    for i, item in enumerate(out):
        arr = np.asarray(item)
        summary.append(
            {
                "index": i,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "sample": arr.ravel()[:5].tolist(),
            }
        )
    print(f"[VERIFY] single-image inference success: image={image_name}, latency_ms={elapsed:.3f}, output={summary}")


def print_status(message: str):
    print(f"[STATUS] {message}", flush=True)


def main():
    args = parse_args()
    if ort is None:
        raise RuntimeError("onnxruntime is required to run this tool")
    if not os.path.isdir(args.image_dir):
        raise RuntimeError(f"image-dir does not exist or is not a directory: {args.image_dir}")
    if not os.path.isfile(args.model):
        raise RuntimeError(f"model does not exist or is not a file: {args.model}")
    output_dir = os.path.dirname(os.path.abspath(args.output_excel))
    if not os.path.isdir(output_dir):
        raise RuntimeError(f"output directory does not exist: {output_dir}")

    print_status("GPU benchmark started")
    print_status(f"execution_provider={args.ep}, image_dir={args.image_dir}, model={args.model}")
    print_status("loading model and probing input metadata")
    probe_session = create_session(args.model, args.ep)
    input_meta = probe_session.get_inputs()[0]
    input_name = input_meta.name
    onnx_input_name, onnx_shape, onnx_dim_names = get_model_info(args.model)
    target_shape = onnx_shape if onnx_shape else [d if isinstance(d, int) else None for d in input_meta.shape]
    dim_names = onnx_dim_names if onnx_dim_names else [None] * len(target_shape)
    target_dtype = resolve_numpy_dtype(input_meta.type)
    print_status("loading and preparing images")
    prepared_images = load_and_prepare_images(
        args.image_dir,
        target_shape,
        dim_names,
        target_dtype,
        args.max_images,
        args.pad,
    )
    print_status(f"prepared_images={len(prepared_images)}")
    first_image_path = os.path.join(args.image_dir, prepared_images[0][0])

    print(
        f"[INFO] model_input(ort)={input_name}, model_input(onnx)={onnx_input_name}, "
        f"shape={target_shape}, dim_names={dim_names}, dtype={target_dtype}"
    )
    print_status(
        f"first_image_original_shape={prepared_images[0][2]}, model_input_shape={prepared_images[0][1].shape}, pad={args.pad}"
    )
    print_single_inference_verification(probe_session, input_name, prepared_images[0][0], prepared_images[0][1])

    all_results: List[ScenarioResult] = []
    print_status("running scenario 1: sequential")
    all_results.append(scenario1_sequential(args.model, args.ep, input_name, prepared_images, args.gpu_index))
    print_status("scenario 1 completed")
    print_status("running scenario 1b: batched")
    all_results.append(
        scenario1b_batched(
            args.model,
            args.ep,
            input_name,
            prepared_images,
            target_shape,
            dim_names,
            args.gpu_index,
        )
    )
    print_status("scenario 1b completed")
    print_status("running scenario 2: task concurrency")
    all_results.extend(
        scenario2_task_concurrency(args.model, args.ep, input_name, prepared_images, args.max_task_concurrency, args.gpu_index)
    )
    print_status("scenario 2 completed")
    print_status("running scenario 3: session concurrency")
    all_results.extend(
        scenario3_session_concurrency(
            args.model,
            args.ep,
            input_name,
            prepared_images,
            args.max_session_concurrency,
            args.max_task_concurrency,
            args.gpu_index,
        )
    )
    print_status("scenario 3 completed")
    print_status("running scenario 4: fixed interval")
    all_results.extend(
        scenario4_fixed_interval(args.model, args.ep, input_name, prepared_images, args.interval_ms, args.gpu_index)
    )
    print_status("scenario 4 completed")
    print_status("running scenario 5: same image on all sessions")
    all_results.append(
        scenario5_all_sessions_same_image(
            args.model,
            args.ep,
            input_name,
            prepared_images,
            args.max_session_concurrency,
            args.gpu_index,
        )
    )
    print_status("scenario 5 completed")

    print_status("collecting environment information")
    env_info = get_environment_info(
        args.model,
        first_image_path,
        prepared_images[0][2],
        tuple(prepared_images[0][1].shape),
    )
    print_status("writing results to Excel")
    write_results_to_excel(args.output_excel, env_info, all_results)
    print(f"[DONE] results saved to {args.output_excel}")


if __name__ == "__main__":
    main()
