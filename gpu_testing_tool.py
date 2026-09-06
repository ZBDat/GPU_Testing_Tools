import argparse
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class InferenceResult:
    image_name: str
    latency_ms: float


@dataclass
class ScenarioResult:
    scenario_name: str
    details: str
    per_image: List[InferenceResult]
    peak_memory_mb: float
    peak_bandwidth_percent: float


class GPUMonitor:
    def __init__(self):
        self.enabled = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_memory_bytes = 0
        self.peak_bandwidth_percent = 0.0
        self._handle = None

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.enabled = True
            except Exception:
                self.enabled = False

    def start(self):
        if not self.enabled:
            return
        self._stop.clear()
        self.peak_memory_bytes = 0
        self.peak_bandwidth_percent = 0.0
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
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle).used
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).memory
                self.peak_memory_bytes = max(self.peak_memory_bytes, mem)
                self.peak_bandwidth_percent = max(self.peak_bandwidth_percent, float(util))
            except Exception:
                return
            time.sleep(0.01)


class SessionWorker(threading.Thread):
    def __init__(self, worker_id: int, session: Any, input_name: str):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.session = session
        self.input_name = input_name
        self.tasks: "queue.Queue[Tuple[Optional[str], Optional[np.ndarray], Optional[queue.Queue]]]" = queue.Queue()
        self._stop = threading.Event()

    def submit(self, image_name: str, arr: np.ndarray, result_queue: queue.Queue):
        self.tasks.put((image_name, arr, result_queue))

    def run(self):
        while not self._stop.is_set():
            image_name, arr, result_queue = self.tasks.get()
            if image_name is None or arr is None or result_queue is None:
                return
            start = time.perf_counter()
            self.session.run(None, {self.input_name: arr})
            latency = (time.perf_counter() - start) * 1000.0
            result_queue.put(InferenceResult(image_name=image_name, latency_ms=latency))

    def stop(self):
        self._stop.set()
        self.tasks.put((None, None, None))


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
    return parser.parse_args()


def get_model_info(model_path: str) -> Tuple[str, List[Optional[int]]]:
    if onnx is None:
        return "unknown", []
    model = onnx.load(model_path)
    input0 = model.graph.input[0]
    dims: List[Optional[int]] = []
    for d in input0.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append(None)
    return input0.name, dims


def resolve_numpy_dtype(ort_type: str) -> np.dtype:
    if ort_type not in ORT_TYPE_TO_NUMPY:
        raise ValueError(f"Unsupported ONNX input dtype: {ort_type}")
    return np.dtype(ORT_TYPE_TO_NUMPY[ort_type])


def normalize_shape(arr: np.ndarray, target_shape: Sequence[Optional[int]]) -> np.ndarray:
    out = arr
    while out.ndim < len(target_shape):
        out = np.expand_dims(out, axis=0)
    if out.ndim > len(target_shape):
        while out.ndim > len(target_shape):
            out = out[0]

    slices = []
    pads = []
    for dim_size, target_dim in zip(out.shape, target_shape):
        effective_target = dim_size if target_dim is None or target_dim <= 0 else target_dim
        if dim_size > effective_target:
            slices.append(slice(0, effective_target))
            pads.append((0, 0))
        elif dim_size < effective_target:
            slices.append(slice(0, dim_size))
            pads.append((0, effective_target - dim_size))
        else:
            slices.append(slice(0, dim_size))
            pads.append((0, 0))

    out = out[tuple(slices)]
    if any(p != (0, 0) for p in pads):
        out = np.pad(out, pads, mode="constant", constant_values=0)
    return out


def load_and_prepare_images(
    image_dir: str,
    target_shape: Sequence[Optional[int]],
    target_dtype: np.dtype,
    max_images: int,
) -> List[Tuple[str, np.ndarray]]:
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
        img = tifffile.imread(path)
        img = normalize_shape(np.asarray(img), target_shape)
        img = img.astype(target_dtype, copy=False)
        prepared.append((f, img))
    return prepared


def create_session(model_path: str, ep: str):
    if ort is None:
        raise RuntimeError("onnxruntime is required to run this tool")
    providers = ["CUDAExecutionProvider"]
    if ep == "tensorrt":
        providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    so = ort.SessionOptions()
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def run_inference_batch(
    workers: List[SessionWorker],
    assignments: List[Tuple[int, str, np.ndarray]],
) -> List[InferenceResult]:
    result_queue: "queue.Queue[InferenceResult]" = queue.Queue()
    for worker_idx, image_name, arr in assignments:
        workers[worker_idx].submit(image_name, arr, result_queue)
    results = [result_queue.get() for _ in assignments]
    return results


def scenario1_sequential(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray]],
) -> ScenarioResult:
    session = create_session(model_path, ep)
    worker = SessionWorker(0, session, input_name)
    worker.start()
    monitor = GPUMonitor()
    monitor.start()
    try:
        assignments = [(0, name, arr) for name, arr in prepared_images]
        results = run_inference_batch([worker], assignments)
    finally:
        monitor.stop()
        worker.stop()
        worker.join(timeout=1.0)
    return ScenarioResult(
        scenario_name="scenario1_sequential",
        details="single_session_sequential_sender",
        per_image=results,
        peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
        peak_bandwidth_percent=monitor.peak_bandwidth_percent,
    )


def scenario2_task_concurrency(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray]],
    max_concurrency: int,
) -> List[ScenarioResult]:
    session = create_session(model_path, ep)
    worker = SessionWorker(0, session, input_name)
    worker.start()
    all_results: List[ScenarioResult] = []
    try:
        for concurrency in range(2, max_concurrency + 1):
            monitor = GPUMonitor()
            monitor.start()
            try:
                assignments = []
                for i, (name, arr) in enumerate(prepared_images):
                    assignments.append((0, f"{name}#sender{i % concurrency}", arr))
                results = run_inference_batch([worker], assignments)
            finally:
                monitor.stop()
            all_results.append(
                ScenarioResult(
                    scenario_name=f"scenario2_task_concurrency_{concurrency}",
                    details=f"single_session_{concurrency}_sender_threads",
                    per_image=results,
                    peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
                    peak_bandwidth_percent=monitor.peak_bandwidth_percent,
                )
            )
    finally:
        worker.stop()
        worker.join(timeout=1.0)
    return all_results


def _is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "cuda error 2" in msg


def scenario3_session_concurrency(
    model_path: str,
    ep: str,
    input_name: str,
    prepared_images: List[Tuple[str, np.ndarray]],
    max_sessions: int,
) -> List[ScenarioResult]:
    scenario_results: List[ScenarioResult] = []
    for session_count in range(2, max_sessions + 1):
        workers: List[SessionWorker] = []
        try:
            for i in range(session_count):
                session = create_session(model_path, ep)
                worker = SessionWorker(i, session, input_name)
                worker.start()
                workers.append(worker)

            assignments = []
            sender_count = 2
            for i, (name, arr) in enumerate(prepared_images):
                sender_id = i % sender_count
                worker_idx = i % session_count
                assignments.append((worker_idx, f"{name}#sender{sender_id}", arr))

            monitor = GPUMonitor()
            monitor.start()
            try:
                results = run_inference_batch(workers, assignments)
            finally:
                monitor.stop()

            scenario_results.append(
                ScenarioResult(
                    scenario_name=f"scenario3_session_concurrency_{session_count}",
                    details=f"two_sender_threads_{session_count}_sessions",
                    per_image=results,
                    peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
                    peak_bandwidth_percent=monitor.peak_bandwidth_percent,
                )
            )
        except Exception as exc:
            if _is_oom_error(exc):
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
    prepared_images: List[Tuple[str, np.ndarray]],
    interval_ms: int,
) -> List[ScenarioResult]:
    if len(prepared_images) < 2:
        raise RuntimeError("Scenario 4 requires at least 2 images")

    def run_subscenario(name: str, workers: List[SessionWorker], fixed_map: Optional[Dict[int, int]] = None):
        result_queue: "queue.Queue[InferenceResult]" = queue.Queue()
        monitor = GPUMonitor()
        monitor.start()
        try:
            sender_jobs = [[], []]
            for i, (img_name, arr) in enumerate(prepared_images):
                sender_id = i % 2
                if fixed_map is not None:
                    worker_idx = fixed_map[sender_id]
                else:
                    worker_idx = i % len(workers)
                sender_jobs[sender_id].append((worker_idx, f"{img_name}#sender{sender_id}", arr))

            sent_counter = {"count": 0}
            send_lock = threading.Lock()

            def sender_thread_fn(jobs: List[Tuple[int, str, np.ndarray]]):
                for worker_idx, image_name, arr in jobs:
                    workers[worker_idx].submit(image_name, arr, result_queue)
                    with send_lock:
                        sent_counter["count"] += 1
                    time.sleep(interval_ms / 1000.0)

            senders = [threading.Thread(target=sender_thread_fn, args=(sender_jobs[i],), daemon=True) for i in range(2)]
            for t in senders:
                t.start()
            for t in senders:
                t.join()

            total = sent_counter["count"]
            results = [result_queue.get() for _ in range(total)]
        finally:
            monitor.stop()

        return ScenarioResult(
            scenario_name=name,
            details=f"fixed_interval_{interval_ms}ms",
            per_image=results,
            peak_memory_mb=monitor.peak_memory_bytes / (1024 * 1024),
            peak_bandwidth_percent=monitor.peak_bandwidth_percent,
        )

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


def get_environment_info(model_path: str, first_image_path: str, first_image_shape: Tuple[int, ...]) -> Dict[str, Any]:
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
        "image_shape": str(first_image_shape),
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
        ws.append(["scenario_name", sr.scenario_name])
        ws.append(["details", sr.details])
        ws.append(["cuda_version", env_info["cuda_version"]])
        ws.append(["tensorrt_version", env_info["tensorrt_version"]])
        ws.append(["model_size_bytes", env_info["model_size_bytes"]])
        ws.append(["image_size_bytes", env_info["image_size_bytes"]])
        ws.append(["image_shape", env_info["image_shape"]])
        ws.append(["peak_memory_mb", sr.peak_memory_mb])
        ws.append(["peak_bandwidth_percent", sr.peak_bandwidth_percent])
        ws.append([])
        ws.append(["image_name", "latency_ms"])
        for row in sr.per_image:
            ws.append([row.image_name, row.latency_ms])

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


def main():
    args = parse_args()
    if ort is None:
        raise RuntimeError("onnxruntime is required to run this tool")

    probe_session = create_session(args.model, args.ep)
    input_meta = probe_session.get_inputs()[0]
    input_name = input_meta.name
    onnx_input_name, onnx_shape = get_model_info(args.model)
    target_shape = onnx_shape if onnx_shape else [d if isinstance(d, int) else None for d in input_meta.shape]
    target_dtype = resolve_numpy_dtype(input_meta.type)
    prepared_images = load_and_prepare_images(args.image_dir, target_shape, target_dtype, args.max_images)
    first_image_path = os.path.join(args.image_dir, prepared_images[0][0])

    print(
        f"[INFO] model_input(ort)={input_name}, model_input(onnx)={onnx_input_name}, "
        f"shape={target_shape}, dtype={target_dtype}"
    )
    print_single_inference_verification(probe_session, input_name, prepared_images[0][0], prepared_images[0][1])

    all_results: List[ScenarioResult] = []
    all_results.append(scenario1_sequential(args.model, args.ep, input_name, prepared_images))
    all_results.extend(
        scenario2_task_concurrency(args.model, args.ep, input_name, prepared_images, args.max_task_concurrency)
    )
    all_results.extend(
        scenario3_session_concurrency(args.model, args.ep, input_name, prepared_images, args.max_session_concurrency)
    )
    all_results.extend(
        scenario4_fixed_interval(args.model, args.ep, input_name, prepared_images, args.interval_ms)
    )

    env_info = get_environment_info(args.model, first_image_path, tuple(prepared_images[0][1].shape))
    write_results_to_excel(args.output_excel, env_info, all_results)
    print(f"[DONE] results saved to {args.output_excel}")


if __name__ == "__main__":
    main()
