import argparse
import os
import queue
import threading
import time
from typing import List, Tuple

import numpy as np

from gpu_testing_tool import (
    RESULT_TIMEOUT_SECONDS,
    SessionWorker,
    create_session,
    get_model_info,
    load_and_prepare_images,
    ort,
    print_single_inference_verification,
    print_status,
    resolve_numpy_dtype,
)


WORKER_COUNT = 4
SENDER_COUNT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously run ONNX inference using two senders and four independent session workers."
    )
    parser.add_argument("--image-dir", required=True, help="Directory with TIFF images")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--ep", choices=["cuda", "tensorrt"], default="cuda", help="Execution provider")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap of images to use")
    parser.add_argument("--duration-seconds", type=float, default=0, help="Run duration; 0 runs until Ctrl+C")
    parser.add_argument("--warmup-rounds", type=int, default=3, help="Rounds to run before reporting steady state")
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error("--max-images must be non-negative")
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be non-negative")
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be non-negative")
    if not os.path.isdir(args.image_dir):
        parser.error(f"--image-dir does not exist or is not a directory: {args.image_dir}")
    if not os.path.isfile(args.model):
        parser.error(f"--model does not exist or is not a file: {args.model}")
    return args


def run_round(
    workers: List[SessionWorker],
    prepared_images: List[Tuple[str, np.ndarray, Tuple[int, ...]]],
    round_number: int,
) -> int:
    """Queue one request per worker and wait before the next round."""
    result_queue: queue.Queue = queue.Queue()
    start_gate = threading.Event()
    sender_jobs = [[], []]

    for worker_index in range(WORKER_COUNT):
        name, image, _ = prepared_images[(round_number * WORKER_COUNT + worker_index) % len(prepared_images)]
        sender_jobs[worker_index % SENDER_COUNT].append((worker_index, name, image))

    def send(jobs):
        start_gate.wait()
        for worker_index, name, image in jobs:
            workers[worker_index].submit(f"round_{round_number}:{name}", image, result_queue)

    senders = [threading.Thread(target=send, args=(jobs,)) for jobs in sender_jobs]
    for sender in senders:
        sender.start()
    start_gate.set()
    for sender in senders:
        sender.join(timeout=RESULT_TIMEOUT_SECONDS)
        if sender.is_alive():
            raise TimeoutError(f"Sender did not submit round {round_number}")

    for _ in range(WORKER_COUNT):
        try:
            outcome = result_queue.get(timeout=RESULT_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for round {round_number}") from exc
        if outcome.error is not None:
            raise RuntimeError(f"Inference failed in round {round_number}: {outcome.error}") from outcome.error
    return WORKER_COUNT


def main():
    args = parse_args()
    if ort is None:
        raise RuntimeError("onnxruntime is required to run this tool")

    print_status("Nsight runner loading model and preparing images")
    probe_session = create_session(args.model, args.ep)
    input_meta = probe_session.get_inputs()[0]
    input_name = input_meta.name
    _, onnx_shape, dim_names = get_model_info(args.model)
    target_shape = onnx_shape if onnx_shape else [d if isinstance(d, int) else None for d in input_meta.shape]
    if not dim_names:
        dim_names = [None] * len(target_shape)
    prepared_images = load_and_prepare_images(
        args.image_dir,
        target_shape,
        dim_names,
        resolve_numpy_dtype(input_meta.type),
        args.max_images,
    )
    print_status(f"prepared_images={len(prepared_images)}, worker_count={WORKER_COUNT}, sender_count={SENDER_COUNT}")
    print_single_inference_verification(probe_session, input_name, prepared_images[0][0], prepared_images[0][1])

    workers = []
    try:
        for worker_index in range(WORKER_COUNT):
            worker = SessionWorker(worker_index, create_session(args.model, args.ep), input_name)
            worker.start()
            workers.append(worker)

        deadline = time.perf_counter() + args.duration_seconds if args.duration_seconds else None
        completed_requests = 0
        round_number = 0
        steady_start = None
        print_status("continuous inference started; press Ctrl+C to stop")
        while deadline is None or time.perf_counter() < deadline:
            completed_requests += run_round(workers, prepared_images, round_number)
            round_number += 1
            if round_number == args.warmup_rounds:
                completed_requests = 0
                steady_start = time.perf_counter()
                print_status("warmup complete; steady-state measurement started")
            if steady_start is not None and round_number % 100 == 0:
                elapsed = time.perf_counter() - steady_start
                print_status(f"steady_state_requests={completed_requests}, throughput={completed_requests / elapsed:.2f} req/s")
    except KeyboardInterrupt:
        print_status("interrupted by user")
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=RESULT_TIMEOUT_SECONDS)

    if steady_start is not None:
        elapsed = time.perf_counter() - steady_start
        print_status(f"finished: steady_state_requests={completed_requests}, throughput={completed_requests / elapsed:.2f} req/s")
    else:
        print_status("finished before warmup completed")


if __name__ == "__main__":
    main()
