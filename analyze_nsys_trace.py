import argparse
import csv
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_NSYS = Path(
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
)


def find_nsys() -> str:
    executable = shutil.which("nsys") or shutil.which("nsys.exe")
    if executable:
        return executable
    if DEFAULT_NSYS.is_file():
        return str(DEFAULT_NSYS)
    raise RuntimeError("Cannot find nsys. Add it to PATH or pass --nsys-path.")


def run(command: Sequence[str]) -> None:
    print("[COMMAND]", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def export_sqlite(nsys: str, report_path: Path, sqlite_path: Path) -> None:
    run(
        [
            nsys,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(report_path),
        ]
    )


def query(connection: sqlite3.Connection, sql: str, params: Iterable = ()) -> List[Tuple]:
    return connection.execute(sql, tuple(params)).fetchall()


def select_worker_streams(connection: sqlite3.Connection, worker_count: int) -> List[int]:
    rows = query(
        connection,
        """
        SELECT streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId
        ORDER BY COUNT(*) DESC
        LIMIT ?
        """,
        [worker_count],
    )
    streams = [row[0] for row in rows]
    if len(streams) != worker_count:
        raise RuntimeError(f"Expected {worker_count} active CUDA streams, found {len(streams)}")
    return streams


def placeholders(values: Sequence[int]) -> str:
    return ",".join("?" for _ in values)


def analyze(sqlite_path: Path, report_path: Path, worker_count: int) -> Path:
    connection = sqlite3.connect(sqlite_path)
    try:
        streams = select_worker_streams(connection, worker_count)
        stream_placeholders = placeholders(streams)
        summary = query(
            connection,
            f"""
            WITH events AS (
                SELECT start AS ts, 1 AS delta FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId IN ({stream_placeholders})
                UNION ALL
                SELECT end AS ts, -1 AS delta FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId IN ({stream_placeholders})
            ), points AS (
                SELECT ts, SUM(delta) AS delta FROM events GROUP BY ts
            ), timeline AS (
                SELECT ts,
                       SUM(delta) OVER (ORDER BY ts ROWS UNBOUNDED PRECEDING) AS active_kernels,
                       LEAD(ts) OVER (ORDER BY ts) - ts AS duration
                FROM points
            )
            SELECT MIN(ts), MAX(ts), SUM(duration),
                   SUM(CASE WHEN active_kernels > 0 THEN duration ELSE 0 END),
                   SUM(active_kernels * duration),
                   MAX(active_kernels)
            FROM timeline
            WHERE duration IS NOT NULL
            """,
            streams * 2,
        )[0]
        first_ns, last_ns, span_ns, busy_ns, total_kernel_ns, max_active = summary
        avg_active = total_kernel_ns / span_ns if span_ns else 0.0

        distribution = query(
            connection,
            f"""
            WITH events AS (
                SELECT start AS ts, 1 AS delta FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId IN ({stream_placeholders})
                UNION ALL
                SELECT end AS ts, -1 AS delta FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE streamId IN ({stream_placeholders})
            ), points AS (
                SELECT ts, SUM(delta) AS delta FROM events GROUP BY ts
            ), timeline AS (
                SELECT SUM(delta) OVER (ORDER BY ts ROWS UNBOUNDED PRECEDING) AS active_kernels,
                       LEAD(ts) OVER (ORDER BY ts) - ts AS duration
                FROM points
            )
            SELECT active_kernels, SUM(duration)
            FROM timeline
            WHERE duration IS NOT NULL
            GROUP BY active_kernels
            ORDER BY active_kernels
            """,
            streams * 2,
        )
        stream_rows = query(
            connection,
            """
            SELECT streamId, COUNT(*), SUM(end - start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE streamId IN (%s)
            GROUP BY streamId
            ORDER BY streamId
            """ % stream_placeholders,
            streams,
        )
        api_rows = query(
            connection,
            """
            SELECT REPLACE(s.value, '_v3020', ''), COUNT(*), SUM(r.end - r.start)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id = r.nameId
            GROUP BY s.value
            ORDER BY SUM(r.end - r.start) DESC
            LIMIT 10
            """,
        )
        copy_rows = query(
            connection,
            """
            SELECT src.label || ' to ' || dst.label, COUNT(*), SUM(m.end - m.start), SUM(m.bytes)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            LEFT JOIN ENUM_CUDA_MEM_KIND src ON src.id = m.srcKind
            LEFT JOIN ENUM_CUDA_MEM_KIND dst ON dst.id = m.dstKind
            GROUP BY src.label, dst.label
            ORDER BY SUM(m.end - m.start) DESC
            """,
        )
    finally:
        connection.close()

    output = report_path.with_name(f"{report_path.stem}_analysis.md")
    one_kernel_percent = next((duration * 100 / span_ns for active, duration in distribution if active == 1), 0.0)
    lines = [
        "# Nsight Systems CUDA Concurrency Analysis",
        "",
        f"- Source report: `{report_path.name}`",
        f"- SQLite export: `{sqlite_path.name}`",
        f"- Selected worker streams: `{', '.join(map(str, streams))}`",
        f"- Kernel trace interval: `{(last_ns - first_ns) / 1e9:.3f} s`",
        "",
        "## Kernel Concurrency",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| GPU busy time | {busy_ns / 1e9:.3f} s |",
        f"| Total kernel time | {total_kernel_ns / 1e9:.3f} s |",
        f"| Average active kernels | {avg_active:.4f} |",
        f"| Maximum active kernels | {max_active} |",
        f"| Time with exactly one active kernel | {one_kernel_percent:.2f}% |",
        "",
        "| Active kernels | Duration (s) | Trace time |",
        "|---:|---:|---:|",
    ]
    lines.extend(f"| {active} | {duration / 1e9:.3f} | {duration * 100 / span_ns:.2f}% |" for active, duration in distribution)
    lines.extend(["", "## Worker Streams", "", "| Stream | Kernels | Kernel time (s) |", "|---:|---:|---:|"])
    lines.extend(f"| {stream} | {count} | {duration / 1e9:.3f} |" for stream, count, duration in stream_rows)
    lines.extend(["", "## CUDA Runtime APIs", "", "| API | Calls | Host duration (s) |", "|---|---:|---:|"])
    lines.extend(f"| `{name}` | {calls} | {duration / 1e9:.3f} |" for name, calls, duration in api_rows)
    lines.extend(["", "## GPU Memory Copies", "", "| Direction | Calls | GPU copy time (s) | Bytes |", "|---|---:|---:|---:|"])
    lines.extend(f"| {direction or 'Unknown'} | {calls} | {duration / 1e9:.3f} | {byte_count} |" for direction, calls, duration, byte_count in copy_rows)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- An average active-kernel count near 1 and a high exactly-one-active percentage indicate effectively serial GPU kernel execution.",
            "- Values materially above 1 show CUDA kernel overlap across the selected worker streams.",
            "- CUDA runtime API totals are host-thread cumulative durations; they can exceed wall-clock duration because multiple threads are included.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Analysis written to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile and analyze Nsight CUDA concurrency traces.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze an existing .nsys-rep trace")
    analyze_parser.add_argument("--report", required=True, type=Path)
    analyze_parser.add_argument("--worker-count", type=int, default=4)
    analyze_parser.add_argument("--nsys-path", default=None)
    profile_parser = subparsers.add_parser("profile", help="Profile the two-sender/four-worker runner and analyze it")
    profile_parser.add_argument("--image-dir", required=True)
    profile_parser.add_argument("--model", required=True)
    profile_parser.add_argument("--ep", choices=["cuda", "tensorrt"], default="cuda")
    profile_parser.add_argument("--duration-seconds", type=float, default=30)
    profile_parser.add_argument("--output", type=Path, default=Path("two_sender_four_worker_trace"))
    profile_parser.add_argument("--nsys-path", default=None)

    args = parser.parse_args()
    nsys = args.nsys_path or find_nsys()
    if args.command == "profile":
        runner = Path(__file__).with_name("nsys_two_sender_four_worker.py")
        report_path = args.output.with_suffix(".nsys-rep")
        run(
            [
                nsys,
                "profile",
                "--trace=cuda,nvtx",
                "--cuda-event-trace=false",
                "--force-overwrite=true",
                "--output",
                str(args.output),
                "python",
                str(runner),
                "--image-dir",
                args.image_dir,
                "--model",
                args.model,
                "--ep",
                args.ep,
                "--duration-seconds",
                str(args.duration_seconds),
            ]
        )
    else:
        report_path = args.report
    if not report_path.is_file():
        raise RuntimeError(f"Nsight report not found: {report_path}")
    sqlite_path = report_path.with_suffix(".sqlite")
    export_sqlite(nsys, report_path, sqlite_path)
    analyze(sqlite_path, report_path, getattr(args, "worker_count", 4))


if __name__ == "__main__":
    main()
