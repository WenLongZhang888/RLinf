#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create a small LeRobot subset with one episode per task.

The script reads a LeRobot dataset, groups episodes by the first task string in
``meta/episodes.jsonl``, selects one episode from each task, and writes a new
LeRobot dataset with contiguous episode/frame indices.

For few-shot LIBERO SFT, a good default is ``--strategy median-length`` because
it avoids unusually short or long demonstrations while staying deterministic.

Example:
    python scripts/sample_one_episode_per_task.py \
            --dataset-path .cache/openpi/physical-intelligence/libero \
            --output-dir .cache/openpi/physical-intelligence/libero_40 \
            --strategy median-length \
            --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=4, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_list(value: object) -> list:
    return value if isinstance(value, list) else [value]


def _resolve_episode_path(
    dataset_path: Path,
    info: dict[str, Any],
    ep_idx: int,
) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    rel_path = data_path.format(
        episode_chunk=ep_idx // chunks_size,
        episode_index=ep_idx,
    )
    return dataset_path / rel_path


def _task_name(episode_meta: dict[str, Any]) -> str:
    tasks = episode_meta.get("tasks") or ["unknown task"]
    return str(tasks[0])


def _select_episode(
    candidates: list[dict[str, Any]],
    *,
    strategy: str,
    rng: random.Random,
) -> dict[str, Any]:
    candidates = sorted(candidates, key=lambda ep: int(ep["episode_index"]))
    if strategy == "first":
        return candidates[0]
    if strategy == "random":
        return rng.choice(candidates)
    if strategy == "median-length":
        lengths = sorted(int(ep.get("length", 0)) for ep in candidates)
        median_length = lengths[len(lengths) // 2]
        return min(
            candidates,
            key=lambda ep: (
                abs(int(ep.get("length", 0)) - median_length),
                int(ep["episode_index"]),
            ),
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def _reindex_episode_stats(
    stats: dict[str, Any],
    *,
    new_ep_idx: int,
    new_task_idx: int,
    new_frame_start: int,
    old_frame_start: int,
) -> dict[str, Any]:
    out = copy.deepcopy(stats)
    frame_offset = new_frame_start - old_frame_start

    if "episode_index" in out:
        count = out["episode_index"].get("count", [1])
        out["episode_index"] = {
            "min": [new_ep_idx],
            "max": [new_ep_idx],
            "mean": [float(new_ep_idx)],
            "std": [0.0],
            "count": count,
        }

    if "task_index" in out:
        count = out["task_index"].get("count", [1])
        out["task_index"] = {
            "min": [new_task_idx],
            "max": [new_task_idx],
            "mean": [float(new_task_idx)],
            "std": [0.0],
            "count": count,
        }

    if "index" in out:
        idx_s = out["index"]
        out["index"] = {
            "min": [v + frame_offset for v in _ensure_list(idx_s.get("min", [0]))],
            "max": [v + frame_offset for v in _ensure_list(idx_s.get("max", [0]))],
            "mean": [v + frame_offset for v in _ensure_list(idx_s.get("mean", [0.0]))],
            "std": idx_s.get("std", [0.0]),
            "count": idx_s.get("count", [1]),
        }

    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a LeRobot subset with one selected episode per task.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Input LeRobot dataset root containing meta/info.json and data/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output LeRobot dataset root to create.",
    )
    parser.add_argument(
        "--strategy",
        choices=["median-length", "random", "first"],
        default="median-length",
        help="Episode selection strategy inside each task group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by --strategy random.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected episodes without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing empty output directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    meta_dir = dataset_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise SystemExit(
            f"Expected a LeRobot dataset at {dataset_path} with meta/info.json "
            "and meta/episodes.jsonl."
        )

    info = _read_json(info_path)
    episodes = _read_jsonl(episodes_path)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_task.setdefault(_task_name(episode), []).append(episode)

    rng = random.Random(args.seed)
    selected = [
        _select_episode(by_task[task], strategy=args.strategy, rng=rng)
        for task in sorted(by_task)
    ]

    print(
        f"[sample] Found {len(episodes)} episodes across {len(by_task)} tasks. "
        f"Selected {len(selected)} episodes with strategy={args.strategy!r}."
    )
    for episode in selected:
        print(
            f"[sample] ep={int(episode['episode_index']):06d} "
            f"len={int(episode.get('length', 0)):04d} task={_task_name(episode)}"
        )

    if args.dry_run:
        return

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Choose a new directory or pass --overwrite."
        )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Use the same environment you use for "
            "RLinf/OpenPI data processing."
        ) from exc

    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    old_episode_stats = {}
    episode_stats_path = meta_dir / "episodes_stats.jsonl"
    if episode_stats_path.is_file():
        old_episode_stats = {
            int(row["episode_index"]): row
            for row in _read_jsonl(episode_stats_path)
            if "episode_index" in row
        }

    selected_tasks = sorted({_task_name(ep) for ep in selected})
    task_to_new_idx = {task: idx for idx, task in enumerate(selected_tasks)}

    total_frames = 0
    output_episodes = []
    output_episode_stats = []
    output_chunks_size = 1000

    for new_ep_idx, episode in enumerate(selected):
        old_ep_idx = int(episode["episode_index"])
        task = _task_name(episode)
        new_task_idx = task_to_new_idx[task]
        source_parquet = _resolve_episode_path(dataset_path, info, old_ep_idx)
        if not source_parquet.is_file():
            raise SystemExit(f"Missing episode parquet: {source_parquet}")

        table = pq.read_table(source_parquet)
        df = table.to_pandas()
        n_frames = len(df)
        old_frame_start = int(df["index"].min()) if "index" in df.columns else 0

        df["episode_index"] = new_ep_idx
        df["index"] = range(total_frames, total_frames + n_frames)
        if "frame_index" in df.columns:
            df["frame_index"] = range(n_frames)
        df["task_index"] = new_task_idx

        chunk_idx = new_ep_idx // output_chunks_size
        chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        output_parquet = chunk_dir / f"episode_{new_ep_idx:06d}.parquet"

        new_table = pa.Table.from_pandas(df, preserve_index=False)
        if table.schema.metadata:
            new_table = new_table.cast(
                new_table.schema.with_metadata(table.schema.metadata)
            )
        pq.write_table(new_table, output_parquet)

        episode_out = {
            "episode_index": new_ep_idx,
            "tasks": episode.get("tasks", [task]),
            "length": n_frames,
        }
        for key, value in episode.items():
            if key not in {"episode_index", "tasks", "length"}:
                episode_out[key] = value
        output_episodes.append(episode_out)

        stats_record = old_episode_stats.get(old_ep_idx)
        if stats_record is not None:
            output_episode_stats.append(
                {
                    "episode_index": new_ep_idx,
                    "stats": _reindex_episode_stats(
                        stats_record["stats"],
                        new_ep_idx=new_ep_idx,
                        new_task_idx=new_task_idx,
                        new_frame_start=total_frames,
                        old_frame_start=old_frame_start,
                    ),
                }
            )

        total_frames += n_frames

    total_episodes = len(output_episodes)
    total_chunks = (total_episodes + output_chunks_size - 1) // output_chunks_size
    info_out = dict(info)
    info_out.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(selected_tasks),
            "total_videos": 0,
            "total_chunks": max(1, total_chunks),
            "chunks_size": output_chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet"
            ),
        }
    )
    info_out.pop("video_path", None)

    _write_json(output_meta_dir / "info.json", info_out)
    _write_jsonl(output_meta_dir / "episodes.jsonl", output_episodes)
    _write_jsonl(
        output_meta_dir / "tasks.jsonl",
        [{"task_index": idx, "task": task} for task, idx in task_to_new_idx.items()],
    )
    if output_episode_stats:
        _write_jsonl(output_meta_dir / "episodes_stats.jsonl", output_episode_stats)

    print(
        f"[sample] Wrote {total_episodes} episodes, {total_frames} frames "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
