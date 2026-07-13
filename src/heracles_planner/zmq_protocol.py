from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import zmq

HEADER_SIZE = 1280


def pack_message(topic: str, version: int, fields: list[tuple[str, str, np.ndarray]]) -> bytes:
    arrays = [(name, dtype, np.ascontiguousarray(value)) for name, dtype, value in fields]
    count = int(arrays[0][2].shape[0]) if arrays and arrays[0][2].ndim > 1 else 1
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": [
            {"name": name, "dtype": dtype, "shape": list(value.shape)}
            for name, dtype, value in arrays
        ],
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    if len(encoded) > HEADER_SIZE:
        raise ValueError("ZMQ packed-message header exceeds 1280 bytes")
    payload = b"".join(value.tobytes(order="C") for _, _, value in arrays)
    return topic.encode() + encoded.ljust(HEADER_SIZE, b"\0") + payload


def pack_pose_v1(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_quat_xyzw: np.ndarray,
    frame_index: np.ndarray,
    topic: str = "pose",
    catch_up: bool = False,
) -> bytes:
    root_wxyz = np.asarray(root_quat_xyzw, dtype=np.float32)[..., [3, 0, 1, 2]]
    return pack_message(
        topic,
        1,
        [
            ("joint_pos", "f32", np.asarray(joint_pos, dtype="<f4")),
            ("joint_vel", "f32", np.asarray(joint_vel, dtype="<f4")),
            ("body_quat_w", "f32", np.asarray(root_wxyz, dtype="<f4")),
            ("frame_index", "i64", np.asarray(frame_index, dtype="<i8")),
            ("catch_up", "u8", np.asarray([catch_up], dtype="u1")),
        ],
    )


DTYPES = {"f32": "<f4", "f64": "<f8", "i64": "<i8", "i32": "<i4", "u8": "u1"}


def unpack_message(message: bytes, topic: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    topic_bytes = topic.encode()
    if not message.startswith(topic_bytes):
        raise ValueError(f"message does not start with topic {topic}")
    header_start = len(topic_bytes)
    raw_header = message[header_start : header_start + HEADER_SIZE].rstrip(b"\0")
    header = json.loads(raw_header)
    offset = header_start + HEADER_SIZE
    fields: dict[str, np.ndarray] = {}
    for field in header["fields"]:
        dtype = np.dtype(DTYPES[field["dtype"]])
        shape = tuple(field["shape"])
        size = int(np.prod(shape)) * dtype.itemsize
        fields[field["name"]] = np.frombuffer(message[offset : offset + size], dtype=dtype).reshape(
            shape
        )
        offset += size
    if offset != len(message):
        raise ValueError("packed message has trailing or missing bytes")
    return header, fields


class PosePublisher:
    def __init__(self, endpoint: str, topic: str = "pose"):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(endpoint)
        self.topic = topic
        time.sleep(0.5)

    def publish(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        root_quat_xyzw: np.ndarray,
        frame_index: np.ndarray,
    ) -> None:
        self.socket.send(
            pack_pose_v1(joint_pos, joint_vel, root_quat_xyzw, frame_index, self.topic)
        )

    def close(self) -> None:
        self.socket.close(linger=0)


class SimStateSubscriber:
    def __init__(self, endpoint: str, topic: str = "sim_state"):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self.socket.connect(endpoint)
        self.topic = topic

    def receive(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        if not self.socket.poll(timeout_ms):
            raise TimeoutError(f"no simulator state on {self.topic} within {timeout_ms} ms")
        _, fields = unpack_message(self.socket.recv(), self.topic)
        return fields

    def close(self) -> None:
        self.socket.close(linger=0)


def append_planner_log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
