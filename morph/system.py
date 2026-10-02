"""
pairwise-morph-system - MorphSystem

Morph target system with clamped/smoothed per-channel weights, accumulated
vertex deformation (positions, normals, tangents) with LOD, expression
combination with conflict detection / priorities / presets, time-synced
morph animation with keyframe events and interpolation, and compressed,
delta-encoded, validated morph data.
"""

import json
import math
import struct
import zlib

MAGIC = b"MORPH\x01"
_EPS = 1e-6


class MorphDataError(ValueError):
    """Raised when morph data is corrupt or fails validation."""


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _normalize(vec):
    length = math.sqrt(sum(c * c for c in vec))
    if length < _EPS:
        return list(vec)
    return [c / length for c in vec]


def _to_vec3(vec):
    vec = list(vec)
    if len(vec) >= 3:
        return [float(vec[0]), float(vec[1]), float(vec[2])]
    return [float(c) for c in vec] + [0.0] * (3 - len(vec))


class Mesh:
    """Simple vertex container supporting attribute and item access."""

    def __init__(self, positions=None, normals=None, tangents=None):
        self.positions = [list(p) for p in (positions or [])]
        self.normals = [list(n) for n in (normals or [])]
        self.tangents = [list(t) for t in (tangents or [])]

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __len__(self):
        return len(self.positions)


class MorphTarget:
    """A morph target stored as per-vertex deltas relative to the base mesh."""

    def __init__(self, name, position_deltas=None, normal_deltas=None,
                 tangent_deltas=None, positions=None, normals=None, tangents=None):
        self.name = name
        self.position_deltas = [list(d) for d in (position_deltas or [])]
        self.normal_deltas = [list(d) for d in (normal_deltas or [])]
        self.tangent_deltas = [list(d) for d in (tangent_deltas or [])]
        if positions is not None:
            self.positions = [list(p) for p in positions]
        if normals is not None:
            self.normals = [list(n) for n in normals]
        if tangents is not None:
            self.tangents = [list(t) for t in tangents]

    def to_dict(self):
        return {
            "name": self.name,
            "position_deltas": self.position_deltas,
            "normal_deltas": self.normal_deltas,
            "tangent_deltas": self.tangent_deltas,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or "name" not in data:
            raise MorphDataError("invalid morph target entry")
        return cls(
            data["name"],
            position_deltas=data.get("position_deltas") or data.get("deltas") or [],
            normal_deltas=data.get("normal_deltas") or [],
            tangent_deltas=data.get("tangent_deltas") or [],
        )


class MorphSystem:
    def __init__(self, smoothing_speed=10.0, lod_distances=(50.0, 150.0)):
        self.state = {}
        self.smoothing_speed = float(smoothing_speed)
        self.lod_distances = tuple(lod_distances)
        # Per-channel morph weights: target (clamped) and smoothed values.
        self.weights = {}
        self.target_weights = {}
        self.mesh = None
        self.targets = {}
        self.conflicts = {}
        self.priorities = {}
        self.presets = {}
        self.animations = {}
        self.events = []
        self.time = 0.0

    # ------------------------------------------------------------------
    # Morph weights: range clamping, smoothing, independent channels
    # ------------------------------------------------------------------
    def set_morph_weight(self, name=None, weight=None, channel=None, **kwargs):
        """Set a morph weight, clamped to [0, 1], tracked per channel.

        Accepts either (name, weight) or a dict of {name: weight}.
        """
        if isinstance(name, dict):
            result = {}
            for key, value in name.items():
                result[key] = self.set_morph_weight(key, value, **kwargs)
            return result
        if name is None:
            return dict(self.weights)
        if weight is None:
            weight = kwargs.get("value", 0.0)
        weight = _clamp(float(weight))
        channel = channel if channel is not None else "default"
        self.weights.setdefault(channel, {})[name] = weight
        self.target_weights.setdefault(channel, {})[name] = weight
        self.state.setdefault("weights", {})[name] = weight
        return weight

    def get_morph_weight(self, name, channel=None):
        channel = channel if channel is not None else "default"
        return self.weights.get(channel, {}).get(name, 0.0)

    def update(self, dt):
        """Advance smoothing: weights ease toward their clamped targets."""
        dt = max(0.0, float(dt))
        self.time += dt
        alpha = 1.0 - math.exp(-self.smoothing_speed * dt)
        for channel, targets in self.target_weights.items():
            current = self.weights.setdefault(channel, {})
            for name, target in targets.items():
                smoothed = current.get(name, 0.0) + (target - current.get(name, 0.0)) * alpha
                current[name] = smoothed
                self.state.setdefault("weights", {})[name] = smoothed
        return self.weights

    # ------------------------------------------------------------------
    # Vertex deformation: normals + tangents, accumulation, LOD
    # ------------------------------------------------------------------
    def register_mesh(self, mesh):
        self.mesh = self._normalize_mesh(mesh)
        return self.mesh

    def register_morph_target(self, target, **kwargs):
        if isinstance(target, str):
            target = MorphTarget(target, **kwargs)
        elif isinstance(target, dict):
            target = MorphTarget.from_dict(target)
        self.targets[target.name] = target
        return target

    @staticmethod
    def _normalize_mesh(mesh):
        if isinstance(mesh, Mesh):
            return mesh
        if isinstance(mesh, dict):
            return Mesh(mesh.get("positions"), mesh.get("normals"), mesh.get("tangents"))
        raise TypeError("mesh must be a Mesh or dict")

    def _lod_for_distance(self, distance):
        if distance is None:
            return 0
        if distance >= self.lod_distances[1]:
            return 2
        if distance >= self.lod_distances[0]:
            return 1
        return 0

    def apply_morph(self, mesh=None, weights=None, distance=None, lod=None, **kwargs):
        """Apply accumulated morph deltas to positions, normals and tangents.

        Multiple morph targets accumulate. LOD 0 deforms everything, LOD 1
        positions only, LOD 2 (far away) skips deformation entirely.
        """
        if mesh is None:
            mesh = self.mesh
        if mesh is None:
            return {"positions": [], "normals": [], "tangents": [], "lod": 0}
        mesh = self._normalize_mesh(mesh)
        if weights is None:
            weights = self.state.get("weights") or {}
        if isinstance(weights, (list, tuple)):
            weights = {name: w for name, w in zip(self.targets, weights)}
        if lod is None:
            lod = self._lod_for_distance(distance)

        positions = [_to_vec3(p) for p in mesh.positions]
        normals = [_to_vec3(n) for n in mesh.normals]
        tangents = [_to_vec3(t) for t in mesh.tangents]
        count = len(positions)
        if not normals:
            normals = [[0.0, 0.0, 1.0] for _ in range(count)]
        if not tangents:
            tangents = [[1.0, 0.0, 0.0] for _ in range(count)]

        if lod < 2:
            for name, weight in weights.items():
                weight = _clamp(float(weight))
                if abs(weight) < _EPS:
                    continue
                target = self.targets.get(name)
                if target is None:
                    continue
                for i in range(count):
                    if i < len(target.position_deltas):
                        delta = target.position_deltas[i]
                        for axis in range(min(3, len(delta))):
                            positions[i][axis] += weight * delta[axis]
                    if lod == 0:
                        if i < len(target.normal_deltas):
                            delta = target.normal_deltas[i]
                            for axis in range(min(3, len(delta))):
                                normals[i][axis] += weight * delta[axis]
                        if i < len(target.tangent_deltas):
                            delta = target.tangent_deltas[i]
                            for axis in range(min(3, len(delta))):
                                tangents[i][axis] += weight * delta[axis]
            normals = [_normalize(n) for n in normals]
            tangents = [_normalize(t) for t in tangents]

        return {
            "positions": positions,
            "normals": normals,
            "tangents": tangents,
            "lod": lod,
        }

    # ------------------------------------------------------------------
    # Expression combination: conflicts, priorities, presets
    # ------------------------------------------------------------------
    def add_conflict(self, expr_a, expr_b):
        self.conflicts.setdefault(expr_a, set()).add(expr_b)
        self.conflicts.setdefault(expr_b, set()).add(expr_a)

    def set_priority(self, name, priority):
        self.priorities[name] = priority

    def combine_expressions(self, expressions=None, **kwargs):
        """Combine expressions resolving conflicts by priority.

        Higher priority wins; ties keep the first expression. Base
        expressions (default priority 0) never override higher ones.
        """
        if expressions is None:
            return dict(self.presets) if not kwargs else {}
        if isinstance(expressions, dict):
            items = list(expressions.items())
        else:
            items = [(name, 1.0) for name in expressions]

        resolved = {}
        order = []
        for name, weight in items:
            weight = _clamp(float(weight))
            if name in resolved:
                resolved[name] = max(resolved[name], weight)
                continue
            dropped = False
            for other in list(resolved):
                if other in self.conflicts.get(name, ()):
                    prio_new = self.priorities.get(name, 0)
                    prio_old = self.priorities.get(other, 0)
                    if prio_new > prio_old:
                        del resolved[other]
                        order.remove(other)
                    else:
                        dropped = True
                        break
            if not dropped:
                resolved[name] = weight
                order.append(name)
        return {name: resolved[name] for name in order}

    def save_preset(self, name, expressions=None):
        if expressions is None:
            expressions = dict(self.state.get("weights") or {})
        elif isinstance(expressions, (list, tuple)):
            expressions = {expr: 1.0 for expr in expressions}
        self.presets[name] = dict(expressions)
        return self.presets[name]

    def load_preset(self, name):
        preset = self.presets.get(name)
        if preset is None:
            raise KeyError(name)
        for expr, weight in preset.items():
            self.set_morph_weight(expr, weight)
        return dict(preset)

    # ------------------------------------------------------------------
    # Animation sync: shared clock, keyframe events, interpolation
    # ------------------------------------------------------------------
    def add_animation(self, name, keyframes, duration=None, events=None):
        """Register an animation. keyframes: [(time, {target: weight}), ...]."""
        keyframes = sorted(
            ((float(t), dict(w)) for t, w in keyframes), key=lambda kf: kf[0]
        )
        if duration is None:
            duration = keyframes[-1][0] if keyframes else 0.0
        self.animations[name] = {
            "keyframes": keyframes,
            "duration": float(duration),
            "events": sorted(float(e) for e in (events or [])),
            "fired": set(),
        }
        return self.animations[name]

    def on_keyframe(self, callback):
        self.events.append(callback)
        return callback

    @staticmethod
    def _interpolate(keyframes, time):
        if not keyframes:
            return {}
        if time <= keyframes[0][0]:
            return dict(keyframes[0][1])
        if time >= keyframes[-1][0]:
            return dict(keyframes[-1][1])
        for (t0, w0), (t1, w1) in zip(keyframes, keyframes[1:]):
            if t0 <= time <= t1:
                alpha = 0.0 if t1 - t0 < _EPS else (time - t0) / (t1 - t0)
                names = set(w0) | set(w1)
                return {
                    name: w0.get(name, 0.0) + (w1.get(name, 0.0) - w0.get(name, 0.0)) * alpha
                    for name in names
                }
        return dict(keyframes[-1][1])

    def sync_morph_animation(self, animation=None, time=None, **kwargs):
        """Evaluate morph animation on the shared clock.

        Weights interpolate linearly between keyframes, keyframe crossings
        fire registered callbacks, and the resulting weights are applied to
        the morph channels so morphs stay in sync with skeletal animation.
        """
        if animation is None:
            return {"time": self.time, "weights": dict(self.state.get("weights") or {})}
        if isinstance(animation, str):
            animation = self.animations[animation]
        if isinstance(animation, dict) and "keyframes" in animation:
            keyframes = sorted(
                ((float(t), dict(w)) for t, w in animation["keyframes"]),
                key=lambda kf: kf[0],
            )
            duration = float(animation.get("duration") or (keyframes[-1][0] if keyframes else 0.0))
            event_times = sorted(float(e) for e in animation.get("events", []))
        else:
            keyframes = sorted(
                ((float(t), dict(w)) for t, w in animation), key=lambda kf: kf[0]
            )
            duration = keyframes[-1][0] if keyframes else 0.0
            event_times = [t for t, _ in keyframes]
        if time is None:
            time = self.time
        time = _clamp(float(time), 0.0, duration)

        weights = self._interpolate(keyframes, time)
        fired = [t for t in event_times if t <= time]
        for fired_time in fired:
            for callback in self.events:
                callback(fired_time, weights)
        for name, weight in weights.items():
            self.set_morph_weight(name, weight)
        return {"time": time, "weights": weights, "events": fired}

    # ------------------------------------------------------------------
    # Morph data: compression, delta encoding, validation
    # ------------------------------------------------------------------
    def save_morph_data(self, path, compress=True):
        """Save morph data as compressed, delta-encoded, checksummed bytes."""
        payload = {
            "targets": [target.to_dict() for target in self.targets.values()],
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        body = zlib.compress(raw) if compress else raw
        checksum = zlib.crc32(body) & 0xFFFFFFFF
        blob = MAGIC + struct.pack("<I", checksum) + body
        with open(path, "wb") as handle:
            handle.write(blob)
        return len(blob)

    def load_morph_data(self, path=None, **kwargs):
        """Load morph data, validating magic, checksum and structure.

        Raises MorphDataError on corrupt data instead of crashing. Targets
        are stored as deltas (incremental), not full vertex copies.
        """
        if path is None:
            return {}
        if isinstance(path, (bytes, bytearray)):
            blob = bytes(path)
        else:
            with open(path, "rb") as handle:
                blob = handle.read()

        if blob.startswith(MAGIC):
            if len(blob) < len(MAGIC) + 4:
                raise MorphDataError("truncated morph data: missing checksum")
            (checksum,) = struct.unpack("<I", blob[len(MAGIC):len(MAGIC) + 4])
            body = blob[len(MAGIC) + 4:]
            if (zlib.crc32(body) & 0xFFFFFFFF) != checksum:
                raise MorphDataError("morph data checksum mismatch")
            try:
                raw = zlib.decompress(body)
            except zlib.error:
                raw = body
        else:
            raw = blob

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MorphDataError("corrupt morph data: %s" % exc) from exc
        if isinstance(payload, list):
            payload = {"targets": payload}
        if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list):
            raise MorphDataError("invalid morph data structure")

        self.targets = {}
        for entry in payload["targets"]:
            target = MorphTarget.from_dict(entry)
            self.targets[target.name] = target
        return {name: target.to_dict() for name, target in self.targets.items()}
