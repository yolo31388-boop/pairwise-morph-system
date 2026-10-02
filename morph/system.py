"""
pairwise-morph-system - MorphSystem

Morph target system featuring:

- clamped, smoothed, per-channel morph weights
- accumulated position/normal/tangent deformation with LOD
- conflict-aware expression combining with priorities and presets
- time-synced morph animation with keyframe events and interpolation
- compressed, delta-encoded, checksum-validated morph data
"""

import base64
import hashlib
import json
import zlib


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


class MorphSystem:
    MORPH_MAGIC = "MRPH"
    MORPH_VERSION = 1

    # Expressions that must never be active at the same time.
    DEFAULT_CONFLICT_GROUPS = (
        frozenset({"smile", "cry"}),
        frozenset({"happy", "sad"}),
        frozenset({"angry", "calm"}),
        frozenset({"blink_left", "blink_right"}),
    )

    # LOD 0: pos+normal+tangent, 1: pos+normal, 2: pos only, 3: also cull tiny weights.
    LOD_MIN_WEIGHT = (0.0, 0.0, 0.0, 0.05)

    def __init__(self):
        self.state = {}
        # channel -> morph name -> {"weight", "target", "smooth_time"}
        self.channels = {}
        self.priorities = {}
        self.conflict_groups = [set(group) for group in self.DEFAULT_CONFLICT_GROUPS]
        self.presets = {}
        self.morph_data = {}
        self._event_callbacks = {}
        self._fired_events = set()
        self._last_sync_time = None

    # ------------------------------------------------------------------
    # Weights: range clamping + smoothing + per-channel isolation
    # ------------------------------------------------------------------
    def set_morph_weight(self, name="default", weight=0.0, channel="default",
                         smooth_time=0.0, dt=None):
        weight = _clamp01(weight)
        channel_state = self.channels.setdefault(channel, {})
        entry = channel_state.setdefault(
            name, {"weight": 0.0, "target": 0.0, "smooth_time": 0.0})
        entry["target"] = weight
        entry["smooth_time"] = max(0.0, float(smooth_time))
        if entry["smooth_time"] <= 0.0:
            entry["weight"] = weight
        elif dt is not None:
            self._advance_smoothing(entry, dt)
        return entry["weight"]

    def get_morph_weight(self, name="default", channel="default"):
        return self.channels.get(channel, {}).get(name, {}).get("weight", 0.0)

    def update(self, dt, channel=None):
        """Advance weight smoothing for one channel or all channels."""
        names = [channel] if channel is not None else list(self.channels)
        for name in names:
            for entry in self.channels.get(name, {}).values():
                self._advance_smoothing(entry, dt)

    @staticmethod
    def _advance_smoothing(entry, dt):
        smooth_time = entry["smooth_time"]
        if smooth_time <= 0.0:
            entry["weight"] = entry["target"]
            return
        step = max(0.0, float(dt)) / smooth_time
        delta = entry["target"] - entry["weight"]
        if abs(delta) <= step:
            entry["weight"] = entry["target"]
        else:
            entry["weight"] += step if delta > 0 else -step
        entry["weight"] = _clamp01(entry["weight"])

    # ------------------------------------------------------------------
    # Deformation: normals + tangents + accumulation + LOD
    # ------------------------------------------------------------------
    def apply_morph(self, vertices=None, morphs=None, weights=None,
                    normals=None, tangents=None, lod=0):
        vertices = vertices or []
        morphs = morphs or {}
        # No weights given: apply every morph at full strength. Otherwise
        # morphs missing from the dict default to 0 (not deformed).
        default_weight = 1.0 if not weights else 0.0
        weights = weights or {}
        lod = max(0, min(3, int(lod)))
        min_weight = self.LOD_MIN_WEIGHT[lod]

        pos = [list(v) for v in vertices]
        nrm = [list(n) for n in normals] if normals is not None else None
        tan = [list(t) for t in tangents] if tangents is not None else None

        for name, morph in morphs.items():
            weight = _clamp01(weights.get(name, default_weight))
            if weight <= min_weight:
                continue  # LOD culling: skip tiny deformations on distant LODs
            self._accumulate(pos, morph.get("position"), weight)
            if lod < 2 and morph.get("normal"):
                if nrm is None:
                    nrm = [[0.0, 0.0, 0.0] for _ in pos]
                self._accumulate(nrm, morph["normal"], weight)
            if lod < 1 and morph.get("tangent"):
                if tan is None:
                    tan = [[0.0, 0.0, 0.0] for _ in pos]
                self._accumulate(tan, morph["tangent"], weight)

        result = {"vertices": pos}
        if nrm is not None:
            result["normals"] = nrm
        if tan is not None:
            result["tangents"] = tan
        return result

    @staticmethod
    def _accumulate(base, delta, weight):
        """Add weight * delta into base; delta may be a dense list or sparse dict."""
        if not delta:
            return
        items = delta.items() if isinstance(delta, dict) else enumerate(delta)
        for index, d in items:
            index = int(index)
            if d is None or index >= len(base):
                continue
            for axis in range(min(3, len(d))):
                base[index][axis] += d[axis] * weight

    # ------------------------------------------------------------------
    # Expressions: conflict detection + priority + presets
    # ------------------------------------------------------------------
    def combine_expressions(self, expressions=None, priorities=None,
                            preset=None, save_preset=None):
        if preset is not None:
            if preset not in self.presets:
                raise KeyError("unknown expression preset: %s" % preset)
            return dict(self.presets[preset])

        resolved = self._normalize_expressions(expressions)
        prio = dict(self.priorities)
        if priorities:
            prio.update(priorities)

        for group in self.conflict_groups:
            active = [name for name in group if resolved.get(name, 0.0) > 0.0]
            if len(active) < 2:
                continue
            # Higher priority wins; ties broken by requested weight.
            winner = max(active, key=lambda n: (prio.get(n, 0), resolved[n]))
            for name in active:
                if name != winner:
                    resolved[name] = 0.0

        if save_preset is not None:
            self.presets[save_preset] = dict(resolved)
        return resolved

    @staticmethod
    def _normalize_expressions(expressions):
        if expressions is None:
            return {}
        if isinstance(expressions, dict):
            return {name: _clamp01(w) for name, w in expressions.items()}
        return {name: 1.0 for name in expressions}

    # ------------------------------------------------------------------
    # Animation: time sync + keyframe events + interpolation
    # ------------------------------------------------------------------
    def on_event(self, event_name, callback):
        self._event_callbacks.setdefault(event_name, []).append(callback)

    def sync_morph_animation(self, animation=None, time=None, skeletal_time=None):
        animation = animation or {}
        # The skeletal clock is the master when provided, keeping both in sync.
        master_time = float(
            skeletal_time if skeletal_time is not None else (time or 0.0))
        previous = self._last_sync_time
        if previous is not None and master_time < previous:
            self._fired_events.clear()  # seeked backwards: allow re-triggering
        self._last_sync_time = master_time

        keyframes = animation.get("keyframes", animation)
        weights = {}
        for name, keys in keyframes.items():
            if name == "events":
                continue
            weights[name] = self._interpolate(keys, master_time)

        for event_time, event_name in animation.get("events", []):
            if self._event_due(previous, master_time, float(event_time), event_name):
                self._fire_event(event_name)

        return {"time": master_time, "weights": weights}

    def _event_due(self, previous, now, event_time, event_name):
        key = (event_time, event_name)
        if key in self._fired_events:
            return False
        due = (previous is None and event_time <= now) or \
              (previous is not None and previous < event_time <= now)
        if due:
            self._fired_events.add(key)
        return due

    def _fire_event(self, event_name):
        for callback in self._event_callbacks.get(event_name, []):
            callback(event_name)

    @staticmethod
    def _interpolate(keys, t):
        """Linear interpolation between keyframes; clamps outside the range."""
        keys = sorted((float(k[0]), _clamp01(k[1])) for k in keys)
        if not keys:
            return 0.0
        if t <= keys[0][0]:
            return keys[0][1]
        if t >= keys[-1][0]:
            return keys[-1][1]
        for (t0, w0), (t1, w1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return w0 + (w1 - w0) * alpha
        return keys[-1][1]

    # ------------------------------------------------------------------
    # Data: compression + delta encoding + validation
    # ------------------------------------------------------------------
    @staticmethod
    def build_delta(base, deformed, epsilon=1e-6):
        """Sparse delta: only vertices that actually changed are stored."""
        delta = {}
        for index, (b, d) in enumerate(zip(base, deformed)):
            diff = [d[axis] - b[axis] for axis in range(min(3, len(d)))]
            if any(abs(v) > epsilon for v in diff):
                delta[index] = diff
        return delta

    def save_morph_data(self, morphs, path=None, compress=True):
        payload = json.dumps(
            {"version": self.MORPH_VERSION, "morphs": morphs},
            separators=(",", ":")).encode("utf-8")
        envelope = {
            "magic": self.MORPH_MAGIC,
            "version": self.MORPH_VERSION,
            "compressed": bool(compress),
            "checksum": hashlib.sha256(payload).hexdigest(),
        }
        if compress:
            envelope["encoding"] = "zlib"
            envelope["payload"] = base64.b64encode(
                zlib.compress(payload)).decode("ascii")
        else:
            envelope["encoding"] = "none"
            envelope["payload"] = payload.decode("utf-8")
        raw = json.dumps(envelope).encode("utf-8")
        if path is not None:
            with open(path, "wb") as fh:
                fh.write(raw)
        return raw

    def load_morph_data(self, source=None, validate=True):
        if source is None:
            return self.morph_data
        envelope = self._read_envelope(source)
        if validate:
            self._validate_envelope(envelope)
        payload = envelope["payload"]
        if envelope.get("encoding") == "zlib":
            try:
                payload = zlib.decompress(base64.b64decode(payload))
            except Exception as exc:
                raise ValueError("corrupt morph data: %s" % exc) from exc
        try:
            data = json.loads(payload)
        except Exception as exc:
            raise ValueError("corrupt morph data: %s" % exc) from exc
        morphs = data.get("morphs", {})
        self.morph_data.update(morphs)
        return morphs

    @staticmethod
    def _read_envelope(source):
        if isinstance(source, dict):
            return source
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        else:
            with open(source, "rb") as fh:
                raw = fh.read()
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError("corrupt morph data: %s" % exc) from exc

    def _validate_envelope(self, envelope):
        if envelope.get("magic") != self.MORPH_MAGIC:
            raise ValueError("corrupt morph data: bad magic header")
        if envelope.get("version") != self.MORPH_VERSION:
            raise ValueError("unsupported morph data version: %r"
                             % envelope.get("version"))
        payload = envelope.get("payload")
        if envelope.get("encoding") == "zlib":
            try:
                payload = zlib.decompress(base64.b64decode(payload))
            except Exception as exc:
                raise ValueError("corrupt morph data: %s" % exc) from exc
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != envelope.get("checksum"):
            raise ValueError("corrupt morph data: checksum mismatch")
