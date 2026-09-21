"""
Halve genie-tts's RAM without slowing synthesis.

Measured on 2026-09-21 (genie-tts 2.0.2, onnxruntime 1.22.1, CPU EP): after
loading kiana plus the shared models and synthesising a few sentences, this
process held 8.6 GB private / 6.2 GB working set for ~2.0 GB of fp32 weights.
Two things account for most of it:

  * genie's load_session_with_fp16_conversion (ModelManager.py) widens each
    fp16 .bin to fp32, copies it into the ONNX proto and builds the session
    from the serialised bytes. The session keeps ~3x the weights resident;
    the same weights loaded from a plain .onnx file keep ~1x.
  * Every InferenceSession has its own CPU arena, which grows to that
    session's inference peak and never shrinks. There are eight sessions
    (five per character, HuBERT, RoBERTa, speaker encoder), and TTSEngine
    serialises genie.tts() behind its lock, so one arena shared between them
    is enough. (The arena is thread-safe; a character switch overlapping a
    synthesis only means both peaks are held at once.)

So this module:

  1. converts each fp16 .bin to fp32 once, into ModelCache/fp32/, with a
     rewritten .onnx pointing at it, and loads that file directly;
  2. registers one process-wide CPU arena and has every session use it.

Result on the same texts: 4.2 GB private / 3.3 GB working set, load
10.5 s -> 7.9 s, synthesis as fast as before (rtf ~0.9 either way). With
feibi loaded as well: 12.0 -> 5.0 GB private. HuBERT's output is
bit-identical to genie's loader. Weight prepacking stays on: turning it off
saved memory too but made synthesis 5-15% slower (measured on the same model
in another genie-tts project), and at rtf ~0.9 that would leave Koodo's
read-aloud waiting on the server. Arena off instead of shared: less memory
still, but also slower.

The cache costs ~1.0 GB of disk for kiana plus HuBERT, ~0.6 GB more per extra
character (both T2S decoders share one .bin). An entry is keyed on the source
files' path, size and mtime, so replacing a model rebuilds it; older entries
for the same file are deleted. If the cache cannot be written, genie's own
loader is used.

Call apply() once, before any character or reference audio is loaded - the
same place as mixed_g2p.apply().
"""
import functools
import glob
import hashlib
import logging
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ModelCache", "fp32")

_applied = False


def _source_key(*paths: str) -> str:
    """Short hash of the files' identity: a replaced model gets a new key."""
    h = hashlib.sha1()
    for path in paths:
        st = os.stat(path)
        h.update(f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def _readable_prefix(path: str) -> str:
    """'kiana-tts_models-vits_fp32' - several characters ship identically named files."""
    parent = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return "-".join([os.path.basename(os.path.dirname(parent)), os.path.basename(parent), stem])


def _replace_atomically(write, final_path: str) -> None:
    """Write to a sibling file first, so a process killed mid-write leaves no
    half-written cache entry behind to be loaded next time."""
    root, ext = os.path.splitext(final_path)
    partial = f"{root}.partial{ext}"
    write(partial)
    os.replace(partial, final_path)


def _drop_stale(prefix: str, ext: str, keep: str) -> None:
    for old in glob.glob(os.path.join(CACHE_DIR, f"{glob.escape(prefix)}.*{ext}")):
        if os.path.normcase(old) != os.path.normcase(keep):
            try:
                os.remove(old)
            except OSError:
                pass


def _cached_fp32_model(onnx_path: str, fp16_bin_path: str) -> str:
    """Path of a plain fp32 .onnx for this model, converting it on first use."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    bin_prefix = _readable_prefix(fp16_bin_path)
    fp32_bin = os.path.join(CACHE_DIR, f"{bin_prefix}.{_source_key(fp16_bin_path)}.bin")
    if not os.path.exists(fp32_bin) or os.path.getsize(fp32_bin) != 2 * os.path.getsize(fp16_bin_path):
        started = time.monotonic()
        _replace_atomically(
            lambda p: np.fromfile(fp16_bin_path, dtype=np.float16).astype(np.float32).tofile(p),
            fp32_bin)
        _drop_stale(bin_prefix, ".bin", fp32_bin)
        logger.info("[ort_memory_patch] Cached fp32 weights %s (%.1fs, one-time).",
                    os.path.basename(fp32_bin), time.monotonic() - started)

    onnx_prefix = _readable_prefix(onnx_path)
    fp32_onnx = os.path.join(CACHE_DIR, f"{onnx_prefix}.{_source_key(onnx_path, fp32_bin)}.onnx")
    if not os.path.exists(fp32_onnx):
        proto = onnx.load(onnx_path, load_external_data=False)
        for tensor in proto.graph.initializer:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            # genie indexes these offsets into the widened fp32 buffer, so they
            # already describe the fp32 file; only the location changes.
            entries = {e.key: e.value for e in tensor.external_data}
            del tensor.external_data[:]
            for key, value in (("location", os.path.basename(fp32_bin)),
                               ("offset", entries.get("offset", "0")),
                               ("length", entries["length"])):
                entry = tensor.external_data.add()
                entry.key, entry.value = key, value
        _replace_atomically(lambda p: onnx.save(proto, p), fp32_onnx)
        _drop_stale(onnx_prefix, ".onnx", fp32_onnx)
    return fp32_onnx


def _patch_fp16_loader() -> None:
    import genie_tts.ModelManager as model_manager_module

    original = model_manager_module.load_session_with_fp16_conversion

    @functools.wraps(original)
    def load_from_fp32_cache(onnx_path, fp16_bin_path, providers, sess_options=None):
        try:
            cached = _cached_fp32_model(onnx_path, fp16_bin_path)
        except Exception as e:
            logger.warning("[ort_memory_patch] fp32 cache unavailable for %s (%s); using genie's loader.",
                           os.path.basename(onnx_path), e)
            return original(onnx_path, fp16_bin_path, providers, sess_options)
        return InferenceSession(cached, sess_options=sess_options, providers=providers)

    # ModelManager.load_character and load_cn_hubert look the name up at call time.
    model_manager_module.load_session_with_fp16_conversion = load_from_fp32_cache


def _share_one_cpu_arena() -> None:
    ort.create_and_register_allocator(
        ort.OrtMemoryInfo("Cpu", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, ort.OrtMemType.DEFAULT),
        ort.OrtArenaCfg(0, -1, -1, -1),  # onnxruntime's defaults
    )
    original_init = InferenceSession.__init__

    @functools.wraps(original_init)
    def init_with_env_allocator(self, path_or_bytes, sess_options=None, providers=None,
                                provider_options=None, **kwargs):
        # Covers every session in this process; genie-tts is the only user.
        sess_options = sess_options or ort.SessionOptions()
        sess_options.add_session_config_entry("session.use_env_allocators", "1")
        original_init(self, path_or_bytes, sess_options, providers, provider_options, **kwargs)

    InferenceSession.__init__ = init_with_env_allocator


def apply() -> None:
    global _applied
    if _applied:
        return
    _share_one_cpu_arena()
    _patch_fp16_loader()
    _applied = True
    logger.info("[ort_memory_patch] fp32 model cache and shared CPU arena enabled.")
