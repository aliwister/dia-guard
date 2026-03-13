"""
Dialect discovery, transformation utilities, and NLP resource setup.

Uses multiprocessing for timeout handling — Process.terminate() can kill
hung C extensions, unlike threading which is blocked by the GIL.
"""

import inspect
import logging
import multiprocessing
import re
import time
import warnings

logger = logging.getLogger(__name__)

# Suppress torch/stanza warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Monkey-patch torch.load for stanza compatibility (torch 2.6+ defaults weights_only=True
# but stanza models contain numpy globals that aren't in the allowlist)
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


SKIP_CLASSES = frozenset({
    "BaseDialect", "DialectFromVector", "DialectFromFeatureList",
    "MultiDialect", "AAVE_Example_From_List",
})

# Dialects that crash (bad allocation) or hang during transform
# Previously broken dialects now work with offline patches + multiprocessing
BROKEN_DIALECTS = frozenset()


def discover_dialects():
    """Discover all 50 dialect classes from the multivalue package."""
    from multivalue import Dialects
    from multivalue.BaseDialect import BaseDialect

    dialect_classes = {}
    for name, obj in inspect.getmembers(Dialects, inspect.isclass):
        if (issubclass(obj, BaseDialect)
                and name not in SKIP_CLASSES
                and hasattr(obj, "transform")):
            dialect_classes[name] = obj

    logger.info(f"Discovered {len(dialect_classes)} dialect classes")
    return dialect_classes


def _apply_offline_patches():
    """Monkey-patch spaCy/NLTK/stanza to skip network downloads.

    All resources are already cached locally. This prevents hangs when
    github.com or raw.githubusercontent.com are unreachable.
    """
    import spacy
    import spacy.cli
    spacy.cli.download = lambda *a, **kw: None

    import nltk
    nltk.download = lambda *a, **kw: True

    from stanza.pipeline.core import DownloadMethod
    import stanza
    _orig_init = stanza.Pipeline.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("download_method", DownloadMethod.REUSE_RESOURCES)
        return _orig_init(self, *args, **kwargs)
    stanza.Pipeline.__init__ = _patched_init
    logger.info("Applied offline patches (spaCy/NLTK/stanza)")


def setup_nlp_resources():
    """Verify spaCy models and NLTK data are available, apply offline patches."""
    import spacy
    try:
        spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm already available")
    except OSError:
        logger.info("Downloading spaCy en_core_web_sm...")
        try:
            from spacy.cli import download
            download("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy download failed (may already be cached): {e}")

    import nltk
    try:
        nltk.download("cmudict", quiet=True)
        nltk.download("wordnet", quiet=True)
    except Exception:
        pass  # Already cached locally
    logger.info("NLTK resources ready")

    # After verifying resources exist, apply offline patches to prevent
    # network downloads during dialect instantiation
    _apply_offline_patches()


# Heuristic to detect code lines
CODE_INDICATORS = re.compile(
    r"^\s*(def |class |import |from |#include|#define|#pragma|"
    r"public |private |protected |package |using |namespace |"
    r"var |let |const |function |if\s*\(|for\s*\(|while\s*\(|"
    r"return |try\s*\{|catch\s*\(|switch\s*\(|"
    r"[{}();]$|//|/\*|\*/|>>>|\.\.\.)",
    re.MULTILINE,
)


def is_code_line(line):
    """Heuristic to detect if a line is likely code rather than natural language."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(CODE_INDICATORS.match(stripped))


def extract_docstrings(text):
    """Extract NL text from Python docstrings and comments."""
    # Match triple-quoted strings (both ''' and """)
    triple_sq = r"'''(.*?)'''"
    triple_dq = '"""(.*?)"""'
    matches = []
    for pat in [triple_sq, triple_dq]:
        for m in re.finditer(pat, text, re.DOTALL):
            matches.append((m.start(1), m.end(1), m.group(1)))
    return sorted(matches, key=lambda x: x[0])


def _safe_transform(text, dialect_instance):
    """Try to transform text; on failure return original."""
    try:
        return dialect_instance.transform(text)
    except Exception:
        return text


def smart_transform(text, dialect_instance, contains_code=False):
    """Transform NL portions of text, preserving code blocks verbatim.

    For code-heavy prompts (SecurityEval, CyberSecEval), extracts NL text
    from docstrings and comments, transforms only those parts, and
    reconstructs the original with transformed NL.
    """
    if not text or not text.strip():
        return text

    if not contains_code:
        return _safe_transform(text, dialect_instance)

    # Strategy: Extract docstrings, transform their NL content, reconstruct
    docstrings = extract_docstrings(text)
    if docstrings:
        result = text
        # Process in reverse order to preserve offsets
        for start, end, content in reversed(docstrings):
            stripped = content.strip()
            if stripped and len(stripped) > 5:
                transformed = _safe_transform(stripped, dialect_instance)
                leading = content[:len(content) - len(content.lstrip())]
                trailing = content[len(content.rstrip()):]
                result = result[:start] + leading + transformed + trailing + result[end:]
        return result

    # Fallback: line-by-line code vs NL detection
    lines = text.split("\n")
    result_lines = []
    nl_buffer = []

    for line in lines:
        if is_code_line(line):
            if nl_buffer:
                nl_text = "\n".join(nl_buffer)
                result_lines.append(_safe_transform(nl_text, dialect_instance))
                nl_buffer = []
            result_lines.append(line)
        else:
            nl_buffer.append(line)

    if nl_buffer:
        nl_text = "\n".join(nl_buffer)
        result_lines.append(_safe_transform(nl_text, dialect_instance))

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Multiprocessing-based dialect worker
# ---------------------------------------------------------------------------
# On Windows, multiprocessing uses 'spawn' so the worker function must be
# a top-level, picklable function.

def _dialect_worker_loop(dialect_cls_name, seed, in_queue, out_queue):
    """Subprocess entry point: instantiate dialect, process transform requests.

    Runs in a separate process so it can be forcefully killed via
    Process.terminate() if it hangs on a C extension call.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Re-apply torch monkey-patch in subprocess
    import torch
    _orig = torch.load
    def _patched(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig(*args, **kwargs)
    torch.load = _patched

    # Prevent all network downloads in subprocess (resources cached locally)
    _apply_offline_patches()

    # Instantiate the dialect
    try:
        from multivalue import Dialects
        dialect_cls = getattr(Dialects, dialect_cls_name)
        dialect_instance = dialect_cls(seed=seed)
        out_queue.put(("ready", None))
    except Exception as e:
        out_queue.put(("init_error", str(e)))
        return

    # Message loop: receive (text, contains_code), send back result
    while True:
        try:
            msg = in_queue.get(timeout=300)  # 5 min idle timeout
        except Exception:
            break

        if msg is None:  # Shutdown signal
            break

        text, contains_code = msg
        try:
            result = smart_transform(text, dialect_instance, contains_code)
            out_queue.put(("success", result))
        except Exception as e:
            out_queue.put(("error", str(e)))


class DialectWorker:
    """Manages a subprocess running a single dialect instance.

    Each transform request is sent to the subprocess via a Queue.
    If the subprocess hangs (C extension blocking), Process.terminate()
    forcefully kills it — something impossible with threads.
    """

    def __init__(self, dialect_cls_name, seed=42, init_timeout=120):
        self.dialect_cls_name = dialect_cls_name
        self.seed = seed
        self.alive = False
        self._process = None
        self._in_queue = None
        self._out_queue = None

        # Use 'spawn' context explicitly (Windows default, but be explicit)
        ctx = multiprocessing.get_context("spawn")
        self._in_queue = ctx.Queue()
        self._out_queue = ctx.Queue()

        self._process = ctx.Process(
            target=_dialect_worker_loop,
            args=(dialect_cls_name, seed, self._in_queue, self._out_queue),
            daemon=True,
        )
        self._process.start()

        # Wait for initialization
        try:
            status, value = self._out_queue.get(timeout=init_timeout)
            if status == "ready":
                self.alive = True
                logger.info(f"  DialectWorker({dialect_cls_name}): subprocess ready (pid={self._process.pid})")
            else:
                logger.error(f"  DialectWorker({dialect_cls_name}): init failed: {value}")
                self._cleanup()
        except Exception:
            logger.error(f"  DialectWorker({dialect_cls_name}): init timed out after {init_timeout}s")
            self._cleanup()

    def transform(self, text, contains_code=False, timeout_sec=30):
        """Send text to subprocess for transformation.

        Returns (transformed_text, success_bool).
        If the subprocess hangs, it is killed and self.alive is set to False.
        """
        if not self.alive:
            return text, False

        try:
            self._in_queue.put((text, contains_code))
        except Exception:
            self.alive = False
            return text, False

        try:
            status, value = self._out_queue.get(timeout=timeout_sec)
            if status == "success":
                return value, True
            else:
                logger.warning(f"  Worker({self.dialect_cls_name}) error: {value}")
                return text, False
        except Exception:
            # Timeout — subprocess is hung, kill it
            logger.warning(
                f"  Worker({self.dialect_cls_name}) timed out after {timeout_sec}s — killing subprocess"
            )
            self._kill()
            return text, False

    def _kill(self):
        """Forcefully terminate the subprocess."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=5)
        self.alive = False

    def _cleanup(self):
        """Clean up after failed init."""
        self._kill()

    def close(self):
        """Gracefully shut down the worker."""
        if self.alive:
            try:
                self._in_queue.put(None)  # Shutdown signal
                self._process.join(timeout=5)
            except Exception:
                pass
        self._kill()
        # Close queues
        for q in [self._in_queue, self._out_queue]:
            if q:
                try:
                    q.close()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
