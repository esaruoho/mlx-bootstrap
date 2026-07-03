#!/usr/bin/env python3
"""mlx-bootstrap — a foolproof local LLM on Apple Silicon via MLX, in ONE file. Zero tokens, $0.

Give it a prompt; get an answer from a model running entirely on your Mac's GPU. No account, no
API key, no cloud, no per-token cost. First run installs everything it needs into its own isolated
venv and downloads a small model; later runs are instant.

    python3 mlx-bootstrap.py "Explain a transformer in one sentence."   # one-shot answer
    python3 mlx-bootstrap.py --serve                                    # OpenAI-compatible server on :8080
    python3 mlx-bootstrap.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit "…"  # smaller/faster
    python3 mlx-bootstrap.py --bootstrap                                # print this whole story
    python3 mlx-bootstrap.py --doctor                                   # check the machine, install nothing

This is a *convey bootstrap*: one whitelabeled, self-contained, self-explaining file that carries a
whole idea — runnable AND documented from the file alone — so it can be handed to another bot or
human and adopted immediately.

WHAT IT DOES, step by step (each idempotent, each checked):
  1. Verify Apple Silicon (arm64 macOS). MLX is Apple-GPU only; on anything else it says so and stops.
  2. Ensure `mlx-lm` is importable. If not, create an isolated venv at ~/.cache/mlx-bootstrap/venv,
     pip-install mlx-lm there, and re-exec itself under that venv's Python. Your system Python stays
     untouched (no --break-system-packages, no global pollution).
  3. Load an instruct model — default `Qwen3-4B-Instruct-2507` 8-bit (~4GB; the same model the
     author's fleet runs via mlx_lm.server), downloading it from Hugging Face on first use and
     caching it forever after. Pass `--model` for a smaller/faster one on a low-RAM Mac.
  4. Apply the model's chat template, run generation ON THE GPU, print the answer. Zero tokens billed.
  5. --serve instead starts `mlx_lm.server` — an OpenAI-compatible endpoint (POST /v1/chat/completions)
     so any tool that speaks the OpenAI API can talk to your local model.

Share and improve freely. MIT-spirit: no warranty, be nice.
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-8bit"   # the model the author's fleet runs (~4GB, 8-bit)
VENV = Path.home() / ".cache" / "mlx-bootstrap" / "venv"
BOOTSTRAP = __doc__.strip()


def die(msg, code=1):
    print(f"[mlx-bootstrap] {msg}", file=sys.stderr)
    sys.exit(code)


def preflight():
    """Apple Silicon macOS + a usable Python. Returns a list of (label, ok, detail)."""
    checks = []
    is_mac = sys.platform == "darwin"
    is_arm = platform.machine() == "arm64"
    checks.append(("macOS", is_mac, sys.platform))
    checks.append(("Apple Silicon (arm64)", is_arm, platform.machine()))
    checks.append(("Python >= 3.9", sys.version_info >= (3, 9), platform.python_version()))
    return checks, (is_mac and is_arm and sys.version_info >= (3, 9))


def ensure_mlx():
    """Import mlx_lm, or build an isolated venv, install it, and re-exec under that venv.
    Guarded against re-exec loops by MLX_BOOTSTRAP_REEXEC."""
    try:
        import mlx_lm  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("MLX_BOOTSTRAP_REEXEC") == "1":
        die("mlx-lm still not importable after venv install — see errors above.")
    vpy = VENV / "bin" / "python3"
    if not vpy.exists():
        print("[mlx-bootstrap] first run: creating isolated venv + installing mlx-lm (~1-2 min)…",
              file=sys.stderr)
        VENV.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        subprocess.run([str(vpy), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
        subprocess.run([str(vpy), "-m", "pip", "install", "-q", "mlx-lm"], check=True)
    env = dict(os.environ, MLX_BOOTSTRAP_REEXEC="1")
    os.execve(str(vpy), [str(vpy), os.path.abspath(__file__), *sys.argv[1:]], env)


def run_generate(model_id, prompt, max_tokens):
    from mlx_lm import load, generate
    print(f"[mlx-bootstrap] loading {model_id} (first time downloads from Hugging Face)…", file=sys.stderr)
    model, tok = load(model_id)
    if getattr(tok, "chat_template", None):
        text_prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    else:
        text_prompt = prompt
    return generate(model, tok, prompt=text_prompt, max_tokens=max_tokens, verbose=False)


def run_serve(model_id, port):
    print(f"[mlx-bootstrap] starting OpenAI-compatible server: http://127.0.0.1:{port}/v1  "
          f"(model {model_id})", file=sys.stderr)
    print(f"[mlx-bootstrap] test it:  curl http://127.0.0.1:{port}/v1/chat/completions "
          f"-d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]}}'", file=sys.stderr)
    os.execvp(sys.executable, [sys.executable, "-m", "mlx_lm.server",
                               "--model", model_id, "--port", str(port)])


def main(argv):
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    if argv and argv[0] in ("--bootstrap", "--about", "bootstrap"):
        print(BOOTSTRAP); return 0

    checks, ok = preflight()
    if argv and argv[0] == "--doctor":
        for label, good, detail in checks:
            print(f"  [{'✓' if good else '✗'}] {label}: {detail}")
        venv_ok = (VENV / "bin" / "python3").exists()
        print(f"  [{'✓' if venv_ok else '·'}] mlx-lm venv: {'ready at ' + str(VENV) if venv_ok else 'not built yet'}")
        return 0 if ok else 1
    if not ok:
        for label, good, detail in checks:
            if not good:
                print(f"  ✗ {label}: {detail}", file=sys.stderr)
        die("this machine can't run MLX (needs Apple Silicon macOS + Python 3.9+).")

    # flags
    model_id = DEFAULT_MODEL
    max_tokens = 256
    serve = False
    port = 8080
    prompt_parts = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--model":
            i += 1; model_id = argv[i]
        elif a == "--max-tokens":
            i += 1; max_tokens = int(argv[i])
        elif a == "--serve":
            serve = True
        elif a == "--port":
            i += 1; port = int(argv[i])
        else:
            prompt_parts.append(a)
        i += 1

    ensure_mlx()   # may re-exec under the venv; everything below runs with mlx_lm available

    if serve:
        run_serve(model_id, port)
        return 0
    prompt = " ".join(prompt_parts).strip() or "Say hello in one short sentence."
    answer = run_generate(model_id, prompt, max_tokens)
    print(answer.strip())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        die(f"error: {e}")
