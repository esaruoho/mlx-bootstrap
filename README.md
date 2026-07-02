# mlx-bootstrap

**A foolproof local LLM on Apple Silicon via [MLX](https://github.com/ml-explore/mlx), in ONE file. Zero tokens, $0.**

Give it a prompt; get an answer from a model running entirely on your Mac's GPU. No account, no API
key, no cloud, no per-token cost. First run installs everything it needs into its own isolated venv
and downloads a small model; later runs are instant.

```bash
python3 mlx-bootstrap.py "Explain a transformer in one sentence."   # one-shot answer
python3 mlx-bootstrap.py --serve                                    # OpenAI-compatible server on :8080
python3 mlx-bootstrap.py --model mlx-community/Qwen2.5-3B-Instruct-4bit "…"
python3 mlx-bootstrap.py --doctor                                   # check the machine, install nothing
python3 mlx-bootstrap.py --bootstrap                               # print the whole story
```

That's the whole thing — **one file, no dependencies to install yourself.** Save it, run it, share it.

## Why

Getting a local model running on a Mac usually means a scavenger hunt: which Python, `pip install`
what, PEP-668 "externally managed environment" errors, which model repo, what's the current
`mlx_lm` API, how to start the server. `mlx-bootstrap` collapses all of that into one command that
**checks each step and does the right thing** — so a friend (or another bot) can go from nothing to a
working local LLM by running a single file.

## What it does, step by step (each idempotent, each checked)

1. **Verify Apple Silicon** (arm64 macOS). MLX is Apple-GPU only; on anything else it says so and stops.
2. **Ensure `mlx-lm` is importable.** If not, it creates an isolated venv at `~/.cache/mlx-bootstrap/venv`,
   `pip install`s `mlx-lm` there, and re-execs itself under that venv's Python. **Your system Python
   stays untouched** — no `--break-system-packages`, no global pollution.
3. **Load a small instruct model** (default a ~1 GB 4-bit Qwen), downloading it from Hugging Face on
   first use and caching it forever after.
4. **Apply the model's chat template, run generation on the GPU, print the answer.** Zero tokens billed.
5. `--serve` instead starts `mlx_lm.server` — an **OpenAI-compatible** endpoint (`POST
   /v1/chat/completions`) so any tool that speaks the OpenAI API can talk to your local model.

## Requirements

- A Mac with **Apple Silicon** (M1/M2/M3/M4…).
- **Python 3.9+** (the one that ships with macOS, or Homebrew's — either works).
- Internet on first run (to install `mlx-lm` and download the model). Offline after that.

Run `python3 mlx-bootstrap.py --doctor` to check your machine without installing anything.

## The pattern: a "bootstrap"

This is a single, self-contained, self-explaining file that carries a whole idea in a form another
bot or human can run **and** understand from the file alone — idea transfer, not just code transfer.
`python3 mlx-bootstrap.py --bootstrap` prints the whole story; the rest of the file *is* the story,
executable.

## Models

Default: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (small, capable, ~1 GB). Override with `--model`
using any [`mlx-community`](https://huggingface.co/mlx-community) repo id, e.g.
`--model mlx-community/Qwen2.5-3B-Instruct-4bit` or a Llama/Mistral 4-bit MLX conversion.

## License

MIT — share and improve freely. No warranty, be nice.
