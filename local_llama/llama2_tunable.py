#!/usr/bin/env python3
"""
==============================================================================
Llama 2 Tunable Predictor  -  adjust the sampling parameters at run time
==============================================================================

WHAT IS THIS?
-------------
A thin "control panel" on top of llama2_next_word_predictor.py (keep both files
in the SAME folder). The model weights are never changed. What you adjust is
HOW the next token is chosen from the model's 32,000 probabilities.

THE PARAMETERS
--------------
  temperature     0 = greedy (always the single most likely token, repeatable)
  (0 - 5)         ~0.7 = balanced, 1.0 = model's raw probabilities, >1 = wilder.
                  Math: probabilities = softmax(logits / temperature).

  top_k           Sample only among the k most likely tokens (0 = off).
  (0 - 32000)     Cuts off the long tail of unlikely tokens.

  top_p           "Nucleus" sampling: keep the smallest set of tokens whose
  (0.01 - 1.0)    probabilities add up to p (1.0 = off). Adapts to how confident
                  the model is: few tokens when sure, many when unsure.

  max_new_tokens  How many tokens to generate after your prompt (1 = pure
  (1 - 255)       next-word prediction, like the original script).

  show_n          How many top candidates to print for the FIRST next token.
  (1 - 20)

  seed            Random seed. Same prompt + same settings + same seed gives the
  (0 - 2^31-1)    same text, so experiments are repeatable.

ORDER OF OPERATIONS for every token:
  logits -> / temperature -> softmax -> top_k filter -> top_p filter
         -> renormalise -> random draw   (temperature = 0 skips all and uses argmax)

HOW TO USE
----------
  Command line (set once):
    python llama2_tunable.py --prompt "Once upon a" --temperature 0.8 --top-p 0.9 --max-new-tokens 20

  Interactive (change on the fly), type these at the prompt:
    /set temperature 0.8        /temp 0.8      (short form)
    /set top_k 40               /topk 40
    /set top_p 0.9              /topp 0.9
    /set max_new_tokens 20      /tokens 20
    /show     print current settings        /defaults   reset to defaults
    /help     list commands                 exit        quit

REFERENCES
----------
  Temperature (softmax with T) : https://arxiv.org/abs/1503.02531
  Top-k sampling               : https://arxiv.org/abs/1805.04833
  Top-p / nucleus sampling     : https://arxiv.org/abs/1904.09751
==============================================================================
"""

import argparse
import time
from dataclasses import dataclass, fields

import numpy as np

# Re-use the tested core (tokenizer, loader, transformer forward pass).
from llama2_next_word_predictor import (
    BOS_TOKEN_ID,
    build_vocab_map,
    bpe_encode,
    decode_token,
    llama2_forward,
    load_checkpoint,
    load_tokenizer,
    softmax,
)

EOS_TOKEN_ID = 2  # "end of text" token; generation stops when it is chosen


# ==============================================================================
# 1. SETTINGS: one place that holds every adjustable parameter
# ==============================================================================
@dataclass
class Settings:
    temperature: float = 0.0     # 0 = greedy (original script behaviour)
    top_k: int = 0               # 0 = off
    top_p: float = 1.0           # 1.0 = off
    max_new_tokens: int = 1      # 1 = single next-word prediction
    show_n: int = 5              # how many candidates to display
    seed: int = 42               # for repeatable sampling


# name -> (type, minimum, maximum, one-line help)
SPECS = {
    "temperature": (float, 0.0, 5.0, "0 = greedy, ~0.7 balanced, >1 more random"),
    "top_k": (int, 0, 32000, "sample only from the k likeliest tokens (0 = off)"),
    "top_p": (float, 0.01, 1.0, "keep tokens until probability sums to p (1.0 = off)"),
    "max_new_tokens": (int, 1, 255, "tokens to generate (1 = next word only)"),
    "show_n": (int, 1, 20, "how many top candidates to print"),
    "seed": (int, 0, 2**31 - 1, "random seed for repeatable results"),
}
ALIASES = {
    "temp": "temperature", "t": "temperature",
    "topk": "top_k", "top-k": "top_k",
    "topp": "top_p", "top-p": "top_p",
    "tokens": "max_new_tokens", "max": "max_new_tokens", "n": "max_new_tokens",
    "shown": "show_n",
}


def set_param(settings, name, raw_value):
    """Validate and apply one setting. Raises ValueError with a friendly message."""
    name = ALIASES.get(name.lower(), name.lower())
    if name not in SPECS:
        raise ValueError(f"Unknown setting '{name}'. Choose from: {', '.join(SPECS)}")
    value_type, low, high, _ = SPECS[name]
    try:
        value = value_type(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"'{raw_value}' is not a valid {value_type.__name__} for {name}")
    if not (low <= value <= high):
        raise ValueError(f"{name} must be between {low} and {high} (got {value})")
    setattr(settings, name, value)
    return name, value


def format_settings(settings):
    return "  ".join(f"{f.name}={getattr(settings, f.name)}" for f in fields(settings))


# ==============================================================================
# 2. SAMPLING: turn 32,000 logits into ONE chosen token
# ==============================================================================
def sample_token(logits, settings, rng):
    """
    Picks the next token ID according to temperature / top_k / top_p.
    temperature = 0 is greedy: always the highest logit (no randomness).
    """
    if settings.temperature <= 0.0:
        return int(np.argmax(logits))

    # Temperature: divide logits, then softmax. T<1 sharpens, T>1 flattens.
    # Convert to float64 and renormalise: float32 probabilities only sum to 1
    # within ~1e-7, which NumPy's random sampler rejects as "not summing to 1".
    probs = softmax(logits / settings.temperature).astype(np.float64)
    probs /= probs.sum()

    # Top-k: zero out everything below the k-th largest probability.
    if 0 < settings.top_k < probs.size:
        kth_value = np.partition(probs, -settings.top_k)[-settings.top_k]
        probs = np.where(probs >= kth_value, probs, 0.0)
        probs /= probs.sum()

    # Top-p (nucleus): keep the smallest group of tokens whose total >= top_p.
    if settings.top_p < 1.0:
        order = np.argsort(-probs)                        # most likely first
        cumulative = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cumulative, settings.top_p)) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep] = probs[keep]
        probs = mask / mask.sum()

    return int(rng.choice(probs.size, p=probs))


# ==============================================================================
# 3. DECODING HELPERS
# ==============================================================================
def decode_tokens(token_ids, vocab):
    """
    Joins several token IDs into text. Raw-byte tokens such as <0xC3> are
    collected as bytes first, so multi-byte characters (e.g. 'é') decode right.
    """
    buffer = bytearray()
    for tid in token_ids:
        piece = vocab[tid]
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            buffer.append(int(piece[3:5], 16))
        else:
            buffer += piece.encode("utf-8")
    return buffer.decode("utf-8", errors="replace")


# ==============================================================================
# 4. GENERATION: prompt -> first-token candidates -> optional continuation
# ==============================================================================
def generate(text, config, weights, vocab, vocab_map, scores, settings):
    head_dim = config["dim"] // config["n_heads"]
    seq_len = config["seq_len"]
    rng = np.random.default_rng(settings.seed)   # fresh RNG => repeatable per prompt

    # The context window holds `seq_len` positions in total, so the prompt gets
    # whatever is left after reserving room for the tokens we will generate.
    max_new = min(settings.max_new_tokens, seq_len - 1)
    max_prompt = seq_len - max_new + 1
    prompt_tokens = bpe_encode(text, vocab, vocab_map, scores)
    truncated = len(prompt_tokens) > max_prompt
    if truncated:                                 # keep BOS + the most recent tokens
        prompt_tokens = [BOS_TOKEN_ID] + prompt_tokens[-(max_prompt - 1):]

    k_cache = np.zeros(
        (config["n_layers"], seq_len, config["n_kv_heads"], head_dim), dtype=np.float32
    )
    v_cache = np.zeros_like(k_cache)

    # --- Prefill: run the prompt through the model, filling the KV cache ------
    start = time.perf_counter()
    logits = None
    for pos, tok in enumerate(prompt_tokens):
        logits = llama2_forward(tok, pos, config, weights, k_cache, v_cache)
    prefill_ms = (time.perf_counter() - start) * 1000.0

    # --- Candidates for the FIRST new token (probabilities at the chosen T) ---
    shown_temperature = settings.temperature if settings.temperature > 0 else 1.0
    shown_probs = softmax(logits / shown_temperature)
    top_ids = np.argsort(shown_probs)[::-1][: settings.show_n]

    # --- Generation loop (the KV cache means one cheap forward pass per token) -
    generated, stop_reason = [], "reached max_new_tokens"
    pos = len(prompt_tokens)                      # position of the next token to feed
    gen_start = time.perf_counter()
    for step in range(max_new):
        next_id = sample_token(logits, settings, rng)
        if next_id in (BOS_TOKEN_ID, EOS_TOKEN_ID):
            stop_reason = "model produced end-of-text"
            break
        generated.append(next_id)
        if step == max_new - 1:
            break                                 # last token needs no forward pass
        logits = llama2_forward(next_id, pos, config, weights, k_cache, v_cache)
        pos += 1
    gen_ms = (time.perf_counter() - gen_start) * 1000.0

    return {
        "n_prompt_tokens": len(prompt_tokens),
        "truncated": truncated,
        "shown_temperature": shown_temperature,
        "top_ids": [int(t) for t in top_ids],
        "top_probs": [float(shown_probs[t]) for t in top_ids],
        "generated_ids": generated,
        "stop_reason": stop_reason,
        "prefill_ms": prefill_ms,
        "gen_ms": gen_ms,
    }


def print_result(text, result, settings, vocab):
    generated_text = decode_tokens(result["generated_ids"], vocab)
    first_id = result["generated_ids"][0] if result["generated_ids"] else None

    if result["truncated"]:
        print("  [Warning] Prompt too long for the context window; kept the most recent tokens.")
    print(f"\n  Settings           : {format_settings(settings)}")
    print(f"  Prompt tokens      : {result['n_prompt_tokens']} (incl. BOS)")
    print(f"  Generated text     : {text}{generated_text}")
    print(f"  New tokens         : {len(result['generated_ids'])} ({result['stop_reason']})")
    print(f"  Prompt time        : {result['prefill_ms']:.1f} ms")
    if len(result["generated_ids"]) > 1 and result["gen_ms"] > 0:
        rate = (len(result["generated_ids"]) - 1) / (result["gen_ms"] / 1000.0)
        print(f"  Generation speed   : {rate:.1f} tokens/s")

    mode = "greedy, T=1 shown" if settings.temperature <= 0 else f"T={result['shown_temperature']}"
    print(f"\n  Top {len(result['top_ids'])} candidates for the first new token ({mode}):")
    for rank, (tid, p) in enumerate(zip(result["top_ids"], result["top_probs"]), start=1):
        marker = "  <- chosen" if tid == first_id else ""
        print(f"    {rank}. {decode_token(tid, vocab)!r:<15} -> {p * 100:6.2f}%{marker}")
    print("-" * 60 + "\n")


# ==============================================================================
# 5. INTERACTIVE CONTROL PANEL
# ==============================================================================
def handle_command(line, settings):
    """Handles lines that start with '/'. Returns nothing; prints feedback."""
    parts = line[1:].split()
    if not parts:
        return
    cmd = parts[0].lower()
    try:
        if cmd == "show":
            print(f"  {format_settings(settings)}")
        elif cmd == "defaults":
            for f in fields(Settings):
                setattr(settings, f.name, f.default)
            print(f"  Reset: {format_settings(settings)}")
        elif cmd == "help":
            print("  Commands: /set <name> <value>, /show, /defaults, /help, exit")
            for name, (_, low, high, text) in SPECS.items():
                print(f"    {name:<15} [{low} .. {high}]  {text}")
        elif cmd == "set" and len(parts) == 3:
            name, value = set_param(settings, parts[1], parts[2])
            print(f"  {name} = {value}")
        elif ALIASES.get(cmd, cmd) in SPECS and len(parts) == 2:
            name, value = set_param(settings, cmd, parts[1])
            print(f"  {name} = {value}")
        else:
            print("  Unknown command. Type /help")
    except ValueError as err:
        print(f"  [Error] {err}")


def run(model_path, tokenizer_path, settings, one_shot_prompt=None):
    print(f"Loading {model_path} and {tokenizer_path}...")
    config, weights = load_checkpoint(model_path)
    vocab, scores = load_tokenizer(tokenizer_path, config["vocab_size"])
    vocab_map = build_vocab_map(vocab)

    if one_shot_prompt is not None:
        result = generate(one_shot_prompt, config, weights, vocab, vocab_map, scores, settings)
        print_result(one_shot_prompt, result, settings, vocab)
        return

    print("\n========================================================")
    print("  Llama 2 Tunable Predictor Ready!")
    print("  Type text + ENTER.  /help for parameter commands.")
    print(f"  {format_settings(settings)}")
    print("========================================================\n")

    while True:
        try:
            line = input("Enter text > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            print("Exiting...")
            break
        if line.startswith("/"):
            handle_command(line, settings)
            continue
        result = generate(line, config, weights, vocab, vocab_map, scores, settings)
        print_result(line, result, settings, vocab)


if __name__ == "__main__":
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Llama 2 predictor with adjustable parameters")
    parser.add_argument("--model", default="stories15M.bin")
    parser.add_argument("--tokenizer", default="tokenizer.bin")
    parser.add_argument("--prompt", default=None, help="run once on this prompt and exit")
    for name, (value_type, low, high, text) in SPECS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=value_type, default=getattr(defaults, name),
            help=f"{text} [{low}..{high}] (default {getattr(defaults, name)})",
        )
    args = parser.parse_args()

    settings = Settings()
    try:
        for name in SPECS:
            set_param(settings, name, getattr(args, name))
    except ValueError as err:
        parser.error(str(err))

    run(args.model, args.tokenizer, settings, args.prompt)
