#!/usr/bin/env python3
"""
==============================================================================
Llama 2 Single Next-Word Predictor  (pure NumPy, CPU only, no PyTorch)
==============================================================================

WHAT IS THIS?
-------------
A from-scratch re-implementation of the Llama 2 *inference* (forward pass) in
~250 lines of NumPy. It loads a tiny pre-trained Llama 2 model in Andrej
Karpathy's "llama2.c" binary format (stories15M.bin, ~15 million parameters,
trained on the TinyStories dataset) and answers ONE question:

    "Given the text typed so far, which token is most likely to come next?"

NOTE ON "WORD": the model predicts a TOKEN (a sub-word piece such as ' time',
'ing' or ','), not always a whole word. 32,000 tokens exist in the vocabulary.

HOW DOES IT WORK?  (pipeline, read top to bottom)
-------------------------------------------------
  text --> [1] BPE tokenizer --> token IDs  (e.g. [1, 9038, 2501, 263])
       --> [2] embedding lookup: each ID -> a 288-number vector
       --> [3] 6 x transformer layer, each containing:
               RMSNorm -> self-attention (Q,K,V + RoPE + KV-cache) -> residual add
               RMSNorm -> SwiGLU feed-forward network               -> residual add
       --> [4] final RMSNorm -> matrix multiply -> 32,000 raw scores (logits)
       --> [5] softmax -> 32,000 probabilities (sum = 1.0)
       --> [6] pick the highest = predicted next token; show the Top-5

MODEL SIZE (stories15M):  dim=288, hidden_dim=768, layers=6, heads=6,
                          kv_heads=6, vocab=32000, max context=256 tokens.

WHAT RESULTS SHOULD I EXPECT?
-----------------------------
For every prompt the program prints:
  * Completed Sentence : your text + the single most likely next token
  * Predicted Next Word: that token and its integer ID
  * Inference Time     : milliseconds for the whole prompt (all tokens)
  * Top-5 candidates   : the 5 highest probabilities, in percent

How to read them:
  * Probabilities are the model's *confidence*, not truth. A sharp peak
    (e.g. one token > 50%) means the context makes the next token obvious;
    a flat list (all < 10%) means many continuations are plausible.
  * The model was trained ONLY on simple children's stories, so expect
    story-like continuations ("Once upon a" -> ' time' is the classic case).
    It knows nothing about finance, news or general facts: do NOT use it for
    trading signals. It is a learning tool for understanding transformers.
  * Speed: 15M parameters is small; a short prompt typically takes tens to a
    few hundred milliseconds on one modern CPU core (depends on your CPU).

HOW TO RUN
----------
  1) Put these files in the same folder as this script:
       stories15M.bin   (model weights, ~60 MB)
       tokenizer.bin    (vocabulary, ~430 KB)
     Source: https://github.com/karpathy/llama2.c  (tokenizer.bin is in the repo;
     stories15M.bin is linked in its README, hosted on Hugging Face:
     https://huggingface.co/karpathy/tinyllamas)
  2) pip install numpy
  3) python llama2_next_word_predictor.py                    (interactive)
     python llama2_next_word_predictor.py --prompt "Once upon a"   (one shot)

REFERENCES
----------
  Llama 2           : https://arxiv.org/abs/2307.09288
  RMSNorm           : https://arxiv.org/abs/1910.07467
  RoPE              : https://arxiv.org/abs/2104.09864
  SwiGLU (GLU var.) : https://arxiv.org/abs/2002.05202
  TinyStories       : https://arxiv.org/abs/2305.07759
  Original C code   : https://github.com/karpathy/llama2.c
==============================================================================
"""

import argparse
import os
import struct
import time

import numpy as np

# ------------------------------------------------------------------------------
# Constants fixed by the llama2.c file format / Llama 2 architecture
# ------------------------------------------------------------------------------
BOS_TOKEN_ID = 1            # "Beginning Of Sequence" token; every prompt starts with it
BYTE_FALLBACK_OFFSET = 3    # IDs 0,1,2 are <unk>,<s>,</s>; raw byte 0x00 is ID 3, 0x01 is ID 4 ...
RMS_EPS = 1e-5              # tiny number added inside RMSNorm to avoid dividing by zero
ROPE_BASE = 10000.0         # base of the rotary-position frequency ladder (RoPE paper)


# ==============================================================================
# 1. MATHEMATICAL BUILDING BLOCKS
# ==============================================================================
def rmsnorm(x, weight):
    """
    Root-Mean-Square normalisation (RMSNorm).

    What : rescales a vector so its average magnitude is ~1, then multiplies by
           a learned per-dimension weight.
    Why  : keeps numbers from exploding/vanishing as they pass through 6 layers.
           Cheaper than LayerNorm because it does not subtract the mean.
    Math : y = x / sqrt(mean(x^2) + eps) * weight
    """
    ss = np.mean(x ** 2) + RMS_EPS          # mean of squares (+ epsilon)
    return (x / np.sqrt(ss)) * weight       # normalise, then apply learned scale


def softmax(x):
    """
    Turns raw scores (logits) into probabilities that are >= 0 and sum to 1.

    Subtracting max(x) first does not change the result mathematically but
    stops exp() from overflowing on large numbers (numerical stability).
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


# ==============================================================================
# 2. BINARY MODEL & TOKENIZER LOADERS
# ==============================================================================
def load_checkpoint(filepath):
    """
    Reads a llama2.c "version 0" checkpoint (.bin) into NumPy arrays.

    File layout (all little-endian):
      [ 28-byte header : 7 x int32 ]
      [ float32 weights, in this exact order ]
          token_embedding (vocab, dim)
          rms_att   (layers, dim)          | per-layer norm before attention
          wq        (layers, dim,    dim)  | query projection
          wk        (layers, kv_dim, dim)  | key projection
          wv        (layers, kv_dim, dim)  | value projection
          wo        (layers, dim,    dim)  | attention output projection
          rms_ffn   (layers, dim)          | per-layer norm before feed-forward
          w1        (layers, hidden, dim)  | FFN gate projection
          w2        (layers, dim, hidden)  | FFN down projection
          w3        (layers, hidden, dim)  | FFN up projection
          rms_final (dim)                  | norm after the last layer
          [legacy RoPE cos/sin tables]     | skipped, we recompute RoPE ourselves
          wcls      (vocab, dim)           | ONLY present if weights are not shared

    Matrices are stored as (output_size, input_size), so `W @ x` works directly.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, "rb") as f:
        raw_header = f.read(28)
        if len(raw_header) != 28:
            raise ValueError(f"{filepath} is too small to be a llama2.c checkpoint.")
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = (
            struct.unpack("7i", raw_header)
        )

        # llama2.c convention: a POSITIVE vocab_size means the output classifier
        # re-uses the input embedding matrix ("weight tying"); NEGATIVE means a
        # separate classifier matrix is stored at the end of the file.
        shared_weights = vocab_size > 0
        vocab_size = abs(vocab_size)

        config = {
            "dim": dim,                # embedding size per token            (288)
            "hidden_dim": hidden_dim,  # feed-forward inner width            (768)
            "n_layers": n_layers,      # number of transformer blocks          (6)
            "n_heads": n_heads,        # query attention heads                 (6)
            "n_kv_heads": n_kv_heads,  # key/value heads (< n_heads => GQA)    (6)
            "vocab_size": vocab_size,  # number of distinct tokens         (32000)
            "seq_len": seq_len,        # maximum context length in tokens    (256)
        }

        # Read every remaining float32 in one fast operation.
        data = np.fromfile(f, dtype=np.float32)

    offset = 0

    def take(shape):
        """Slice the next tensor of `shape` out of the flat weight array."""
        nonlocal offset
        size = int(np.prod(shape))
        if offset + size > data.size:
            raise ValueError(
                "Checkpoint is truncated or corrupt: expected more weights than "
                f"the file contains (needed {offset + size}, found {data.size})."
            )
        arr = data[offset: offset + size].reshape(shape)
        offset += size
        return arr

    head_dim = dim // n_heads
    kv_dim = (dim * n_kv_heads) // n_heads   # total width of the K and V vectors

    weights = {
        "token_embedding": take((vocab_size, dim)),
        "rms_att": take((n_layers, dim)),
        "wq": take((n_layers, dim, dim)),
        "wk": take((n_layers, kv_dim, dim)),
        "wv": take((n_layers, kv_dim, dim)),
        "wo": take((n_layers, dim, dim)),
        "rms_ffn": take((n_layers, dim)),
        "w1": take((n_layers, hidden_dim, dim)),
        "w2": take((n_layers, dim, hidden_dim)),
        "w3": take((n_layers, hidden_dim, dim)),
        "rms_final": take((dim,)),
    }

    # Older llama2.c files store two RoPE tables (cos and sin), each of size
    # seq_len * head_dim/2. We compute RoPE on the fly, so just skip both.
    offset += seq_len * head_dim

    weights["wcls"] = (
        weights["token_embedding"] if shared_weights else take((vocab_size, dim))
    )
    return config, weights


def load_tokenizer(filepath, vocab_size):
    """
    Reads vocabulary strings and BPE merge scores from llama2.c's tokenizer.bin.

    File layout: [int32 max_token_length] then, for each token:
                 [float32 score][int32 byte_length][utf-8 bytes].

    IMPORTANT: llama2.c already converted SentencePiece's '▁' (U+2581, "word
    starts here") into an ordinary space ' ' when it exported this file. So
    ' time' in the vocabulary means "the word 'time' preceded by a space".
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

    vocab, scores = [], []
    with open(filepath, "rb") as f:
        _ = struct.unpack("i", f.read(4))            # max_token_length (unused)
        for _ in range(vocab_size):
            score, length = struct.unpack("fi", f.read(8))
            vocab.append(f.read(length).decode("utf-8", errors="replace"))
            scores.append(score)
    return vocab, scores


def build_vocab_map(vocab):
    """
    Builds a {token_string: token_id} dictionary ONCE (fast lookups later).
    If two IDs share the same string, the first (lowest) ID wins.
    """
    vocab_map = {}
    for token_id, token in enumerate(vocab):
        vocab_map.setdefault(token, token_id)
    return vocab_map


def bpe_encode(text, vocab, vocab_map, scores):
    """
    Converts text into token IDs using Byte-Pair Encoding (BPE).

    Steps (same as llama2.c's encode()):
      1. Start with BOS (ID 1).
      2. Add a "dummy prefix" space so the first word is encoded like every
         other word (' Once', not 'Once').
      3. Split the text into single characters; look each up in the vocabulary.
         Unknown characters (emoji, accents ...) fall back to raw UTF-8 bytes.
      4. Repeatedly merge the adjacent pair whose merged string exists in the
         vocabulary with the HIGHEST score, until no merge is possible.
         Example: ' ','O','n','c','e' -> ' Once'  (ID 9038).
    """
    pieces = []

    if text:                                       # step 2: dummy prefix
        pieces.append(vocab_map[" "])

    for ch in text:                                # step 3: characters / bytes
        token_id = vocab_map.get(ch)
        if token_id is not None:
            pieces.append(token_id)
        else:
            pieces.extend(b + BYTE_FALLBACK_OFFSET for b in ch.encode("utf-8"))

    while len(pieces) >= 2:                        # step 4: greedy BPE merging
        best_score, best_id, best_idx = -1e10, -1, -1
        for i in range(len(pieces) - 1):
            merged = vocab[pieces[i]] + vocab[pieces[i + 1]]
            merged_id = vocab_map.get(merged)
            if merged_id is not None and scores[merged_id] > best_score:
                best_score, best_id, best_idx = scores[merged_id], merged_id, i
        if best_idx == -1:                         # nothing left to merge
            break
        pieces[best_idx] = best_id
        del pieces[best_idx + 1]

    return [BOS_TOKEN_ID] + pieces                 # step 1: BOS goes in front


def decode_token(token_id, vocab):
    """
    Converts one token ID back to readable text.
    Raw-byte tokens look like '<0x0A>' (= newline) and are turned into the
    real character. All other tokens already contain normal spaces.
    """
    piece = vocab[token_id]
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        return bytes([int(piece[3:5], 16)]).decode("utf-8", errors="replace")
    return piece


# ==============================================================================
# 3. LLAMA 2 TRANSFORMER FORWARD PASS (ONE TOKEN AT A TIME)
# ==============================================================================
def llama2_forward(token, pos, config, weights, k_cache, v_cache):
    """
    Pushes ONE token (at position `pos`) through all transformer layers and
    returns the logits (raw scores) for every possible NEXT token.

    The KV cache ("Key-Value cache") stores each earlier token's Key and Value
    vectors, so every new token only does its own new work instead of
    recomputing the whole history. Tokens MUST therefore be fed in order:
    pos = 0, 1, 2, ... (k_cache / v_cache are updated in place).
    """
    dim = config["dim"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    head_dim = dim // n_heads                 # size of each attention head (48)
    kv_mul = n_heads // n_kv_heads            # how many query heads share one K/V head

    # Step 1: embedding lookup. Token ID -> a learned 288-number vector.
    # .copy() because we modify x in place below and must not touch the weights.
    x = weights["token_embedding"][token].copy()

    # Step 2: RoPE (Rotary Position Embedding) angles for this position.
    # RoPE rotates pairs of numbers in Q and K by an angle proportional to the
    # position, so the dot-product Q.K depends on the *distance* between tokens.
    freqs = 1.0 / (
        ROPE_BASE ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    angles = pos * freqs
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    # Step 3: run through every transformer layer.
    for layer in range(config["n_layers"]):

        # ---- 3a. Self-attention block ("which earlier tokens matter now?") ----
        xb = rmsnorm(x, weights["rms_att"][layer])           # normalise first

        # Project to Query (what I look for), Key (what I offer), Value (what I give).
        q = (weights["wq"][layer] @ xb).reshape(n_heads, head_dim)
        k = (weights["wk"][layer] @ xb).reshape(n_kv_heads, head_dim)
        v = (weights["wv"][layer] @ xb).reshape(n_kv_heads, head_dim)

        # Rotate Q and K (NOT V) with RoPE. Adjacent pairs (0,1), (2,3) ... are
        # treated as 2-D points and rotated. Both new values are computed from
        # the old ones BEFORE assignment, so in-place update is safe.
        q0, q1 = q[:, 0::2], q[:, 1::2]
        q[:, 0::2], q[:, 1::2] = q0 * cos_a - q1 * sin_a, q0 * sin_a + q1 * cos_a
        k0, k1 = k[:, 0::2], k[:, 1::2]
        k[:, 0::2], k[:, 1::2] = k0 * cos_a - k1 * sin_a, k0 * sin_a + k1 * cos_a

        # Save this token's K and V so future tokens can attend to it.
        k_cache[layer, pos] = k
        v_cache[layer, pos] = v

        # Fetch K and V of tokens 0..pos (causal: a token never sees the future).
        # np.repeat duplicates K/V heads when several query heads share one
        # (Grouped-Query Attention). With 6 heads / 6 kv_heads it copies nothing.
        keys = np.repeat(k_cache[layer, : pos + 1], kv_mul, axis=1)   # (t, heads, head_dim)
        vals = np.repeat(v_cache[layer, : pos + 1], kv_mul, axis=1)

        # Attention scores = Q.K / sqrt(head_dim); softmax over earlier tokens.
        attn_scores = np.einsum("hd,thd->ht", q, keys) / np.sqrt(head_dim)
        attn_probs = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_probs /= np.sum(attn_probs, axis=-1, keepdims=True)

        # Weighted average of the Values, merge heads, project, then add to x
        # (residual connection: the layer only learns a *correction* to x).
        att_out = np.einsum("ht,thd->hd", attn_probs, vals).reshape(dim)
        x += weights["wo"][layer] @ att_out

        # ---- 3b. Feed-forward block ("think about what attention gathered") ----
        xb = rmsnorm(x, weights["rms_ffn"][layer])
        gate = weights["w1"][layer] @ xb
        up = weights["w3"][layer] @ xb
        # SiLU(g) = g * sigmoid(g). Very negative g makes exp() overflow to inf
        # and the result correctly becomes ~0, so the warning is silenced.
        with np.errstate(over="ignore"):
            silu = gate / (1.0 + np.exp(-gate))
        x += weights["w2"][layer] @ (silu * up)               # SwiGLU, then residual add

    # Step 4: final normalisation, then one score per vocabulary token.
    x = rmsnorm(x, weights["rms_final"])
    logits = weights["wcls"] @ x                              # shape (32000,)
    return logits


# ==============================================================================
# 4. PREDICTION + INTERACTIVE LOOP
# ==============================================================================
def predict_next_token(text, config, weights, vocab, vocab_map, scores, top_k=5):
    """
    Full pipeline for one prompt: tokenise -> prefill -> softmax -> top-k.
    Returns a dict with everything the display code needs.
    """
    head_dim = config["dim"] // config["n_heads"]
    seq_len = config["seq_len"]

    prompt_tokens = bpe_encode(text, vocab, vocab_map, scores)

    # The model can only see `seq_len` tokens. If the prompt is longer, keep BOS
    # plus the MOST RECENT tokens, because the next word depends mostly on them.
    truncated = len(prompt_tokens) > seq_len
    if truncated:
        prompt_tokens = [BOS_TOKEN_ID] + prompt_tokens[-(seq_len - 1):]

    # Fresh, empty KV cache for this prompt: (layers, positions, kv_heads, head_dim).
    k_cache = np.zeros(
        (config["n_layers"], seq_len, config["n_kv_heads"], head_dim), dtype=np.float32
    )
    v_cache = np.zeros_like(k_cache)

    start = time.perf_counter()

    # "Prefill": feed the prompt token by token so the KV cache gets filled.
    # Only the logits after the LAST prompt token matter for the prediction.
    logits = None
    for pos, tok in enumerate(prompt_tokens):
        logits = llama2_forward(tok, pos, config, weights, k_cache, v_cache)

    probs = softmax(logits)
    top_ids = np.argsort(probs)[::-1][:top_k]                 # highest probability first

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "n_prompt_tokens": len(prompt_tokens),
        "truncated": truncated,
        "top_ids": [int(t) for t in top_ids],
        "top_probs": [float(probs[t]) for t in top_ids],
        "elapsed_ms": elapsed_ms,
    }


def print_result(text, result, vocab):
    """Pretty-prints one prediction result."""
    best_id = result["top_ids"][0]
    best_piece = decode_token(best_id, vocab)

    if result["truncated"]:
        print("  [Warning] Prompt exceeded the context window; kept the most recent tokens.")
    print(f"\n  Prompt tokens      : {result['n_prompt_tokens']} (incl. BOS)")
    print(f"  Completed Sentence : {text}{best_piece}")
    print(f"  Predicted Next Word: {best_piece.strip()!r} (Token ID: {best_id})")
    print(f"  Inference Time     : {result['elapsed_ms']:.1f} ms")
    print("\n  Top 5 Next-Word Candidates:")
    for rank, (tid, p) in enumerate(zip(result["top_ids"], result["top_probs"]), start=1):
        print(f"    {rank}. {decode_token(tid, vocab)!r:<15} -> {p * 100:6.2f}%")
    print("-" * 56 + "\n")


def run_next_word_predictor(model_path, tokenizer_path, one_shot_prompt=None):
    print(f"Loading {model_path} and {tokenizer_path}...")
    config, weights = load_checkpoint(model_path)
    vocab, scores = load_tokenizer(tokenizer_path, config["vocab_size"])
    vocab_map = build_vocab_map(vocab)

    # One-shot mode (handy for scripts, tests and AWS batch jobs).
    if one_shot_prompt is not None:
        result = predict_next_token(one_shot_prompt, config, weights, vocab, vocab_map, scores)
        print_result(one_shot_prompt, result, vocab)
        return

    print("\n========================================================")
    print("  Llama 2 Single Next-Word Predictor Ready!")
    print("  Type a phrase and press ENTER to predict the next word.")
    print("  Type 'exit' to quit.")
    print("========================================================\n")

    while True:
        try:
            user_text = input("Enter text > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            print("Exiting...")
            break

        result = predict_next_token(user_text, config, weights, vocab, vocab_map, scores)
        print_result(user_text, result, vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 2 next-word predictor (NumPy, CPU)")
    parser.add_argument("--model", default="stories15M.bin", help="path to llama2.c model .bin")
    parser.add_argument("--tokenizer", default="tokenizer.bin", help="path to llama2.c tokenizer.bin")
    parser.add_argument("--prompt", default=None, help="run once on this prompt and exit")
    args = parser.parse_args()
    run_next_word_predictor(args.model, args.tokenizer, args.prompt)
