import os
import struct
import time
import numpy as np


# --- 1. VECTORIZED MATH HELPERS ---
def rmsnorm(x, weight):
  ss = np.mean(x**2) + 1e-5
  return (x / np.sqrt(ss)) * weight


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x)


# --- 2. MODEL & TOKENIZER LOADERS ---
def load_checkpoint(filepath):
  if not os.path.exists(filepath):
    raise FileNotFoundError(f'Model file not found: {filepath}')

  with open(filepath, 'rb') as f:
    header = struct.unpack('7i', f.read(28))
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
    shared_weights = vocab_size > 0
    vocab_size = abs(vocab_size)

    config = {
        'dim': dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_kv_heads': n_kv_heads,
        'vocab_size': vocab_size,
        'seq_len': seq_len,
    }

    data = np.fromfile(f, dtype=np.float32)
    offset = 0

    def take(shape):
      nonlocal offset
      size = int(np.prod(shape))
      arr = data[offset : offset + size].reshape(shape)
      offset += size
      return arr

    kv_dim = (dim * n_kv_heads) // n_heads
    weights = {
        'token_embedding': take((vocab_size, dim)),
        'rms_att': take((n_layers, dim)),
        'wq': take((n_layers, dim, dim)),
        'wk': take((n_layers, kv_dim, dim)),
        'wv': take((n_layers, kv_dim, dim)),
        'wo': take((n_layers, dim, dim)),
        'rms_ffn': take((n_layers, dim)),
        'w1': take((n_layers, hidden_dim, dim)),
        'w2': take((n_layers, dim, hidden_dim)),
        'w3': take((n_layers, hidden_dim, dim)),
        'rms_final': take((dim,)),
    }
    offset += seq_len * (dim // n_heads)
    weights['wcls'] = (
        weights['token_embedding']
        if shared_weights
        else take((vocab_size, dim))
    )

  return config, weights


def load_tokenizer(filepath, vocab_size):
  if not os.path.exists(filepath):
    raise FileNotFoundError(f'Tokenizer file not found: {filepath}')

  vocab, scores = [], []
  with open(filepath, 'rb') as f:
    _ = struct.unpack('i', f.read(4))
    for _ in range(vocab_size):
      score, length = struct.unpack('fi', f.read(8))
      token_bytes = f.read(length)
      vocab.append(token_bytes.decode('utf-8', errors='replace'))
      scores.append(score)
  return vocab, scores


def bpe_encode(text, vocab, scores):
  # Explicitly use \u2581 ('▁') so spaces never get lost in copy-paste
  text = '\u2581' + text.replace(' ', '\u2581')
  vocab_map = {token: i for i, token in enumerate(vocab)}

  tokens = [1]  # 1 = BOS (Beginning of Sequence)
  for ch in text:
    if ch in vocab_map:
      tokens.append(vocab_map[ch])

  while True:
    best_score = -1e10
    best_id = -1
    best_idx = -1
    for i in range(len(tokens) - 1):
      merged = vocab[tokens[i]] + vocab[tokens[i + 1]]
      tid = vocab_map.get(merged, -1)
      if tid != -1 and scores[tid] > best_score:
        best_score = scores[tid]
        best_id = tid
        best_idx = i
    if best_idx == -1:
      break
    tokens[best_idx] = best_id
    tokens.pop(best_idx + 1)

  return tokens


def decode_token(token_id, vocab):
  piece = vocab[token_id]
  # Convert SentencePiece byte tokens like <0x0A> into real characters (\n)
  if piece.startswith('<0x') and piece.endswith('>') and len(piece) == 6:
    byte_val = int(piece[3:5], 16)
    return chr(byte_val)
  return piece.replace('\u2581', ' ')


# --- 3. LLAMA 2 TRANSFORMER FORWARD PASS ---
def llama2_forward(token, pos, config, weights, k_cache, v_cache):
  dim = config['dim']
  n_heads = config['n_heads']
  n_kv_heads = config['n_kv_heads']
  head_dim = dim // n_heads
  kv_mul = n_heads // n_kv_heads

  x = weights['token_embedding'][token].copy()

  freqs = 1.0 / (
      10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
  )
  angles = pos * freqs
  cos_a, sin_a = np.cos(angles), np.sin(angles)

  for l in range(config['n_layers']):
    xb = rmsnorm(x, weights['rms_att'][l])

    q = (weights['wq'][l] @ xb).reshape(n_heads, head_dim)
    k = (weights['wk'][l] @ xb).reshape(n_kv_heads, head_dim)
    v = (weights['wv'][l] @ xb).reshape(n_kv_heads, head_dim)

    q0, q1 = q[:, 0::2], q[:, 1::2]
    q[:, 0::2], q[:, 1::2] = q0 * cos_a - q1 * sin_a, q0 * sin_a + q1 * cos_a

    k0, k1 = k[:, 0::2], k[:, 1::2]
    k[:, 0::2], k[:, 1::2] = k0 * cos_a - k1 * sin_a, k0 * sin_a + k1 * cos_a

    k_cache[l, pos] = k
    v_cache[l, pos] = v

    keys = np.repeat(k_cache[l, : pos + 1], kv_mul, axis=1)
    vals = np.repeat(v_cache[l, : pos + 1], kv_mul, axis=1)

    scores = np.einsum('hd,thd->ht', q, keys) / np.sqrt(head_dim)
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)

    xb_att = np.einsum('ht,thd->hd', probs, vals).reshape(dim)
    x += weights['wo'][l] @ xb_att

    xb = rmsnorm(x, weights['rms_ffn'][l])
    w1 = weights['w1'][l] @ xb
    w3 = weights['w3'][l] @ xb
    silu = w1 * (1.0 / (1.0 + np.exp(-w1)))
    x += weights['w2'][l] @ (silu * w3)

  x = rmsnorm(x, weights['rms_final'])
  return weights['wcls'] @ x


# --- 4. INTERACTIVE CHAT / STORY LOOP ---
def interactive_session(
    model_path, tokenizer_path, max_tokens=180, temperature=0.8
):
  print(f'Loading {model_path}...')
  config, weights = load_checkpoint(model_path)
  vocab, scores = load_tokenizer(tokenizer_path, config['vocab_size'])
  print(
      f"Ready! (Layers: {config['n_layers']}, Dim: {config['dim']}, Max"
      f" Context: {config['seq_len']})"
  )
  print("Type a story starter and press Enter. Type 'exit' to quit.\n")

  head_dim = config['dim'] // config['n_heads']

  while True:
    try:
      prompt = input('\n>>> Prompt: ').strip()
    except (KeyboardInterrupt, EOFError):
      break

    if not prompt:
      continue
    if prompt.lower() in ('exit', 'quit', '/bye'):
      print('Goodbye!')
      break

    # Reset KV cache for each new prompt (max context is 256 tokens)
    k_cache = np.zeros(
        (config['n_layers'], config['seq_len'], config['n_kv_heads'], head_dim),
        dtype=np.float32,
    )
    v_cache = np.zeros_like(k_cache)

    prompt_tokens = bpe_encode(prompt, vocab, scores)
    start_time = time.time()
    tokens_generated = 0
    token = prompt_tokens[0]

    print()  # Clean blank line before model output
    for pos in range(min(config['seq_len'], len(prompt_tokens) + max_tokens)):
      logits = llama2_forward(token, pos, config, weights, k_cache, v_cache)

      if pos + 1 < len(prompt_tokens):
        next_token = prompt_tokens[pos + 1]
      else:
        if temperature == 0.0:
          next_token = int(np.argmax(logits))
        else:
          probs = softmax(logits / temperature)
          next_token = int(np.random.choice(len(probs), p=probs))
        tokens_generated += 1

      if next_token == 2:  # EOS token
        break

      text_piece = decode_token(next_token, vocab)
      # Strip the single leading space on the very first word of output
      if pos == 0 and text_piece.startswith(' '):
        text_piece = text_piece[1:]
      print(text_piece, end='', flush=True)
      token = next_token

    elapsed = time.time() - start_time
    print(
        f'\n\n[{tokens_generated} tokens | {elapsed:.2f}s |'
        f' {tokens_generated / max(elapsed, 1e-5):.1f} tok/s]'
    )


if __name__ == '__main__':
  interactive_session(
      model_path='stories15M.bin',
      tokenizer_path='tokenizer.bin',
      max_tokens=180,
      temperature=0.8,
  )
