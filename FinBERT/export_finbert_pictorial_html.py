import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==============================================================================
# 1. CONFIGURATION & INPUT SENTENCE
# ==============================================================================
SENTENCE = "The bank raised interest rates"
LAYER_IDX = 0  # Choose any layer from 0 to 11
OUTPUT_HTML = "finbert_real_pictorial_qkv.html"
MODEL_NAME = "ProsusAI/finbert"

# ==============================================================================
# 2. EXTRACT REAL TENSOR VALUES FROM FINBERT
# ==============================================================================
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

inputs = tokenizer(SENTENCE, return_tensors="pt")
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
seq_len = len(tokens)

with torch.no_grad():
    # Full forward pass to capture hidden states at every layer + final logits
    outputs = model(**inputs, output_hidden_states=True)

    # Input tensor entering the selected encoder layer: shape [seq_len, 768]
    X = outputs.hidden_states[LAYER_IDX][0]

    # Grab self-attention module for the chosen layer
    attn_self = model.bert.encoder.layer[LAYER_IDX].attention.self

    # Linear projections: [seq_len, 768]
    Q_all = attn_self.query(X)
    K_all = attn_self.key(X)
    V_all = attn_self.value(X)

    # Reshape into [12 heads, seq_len, 64 dims]
    Q_heads = Q_all.view(seq_len, 12, 64).permute(1, 0, 2)
    K_heads = K_all.view(seq_len, 12, 64).permute(1, 0, 2)
    V_heads = V_all.view(seq_len, 12, 64).permute(1, 0, 2)

    # Compute scaled dot-product attention & head outputs Z for all 12 heads
    # scores shape: [12, seq_len, seq_len]
    scores = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) / 8.0
    attn_probs = F.softmax(scores, dim=-1)
    Z_heads = torch.matmul(attn_probs, V_heads)  # [12, seq_len, 64]

    # Sentiment probabilities
    logits = outputs.logits[0]
    probs = F.softmax(logits, dim=-1)
    id2label = model.config.id2label
    sentiment = {
        id2label[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))
    }

# Package tensors cleanly for JSON injection into HTML
payload = {
    "sentence": SENTENCE,
    "model": MODEL_NAME,
    "layer": LAYER_IDX,
    "tokens": tokens,
    "sentiment": sentiment,
    "X": X.round(decimals=3).tolist(),
    "Q": Q_heads.round(decimals=3).tolist(),
    "K": K_heads.round(decimals=3).tolist(),
    "V": V_heads.round(decimals=3).tolist(),
    "Z": Z_heads.round(decimals=3).tolist(),
    "attn": attn_probs.round(decimals=4).tolist(),
}

# ==============================================================================
# 3. GENERATE STANDALONE INTERACTIVE PICTORIAL HTML
# ==============================================================================
html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FinBERT Real Tensor Pictorial QKV Visualizer</title>
  <style>
    :root {{
      --bg: #0f172a; --panel: #1e293b; --panel-2: #334155; --border: #475569;
      --text: #f8fafc; --muted: #94a3b8; --accent: #38bdf8;
      --q-color: #22d3ee; --k-color: #f472b6; --v-color: #60a5fa; --z-color: #34d399;
    }}
    * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ margin: 0; padding: 20px; background: var(--bg); color: var(--text); }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    h1 {{ font-size: 1.45rem; margin: 0 0 6px 0; }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 18px; }}
    .top-bar {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 14px; background: var(--panel); padding: 14px 18px; border-radius: 10px;
      border: 1px solid var(--border); margin-bottom: 18px;
    }}
    .token-list {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .tok-btn {{
      padding: 7px 14px; border-radius: 6px; border: 1px solid var(--border);
      background: var(--panel-2); color: var(--text); font-family: monospace;
      font-size: 0.95rem; cursor: pointer; transition: 0.15s;
    }}
    .tok-btn:hover {{ border-color: var(--accent); }}
    .tok-btn.active {{ background: var(--accent); color: #0f172a; font-weight: 700; border-color: var(--accent); }}
    .legend {{ display: flex; align-items: center; gap: 10px; font-size: 0.8rem; color: var(--muted); }}
    .grad-bar {{
      width: 140px; height: 12px; border-radius: 4px;
      background: linear-gradient(90deg, #ef4444 0%, #1e293b 50%, #10b981 100%);
      border: 1px solid var(--border);
    }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 16px; }}
    .tab-btn {{
      padding: 8px 16px; border-radius: 6px; border: 1px solid var(--border);
      background: var(--panel); color: var(--muted); cursor: pointer; font-weight: 600;
    }}
    .tab-btn.active {{ background: var(--panel-2); color: var(--text); border-color: var(--accent); }}
    .card {{
      background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
      padding: 16px; margin-bottom: 18px;
    }}
    .card h2 {{ font-size: 1.05rem; margin: 0 0 10px 0; }}
    .strip-wrap {{
      display: flex; height: 22px; width: 100%; border-radius: 4px; overflow: hidden;
      border: 1px solid #000; background: #020617;
    }}
    .strip-cell {{ flex: 1; height: 100%; }}
    .heads-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; }}
    .head-card {{
      background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
    }}
    .head-header {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 10px; font-size: 0.9rem;
    }}
    .match-pill {{
      font-size: 0.78rem; padding: 3px 8px; border-radius: 999px;
      background: rgba(56, 189, 248, 0.15); color: var(--accent); font-family: monospace;
    }}
    .vec-row {{ display: grid; grid-template-columns: 74px 1fr; align-items: center; gap: 8px; margin-bottom: 7px; }}
    .vec-label {{ font-size: 0.78rem; font-weight: 700; font-family: monospace; }}
    .vec-strip {{
      display: grid; grid-template-columns: repeat(64, 1fr); height: 16px;
      border-radius: 3px; overflow: hidden; border: 1px solid #000;
    }}
    .attn-mini {{ display: flex; gap: 4px; margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--panel-2); }}
    .attn-tok {{ flex: 1; text-align: center; font-size: 0.7rem; font-family: monospace; }}
    .attn-bar-bg {{ height: 6px; background: #0f172a; border-radius: 2px; overflow: hidden; margin-top: 2px; }}
    .attn-bar-fill {{ height: 100%; background: var(--accent); }}
    table.matrix {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    table.matrix th, table.matrix td {{ border: 1px solid var(--border); padding: 6px; text-align: center; }}
    table.matrix th {{ background: var(--panel-2); }}
    .triplet {{ display: flex; flex-direction: column; gap: 2px; }}
    .triplet .vec-strip {{ height: 7px; border: none; }}
    .sent-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .sent-box {{ background: var(--panel-2); padding: 12px; border-radius: 8px; }}
    .sent-bar {{ height: 10px; border-radius: 4px; background: #0f172a; overflow: hidden; margin-top: 6px; }}
    .sent-fill {{ height: 100%; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>FinBERT Real Tensor Pictorial Q, K, V Visualizer</h1>
    <div class="sub">
      Model: <strong>{MODEL_NAME}</strong> | Layer: <strong>{LAYER_IDX}</strong> |
      Sentence: <strong>"{SENTENCE}"</strong>
    </div>

    <div class="top-bar">
      <div class="token-list" id="tokenButtons"></div>
      <div class="legend">
        <span>-2.5</span>
        <div class="grad-bar"></div>
        <span>+2.5 (Real Tensor Values)</span>
      </div>
    </div>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('single', this)">1. Word-by-Word 12-Head Inspector</button>
      <button class="tab-btn" onclick="switchTab('matrix', this)">2. All-Words &times; All-12-Heads QKV Matrix</button>
    </div>

    <div id="viewSingle">
      <div class="card">
        <h2 id="xTitle">Input Vector X [768 dims]</h2>
        <div class="strip-wrap" id="xStrip"></div>
      </div>
      <div class="heads-grid" id="headsGrid"></div>
    </div>

    <div id="viewMatrix" style="display:none;">
      <div class="card" style="overflow-x:auto;">
        <h2>All Tokens &times; All 12 Heads (Top=Q, Mid=K, Bot=V)</h2>
        <table class="matrix" id="matrixTable"></table>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <h2>Final Model Sentiment Prediction (After All 12 Layers)</h2>
      <div class="sent-grid" id="sentimentGrid"></div>
    </div>
  </div>

  <script>
    const DATA = {json.dumps(payload)};
    let selectedTok = DATA.tokens.includes("bank") ? DATA.tokens.indexOf("bank") : 0;

    function valToColor(v) {{
      const clamped = Math.max(-2.5, Math.min(2.5, v));
      const norm = (clamped + 2.5) / 5.0;
      if (norm < 0.5) {{
        const t = norm * 2;
        const r = Math.round(239 * (1 - t) + 30 * t);
        const g = Math.round(68 * (1 - t) + 41 * t);
        const b = Math.round(68 * (1 - t) + 59 * t);
        return `rgb(${{r}},${{g}},${{b}})`;
      }} else {{
        const t = (norm - 0.5) * 2;
        const r = Math.round(30 * (1 - t) + 16 * t);
        const g = Math.round(41 * (1 - t) + 185 * t);
        const b = Math.round(59 * (1 - t) + 129 * t);
        return `rgb(${{r}},${{g}},${{b}})`;
      }}
    }}

    function buildStripHTML(vec, label) {{
      return vec.map((v, idx) =>
        `<div class="strip-cell" style="background:${{valToColor(v)}}" title="${{label}}[${{idx}}] = ${{v}}"></div>`
      ).join("");
    }}

    function renderTokens() {{
      const wrap = document.getElementById("tokenButtons");
      wrap.innerHTML = `<span style="font-size:0.85rem;color:var(--muted);margin-right:4px;">Select Token:</span>` +
        DATA.tokens.map((tok, i) =>
          `<button class="tok-btn ${{i === selectedTok ? 'active' : ''}}" onclick="selectToken(${{i}})">${{tok}}</button>`
        ).join("");
    }}

    function selectToken(idx) {{
      selectedTok = idx;
      renderTokens();
      renderSingleView();
    }}

    function renderSingleView() {{
      const tok = DATA.tokens[selectedTok];
      document.getElementById("xTitle").textContent = `Input Vector X for "${{tok}}" [768 real dimensions]`;
      document.getElementById("xStrip").innerHTML = buildStripHTML(DATA.X[selectedTok], "X");

      const grid = document.getElementById("headsGrid");
      let html = "";
      for (let h = 0; h < 12; h++) {{
        const q = DATA.Q[h][selectedTok];
        const k = DATA.K[h][selectedTok];
        const v = DATA.V[h][selectedTok];
        const z = DATA.Z[h][selectedTok];
        const attnRow = DATA.attn[h][selectedTok];

        let maxIdx = 0;
        attnRow.forEach((p, i) => {{ if (p > attnRow[maxIdx]) maxIdx = i; }});

        const miniBars = DATA.tokens.map((t, i) => `
          <div class="attn-tok" title="${{tok}} -> ${{t}}: ${{(attnRow[i]*100).toFixed(1)}}%">
            <div>${{t.replace("##","")}}</div>
            <div class="attn-bar-bg"><div class="attn-bar-fill" style="width:${{attnRow[i]*100}}%"></div></div>
            <div style="color:var(--muted);font-size:0.65rem;">${{(attnRow[i]*100).toFixed(0)}}%</div>
          </div>
        `).join("");

        html += `
          <div class="head-card">
            <div class="head-header">
              <strong>Head ${{h}}</strong>
              <span class="match-pill">Top Key: "${{DATA.tokens[maxIdx]}}" (${{(attnRow[maxIdx]*100).toFixed(1)}}%)</span>
            </div>
            <div class="vec-row"><span class="vec-label" style="color:var(--q-color)">Query Q</span><div class="vec-strip">${{buildStripHTML(q, `H${{h}}.Q`)}}</div></div>
            <div class="vec-row"><span class="vec-label" style="color:var(--k-color)">Key K</span><div class="vec-strip">${{buildStripHTML(k, `H${{h}}.K`)}}</div></div>
            <div class="vec-row"><span class="vec-label" style="color:var(--v-color)">Value V</span><div class="vec-strip">${{buildStripHTML(v, `H${{h}}.V`)}}</div></div>
            <div class="vec-row"><span class="vec-label" style="color:var(--z-color)">Out Z</span><div class="vec-strip">${{buildStripHTML(z, `H${{h}}.Z`)}}</div></div>
            <div class="attn-mini">${{miniBars}}</div>
          </div>
        `;
      }}
      grid.innerHTML = html;
    }}

    function renderMatrixView() {{
      const tbl = document.getElementById("matrixTable");
      let headRow = `<tr><th>Token</th>` + Array.from({{length: 12}}, (_, h) => `<th>H${{h}}</th>`).join("") + `</tr>`;
      let bodyRows = DATA.tokens.map((tok, tIdx) => {{
        let cells = Array.from({{length: 12}}, (_, h) => `
          <td>
            <div class="triplet">
              <div class="vec-strip">${{buildStripHTML(DATA.Q[h][tIdx], `H${{h}}.Q`)}}</div>
              <div class="vec-strip">${{buildStripHTML(DATA.K[h][tIdx], `H${{h}}.K`)}}</div>
              <div class="vec-strip">${{buildStripHTML(DATA.V[h][tIdx], `H${{h}}.V`)}}</div>
            </div>
          </td>
        `).join("");
        return `<tr><td><strong>${{tok}}</strong></td>${{cells}}</tr>`;
      }}).join("");
      tbl.innerHTML = headRow + bodyRows;
    }}

    function renderSentiment() {{
      const colors = {{ positive: "#10b981", negative: "#ef4444", neutral: "#94a3b8" }};
      document.getElementById("sentimentGrid").innerHTML = Object.entries(DATA.sentiment).map(([label, pct]) => `
        <div class="sent-box">
          <div style="display:flex;justify-content:space-between;font-weight:700;">
            <span>${{label.toUpperCase()}}</span>
            <span>${{pct.toFixed(2)}}%</span>
          </div>
          <div class="sent-bar"><div class="sent-fill" style="width:${{pct}}%;background:${{colors[label] || '#38bdf8'}}"></div></div>
        </div>
      `).join("");
    }}

    function switchTab(view, btn) {{
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("viewSingle").style.display = view === "single" ? "block" : "none";
      document.getElementById("viewMatrix").style.display = view === "matrix" ? "block" : "none";
    }}

    renderTokens();
    renderSingleView();
    renderMatrixView();
    renderSentiment();
  </script>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Done! Saved real tensor visualization to: {OUTPUT_HTML}")
