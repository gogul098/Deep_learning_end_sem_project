"""
legal_hybrid.py (v7): Hyena long-document encoder + Llama verifier (shortlist → verify → fuse).

  judgment PDF ─► sentences ─► InLegalBERT (frozen, overlapping windows, nothing cut) ─► one vector per sentence
     ─► Hyena encoder over ALL sentences (FFT long convolution, sub-quadratic) with label-wise attention and a
        label-correlation layer ─► P(section) for 100 IPC sections + sentence importance + evidence per section
     ─► shortlist: top candidates (high recall)
     ─► Llama-3.2-3B (QLoRA) reads the opening + key passages + each candidate's evidence and answers
        "Section 302: yes/no" for every candidate → P_llm(section)
     ─► fusion: α·P_encoder + (1−α)·P_llm, threshold tuned on the dev set
  Llama reads ~3k tokens however long the judgment is, so GPU memory stays flat.
"""
import math, re, random
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def fftconv(u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal linear convolution y[t] = sum_{s<=t} h[s] u[t-s] via FFT (zero-padded to >=2L: no wrap-around).
    u: (..., L), h: (..., L) broadcastable. Computed in fp32."""
    L = u.shape[-1]
    n = next_pow2(2 * L)
    U = torch.fft.rfft(u.float(), n=n)
    H = torch.fft.rfft(h.float(), n=n)
    return torch.fft.irfft(U * H, n=n)[..., :L]


class MultiBandFilter(nn.Module):
    """Long-convolution filter with explicit LOW / HIGH frequency bands.

    LOW band : long filter built from damped oscillatory modes  h(t) = Σ_n R_n·a_n·e^{-a_n t}·cos(w_n t)
               (smooth, long memory → whole-document context)
    HIGH band: short explicit filter (k taps) → exact local detail (section numbers, names, dates)
    The two are combined in the frequency domain with a learnable soft cutoff per channel:
        H(ω) = m(ω)·H_long(ω) + (1 − m(ω))·H_short(ω),   m(ω) = σ((ω_c − ω)/τ)
    ω is normalised so 1.0 = Nyquist. Initialising ω_c = 1/R makes the low band anti-aliased for ×R downsampling.
    FFTs run in fp32 and are zero-padded to 2L (no circular wrap-around).
    """

    def __init__(self, d: int, n_modes: int = 8, short_len: int = 7, cutoff: float = 1 / 16,
                 tau: float = 0.01, bidirectional: bool = True):
        super().__init__()
        self.d, self.tau, self.bidirectional = d, tau, bidirectional
        nd = 2 if bidirectional else 1
        self.log_rate = nn.Parameter(torch.empty(nd, d, n_modes).uniform_(math.log(1e-4), math.log(5e-2)))
        self.freq = nn.Parameter(torch.rand(nd, d, n_modes) * math.pi * cutoff)
        self.residues = nn.Parameter(torch.randn(nd, d, n_modes) / math.sqrt(n_modes))
        self.short = nn.Parameter(torch.randn(nd, d, short_len) / short_len)
        c = min(max(cutoff, 1e-3), 0.999)
        self.cutoff_logit = nn.Parameter(torch.full((d,), math.log(c / (1 - c))))

    def long_kernel(self, L: int) -> torch.Tensor:
        """(nd, d, L) long filter; accumulated mode-by-mode to keep memory at O(d·L)."""
        t = torch.arange(L, device=self.residues.device, dtype=torch.float32)
        rate = self.log_rate.float().exp()
        h = torch.zeros(*self.residues.shape[:2], L, device=t.device)
        for m in range(rate.shape[-1]):
            a = rate[..., m:m + 1]
            h = h + self.residues[..., m:m + 1].float() * a * torch.exp(-a * t) * torch.cos(self.freq[..., m:m + 1].float() * t)
        return h

    def band_masks(self, n_fft: int, device):
        omega = torch.linspace(0, 1, n_fft // 2 + 1, device=device)
        cutoff = torch.sigmoid(self.cutoff_logit.float())[:, None]
        m_low = torch.sigmoid((cutoff - omega) / self.tau)
        return m_low, 1 - m_low

    def forward(self, u: torch.Tensor):
        """u: (B, d, L) → (y_low, y_high), each (B, d, L) fp32."""
        L = u.shape[-1]
        n = next_pow2(2 * L)
        m_low, m_high = self.band_masks(n, u.device)
        H_low = torch.fft.rfft(self.long_kernel(L), n=n) * m_low
        H_high = torch.fft.rfft(self.short.float(), n=n) * m_high
        y_low = y_high = 0
        for i in range(H_low.shape[0]):
            x = u.float() if i == 0 else u.float().flip(-1)       # second direction = anti-causal pass
            X = torch.fft.rfft(x, n=n)
            yl = torch.fft.irfft(X * H_low[i], n=n)[..., :L]
            yh = torch.fft.irfft(X * H_high[i], n=n)[..., :L]
            if i == 1:
                yl, yh = yl.flip(-1), yh.flip(-1)
            y_low, y_high = y_low + yl, y_high + yh
        return y_low, y_high


class HyenaBlock(nn.Module):
    """Hyena operator y = W_o·(q ⊙ (h ∗ (k ⊙ v))) with the filter split into low/high bands."""

    def __init__(self, d: int, dropout: float = 0.1, **filter_kw):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 3 * d)
        self.short_conv = nn.Conv1d(3 * d, 3 * d, 3, padding=1, groups=3 * d)
        self.filter = MultiBandFilter(d, **filter_kw)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(d))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        z = self.short_conv(self.in_proj(self.norm1(x)).transpose(1, 2))
        q, k, v = z.chunk(3, dim=1)
        y_low, y_high = self.filter(k * v)
        low = self.out_proj((q.float() * y_low).transpose(1, 2).to(x.dtype))
        high = self.out_proj((q.float() * y_high).transpose(1, 2).to(x.dtype))
        x = x + self.drop(low + high + self.out_bias)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, low, high



def lowpass_downsample(x: torch.Tensor, R: int, tau: float = 0.01) -> torch.Tensor:
    """Anti-aliased decimation along the sequence: FFT low-pass at the new Nyquist (1/R), then average-pool by R.
    x: (B, L, d) → (B, ceil(L/R), d). R = 1 returns x unchanged."""
    if R <= 1:
        return x.float()
    B, L, d = x.shape
    n = next_pow2(2 * L)
    X = torch.fft.rfft(x.float().transpose(1, 2), n=n)
    omega = torch.linspace(0, 1, n // 2 + 1, device=x.device)
    X = X * torch.sigmoid((1.0 / R - omega) / tau)
    y = torch.fft.irfft(X, n=n)[..., :L]
    pad = (-L) % R
    if pad:
        y = F.pad(y, (0, pad), mode="replicate")
    return F.avg_pool1d(y, R, R).transpose(1, 2)



class SentenceEmbedder:
    """Frozen InLegalBERT → one vector per sentence, with NO hard chunk boundaries and NO sentence truncation:
    the whole judgment is tokenised as one stream and read in overlapping 510-token windows (stride 255). Every token
    takes its states from the window where it sits closest to the centre, so it always has ≥ ~128 tokens of context on
    both sides. A sentence vector is the mean of its tokens. Cost grows linearly with the judgment's length."""

    def __init__(self, model, tokenizer, device, dtype=torch.float32, window: int = 510, batch: int = 16, **_):
        self.model, self.tok, self.device = model.to(device).eval(), tokenizer, torch.device(device)
        self.dtype, self.window, self.batch = dtype, window, batch
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = self.model.config.hidden_size

    @classmethod
    def load(cls, name: str, device, dtype=None, **kw):
        from transformers import AutoTokenizer, AutoModel
        dev = torch.device(device)
        dtype = dtype or (torch.float16 if dev.type == "cuda" else torch.float32)
        return cls(AutoModel.from_pretrained(name, dtype=dtype), AutoTokenizer.from_pretrained(name), dev, dtype, **kw)

    @torch.no_grad()
    def __call__(self, sentences: List[str]) -> torch.Tensor:
        dev = self.device
        ids = self.tok(list(sentences), add_special_tokens=False)["input_ids"]
        flat, sid = [], []
        for j, t in enumerate(ids):
            t = t or [self.tok.unk_token_id]
            flat += t; sid += [j] * len(t)
        N, W = len(flat), self.window
        starts = [0] if N <= W else sorted(set(list(range(0, N - W + 1, W // 2)) + [N - W]))
        states = torch.zeros(N, self.dim, device=dev)
        best = torch.full((N,), float("inf"), device=dev)
        cls_id, sep_id, pad_id = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id or 0
        for b in range(0, len(starts), self.batch):
            group = starts[b:b + self.batch]
            seqs = [[cls_id] + flat[s:s + W] + [sep_id] for s in group]
            T = max(map(len, seqs))
            x = torch.tensor([q + [pad_id] * (T - len(q)) for q in seqs], device=dev)
            m = torch.tensor([[1] * len(q) + [0] * (T - len(q)) for q in seqs], device=dev)
            with torch.autocast(dev.type, dtype=self.dtype, enabled=dev.type == "cuda"):
                h = self.model(input_ids=x, attention_mask=m).last_hidden_state.float()
            for r, s in enumerate(group):
                n = min(W, N - s)
                pos = torch.arange(s, s + n, device=dev)
                dist = (pos - (s + n / 2)).abs()
                upd = dist < best[pos]
                states[pos[upd]] = h[r, 1:1 + n][upd]
                best[pos[upd]] = dist[upd]
        sid = torch.tensor(sid, device=dev)
        S = len(sentences)
        out = torch.zeros(S, self.dim, device=dev).index_add_(0, sid, states)
        cnt = torch.zeros(S, device=dev).index_add_(0, sid, torch.ones(N, device=dev))
        return out / cnt.clamp_min(1)[:, None]



# ───────────────────────────── document utilities ─────────────────────────────
@dataclass
class Doc:
    ids: torch.Tensor          # (L,)
    sent_ids: torch.Tensor     # (L,)
    sentences: List[str]
    counts: List[int]          # tokens per kept sentence
    flags: Optional[List[bool]] = None   # True = a statute citation was masked in this sentence (importance target)


def split_sentences(text: str, min_chars: int = 25) -> List[str]:
    parts = re.split(r"(?<=[.?!;:])\s+|\n{2,}", re.sub(r"[ \t]+", " ", text))
    out, buf = [], ""
    for p in (p.strip() for p in parts):
        if not p:
            continue
        buf = f"{buf} {p}".strip() if buf else p
        if len(buf) >= min_chars:
            out.append(buf); buf = ""
    if buf:
        out.append(buf)
    return out


def prepare_doc(sentences: List[str], tokenizer, max_tokens: int, flags: Optional[List[bool]] = None) -> Doc:
    enc = tokenizer([" " + s.strip() for s in sentences], add_special_tokens=False)["input_ids"]
    ids, sid, kept, counts, kf = [], [], [], [], []
    flags = flags if flags is not None else [False] * len(sentences)
    for s, t, fl in zip(sentences, enc, flags):
        if not t:
            continue
        if len(ids) + len(t) > max_tokens:
            t = t[: max_tokens - len(ids)]
            if len(t) < 8:
                break
        sid += [len(kept)] * len(t); ids += t; kept.append(s.strip()); counts.append(len(t)); kf.append(bool(fl))
        if len(ids) >= max_tokens:
            break
    return Doc(torch.tensor(ids, dtype=torch.long), torch.tensor(sid, dtype=torch.long), kept, counts, kf)


# ───────────────────────────── metrics ─────────────────────────────
SEC_RE = re.compile(r"Section\s+(\d+[A-Z]*(?:\(\d+\))?)", re.I)


def parse_sections(text: str, valid=None) -> set:
    s = {f"Section {m.upper()}" for m in SEC_RE.findall(text)}
    return s & set(valid) if valid is not None else s


def micro_prf(golds, preds):
    tp = sum(len(g & p) for g, p in zip(golds, preds))
    fp = sum(len(p - g) for g, p in zip(golds, preds))
    fn = sum(len(g - p) for g, p in zip(golds, preds))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R) if P + R else 0.0, 3)}


# ═══════════════════════════════ shared helpers ═══════════════════════════════
from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint as _ckpt

PLACEHOLDER_RE = re.compile(r"\[(?:SECTION|CODE)\]")


def clean_sentences(sentences: List[str]):
    """Strip masking placeholders; flag sentences that contained a masked statute citation (importance targets)."""
    out, flags = [], []
    for s in sentences:
        flags.append("[SECTION]" in s)
        out.append(re.sub(r"\s{2,}", " ", PLACEHOLDER_RE.sub(" ", s)).strip())
    return out, flags


def select_with_opening(scores: torch.Tensor, counts: List[int], budget: int, head_frac: float = 0.5):
    """Opening sentences up to head_frac·budget, then the highest-scoring later sentences. Returns (sorted idx, n_head)."""
    chosen, used, n_head = [], 0, 0
    for j, c in enumerate(counts):
        if used + c > budget * head_frac:
            break
        chosen.append(j); used += c; n_head += 1
    start = n_head
    for j in torch.argsort(scores.detach().float(), descending=True).tolist():
        if j < start or used + counts[j] > budget:
            continue
        chosen.append(j); used += counts[j]
    return sorted(chosen), n_head


def opening_selection_text(doc: Doc, idx: List[int], n_head: int) -> str:
    head = " ".join(doc.sentences[j] for j in idx if j < n_head)
    rest = "\n".join(f"- {doc.sentences[j]}" for j in idx if j >= n_head)
    return (f"Opening of the judgment:\n{head}\n\n" if head else "") + (f"Other key passages:\n{rest}" if rest else "")


def _base(llm):
    return llm.get_base_model() if hasattr(llm, "get_base_model") else llm


def _decoder_layers(llm):
    return _base(llm).model.layers


def llm_body(llm):
    return _base(llm).model


def llm_head(llm):
    return _base(llm).lm_head


def chat_prompt_embeds(llm, tokenizer, system: str, user: str, answer: Optional[str] = None):
    """Embeddings for a chat prompt (+ optional answer). Returns (embeds, labels, prompt_len)."""
    emb = llm.get_input_embeddings(); dev = emb.weight.device
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = _tok(tokenizer, prompt, dev); n = len(ids); labels = [-100] * n
    if answer is not None:
        full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)
        assert full.startswith(prompt), "chat template is not prefix-stable"
        ans = _tok(tokenizer, full[len(prompt):], dev)
        ids = torch.cat([ids, ans]); labels += ans.tolist()
    return emb(ids)[None], torch.tensor([labels], device=dev), n


def answer_loss(llm, embeds, labels):
    """Cross-entropy on answer tokens only (LM head applied to those positions only → small memory)."""
    h = llm_body(llm)(inputs_embeds=embeds, use_cache=False).last_hidden_state[0, :-1]
    tgt = labels[0, 1:]; sel = tgt != -100
    return F.cross_entropy(llm_head(llm)(h[sel]).float(), tgt[sel])


_USS_RE = re.compile(r"\b[Uu]/[Ss]{1,2}\.?\s*\d{1,3}[A-Z]{0,2}(?:\s*\(\s*[0-9a-z]{1,3}\s*\))*(?:\s*(?:,|/|&|and|or)\s*\d{1,3}[A-Z]{0,2}(?:\s*\(\s*[0-9a-z]{1,3}\s*\))*)*"
                     r"|\b[Ss]{2}\.\s*\d{1,3}[A-Z]{0,2}(?:\s*(?:,|/|&|and|or)\s*\d{1,3}[A-Z]{0,2})*")


def view_sentences(sentences: List[str], view: str = "full"):
    """Sentences + importance flags for a document view.
    full : citation numbers hidden ([SECTION] placeholders stripped), all sentences kept
    facts: every sentence that cited a section is DROPPED → the model must infer the law from the facts alone"""
    sentences = [_USS_RE.sub("[SECTION]", x) for x in sentences]       # forms the v4 masking missed ("u/ss.376 …")
    s, flags = clean_sentences(sentences)
    if view == "facts":
        kept = [(a, f) for a, f in zip(s, flags) if not f]
        if len(kept) >= 3:
            s, flags = [a for a, _ in kept], [False] * len(kept)
    return s, flags


SEC_CITE_RE = None


def hide_cited_sections(text: str) -> str:
    """Hide statute citations in a raw document (same rules as data preparation): 'Section(s) 376(2)(n), 377 & 506 IPC' →
    '[SECTION] [CODE]'. Used at inference for a fair test on judgments that already name their sections."""
    global SEC_CITE_RE
    if SEC_CITE_RE is None:
        num = r"\d{1,3}(?!\d)(?:-?(?-i:[A-Z]{1,2})\b)?(?:\s*\(\s*[0-9a-z]{1,3}\s*\))*(?:\s+Part\s+I{1,2}\b)?"
        join = r"\s*(?:,|/|&|\band\b|\bor\b|r/w|read\s+with|alongwith|along\s+with)\s*(?:(?:[Ss]ections?(?:\(s\))?|[Ss]ecs?\.?|S\.)\s*)?"
        sec = r"(?:\b[Ss]ections?(?:\s*\(s\))?|\b[Ss]ecs?\.?|(?<![A-Za-z.])S\.|\b[Uu]/[Ss]{1,2}\.?|\b[Ss]{2}\.|\b[Uu]nder\s+[Ss]ection)"
        SEC_CITE_RE = (re.compile(rf"{sec}\s*{num}(?:{join}{num})*", re.I),
                       re.compile(r"\b(?:Indian\s+Penal\s+Code|I\.\s?P\.\s?C\.?|IPC)\b", re.I))
    a, b = SEC_CITE_RE
    return b.sub("[CODE]", a.sub("[SECTION]", text))


# ═══════════════════════════════ v7: Hyena encoder (+ label correlation) → shortlist → Llama verifier → fusion ═══════════════════════════════
class LegalHyenaEncoder(nn.Module):
    """Hyena encoder over the sequence of sentence vectors (the WHOLE judgment, sub-quadratic). Outputs:
       scores  (S,)            sentence importance (HIGH band: sharp, sentence-level detail)
       logits  (n_labels,)     IPC-section logits: label-wise attention over sentences, then a label-correlation layer
                               (the 100 sections attend to each other, so co-occurring sections such as 302 + 34 support each other)
       attn    (n_labels, S)   which sentences each section looked at (evidence)"""

    def __init__(self, in_dim: int = 768, d: int = 512, n_layers: int = 4, n_labels: int = 100, band_cutoff: float = 0.25,
                 n_modes: int = 8, short_len: int = 3, dropout: float = 0.1, label_init: Optional[torch.Tensor] = None,
                 label_mix: bool = True):
        super().__init__()
        self.register_buffer("emb_mean", torch.zeros(in_dim))     # corpus statistics of InLegalBERT vectors:
        self.register_buffer("emb_std", torch.ones(in_dim))       # removing the shared component makes sentences distinct
        self.in_norm = nn.LayerNorm(in_dim)
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([HyenaBlock(d, dropout=dropout, n_modes=n_modes, short_len=short_len,
                                                cutoff=band_cutoff) for _ in range(n_layers)])
        self.norm_low, self.norm_high, self.norm_out = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.score_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.label_query = nn.Parameter(torch.randn(n_labels, d) / math.sqrt(d))
        self.label_from_desc = nn.Linear(in_dim, d, bias=False)
        self.register_buffer("label_desc", label_init.float() if label_init is not None else torch.zeros(n_labels, in_dim))
        self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d)
        self.label_mix = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d, dropout=dropout, batch_first=True,
                                                    norm_first=True) if label_mix else None
        self.label_w = nn.Parameter(torch.randn(n_labels, d) / math.sqrt(d))
        self.label_b = nn.Parameter(torch.full((n_labels,), -2.0))

    def forward(self, sent_emb: torch.Tensor):
        """sent_emb: (S, in_dim) sentence vectors from SentenceEmbedder (or the cache)."""
        z = (sent_emb.float() - self.emb_mean) / self.emb_std
        x = self.inp(self.in_norm(z))[None]                         # (1, S, d)
        low_sum = high_sum = 0
        for blk in self.blocks:
            x, low, high = blk(x)
            low_sum, high_sum = low_sum + low, high_sum + high
        x = self.norm_out(x)
        s_low, s_high = self.norm_low(low_sum)[0], self.norm_high(high_sum)[0]
        scores = self.score_head(torch.cat([s_high, x[0]], -1)).squeeze(-1)                     # (S,)
        q = self.label_query + self.label_from_desc((self.label_desc - self.emb_mean) / self.emb_std)   # (n_labels, d)
        att = torch.softmax(q @ self.k(x[0]).T / math.sqrt(q.shape[-1]), dim=-1)                 # (n_labels, S)
        ctx = att @ self.v(x[0] + s_low)                                                          # (n_labels, d)
        if self.label_mix is not None:
            ctx = self.label_mix((ctx + q)[None])[0]                                              # sections see each other
        logits = (ctx * self.label_w).sum(-1) + self.label_b
        return {"scores": scores, "logits": logits, "attn": att}


def ensemble_probs(encoders, sent_emb):
    """Average section probabilities of several encoders; importance + evidence from the first (the full-data encoder)."""
    outs = [e(sent_emb) for e in encoders]
    p = torch.stack([torch.sigmoid(o["logits"].float()) for o in outs]).mean(0)
    return p, outs[0]


def macro_f1(golds, preds, labels):
    f = []
    for c in labels:
        tp = sum(1 for g, p in zip(golds, preds) if c in g and c in p)
        fp = sum(1 for g, p in zip(golds, preds) if c not in g and c in p)
        fn = sum(1 for g, p in zip(golds, preds) if c in g and c not in p)
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return round(float(sum(f) / len(f)), 3) if f else 0.0


def shortlist(probs: torch.Tensor, k_min: int = 5, k_max: int = 10, min_p: float = 0.05) -> List[int]:
    """Candidate sections: the top k_max by encoder probability, keeping at least k_min, dropping those below min_p."""
    order = torch.argsort(probs.float(), descending=True).tolist()
    return [j for r, j in enumerate(order[:k_max]) if r < k_min or float(probs[j]) >= min_p]


def clip_words(s: str, n: int = 45) -> str:
    w = s.split()
    return s if len(w) <= n else " ".join(w[:n]) + " …"


def key_passages(doc: "Doc", scores: torch.Tensor, budget: int = 1500, head: int = 400) -> str:
    """Opening of the judgment + its most important sentences (Hyena importance), in document order."""
    idx, n_head = select_with_opening(scores, doc.counts, budget, head_frac=head / max(1, budget))
    return opening_selection_text(doc, idx, n_head)


def evidence_for(doc: "Doc", attn: torch.Tensor, j: int, n: int = 2) -> List[str]:
    if not doc.sentences:
        return []
    top = attn[j].float().topk(min(n, len(doc.sentences))).indices.tolist()
    return [clip_words(doc.sentences[t]) for t in top]


VERIFY_SYSTEM = ("You are a legal assistant for Indian criminal law. You are given the key passages of a Supreme Court judgment "
                 "and a shortlist of candidate Indian Penal Code (IPC) sections proposed by a document classifier. Some candidates "
                 "are wrong. Decide for each candidate whether it applies to the accused in this case.")
GENERATE_SYSTEM = ("You are a legal assistant for Indian criminal law. Read the key passages of the judgment and list every "
                   "Indian Penal Code (IPC) section that applies, one per line, as 'Section <number>'.")


def verify_user(passages: str, cands: List[str], desc: dict, evidence: Optional[dict] = None) -> str:
    lines = []
    for c in cands:
        d = (desc.get(c) or "").strip()
        d = (d[:90].rsplit(" ", 1)[0] + "…") if len(d) > 90 else d
        ev = " | ".join(f'"{e}"' for e in (evidence or {}).get(c, []))
        lines.append(f"- {c}" + (f" ({d})" if d else "") + (f". Evidence: {ev}" if ev else ""))
    return (f"JUDGMENT (opening and key passages):\n{passages}\n\nCANDIDATE IPC SECTIONS:\n" + "\n".join(lines) +
            "\n\nFor each candidate, in the order given, answer one line: 'Section <number>: yes' if it applies to this case, "
            "otherwise 'Section <number>: no'.")


def generate_user(passages: str) -> str:
    return f"JUDGMENT (opening and key passages):\n{passages}\n\nWhich IPC sections apply to this case?"


def verify_answer(cands: List[str], gold: set) -> str:
    return "\n".join(f"{c}: {'yes' if c in gold else 'no'}" for c in cands)


def chat_ids(tokenizer, system: str, user: str, answer: Optional[str] = None):
    """Token ids (+ labels on the answer only) for a chat prompt. Returns (ids list, labels list, prompt text)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    labels = [-100] * len(ids)
    if answer is not None:
        full = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)
        assert full.startswith(prompt), "chat template is not prefix-stable"
        ans = tokenizer(full[len(prompt):], add_special_tokens=False).input_ids
        ids, labels = ids + ans, labels + ans
    return ids, labels, prompt


def answer_loss_ids(llm, ids: List[int], labels: List[int]):
    dev = llm_head(llm).weight.device
    x = torch.tensor([ids], device=dev)
    h = llm_body(llm)(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
    tgt = torch.tensor(labels[1:], device=dev); sel = tgt != -100
    return F.cross_entropy(llm_head(llm)(h[sel]).float(), tgt[sel])


def yes_no_ids(tokenizer):
    """Kept for compatibility: decision tokens are now found per position by _decision_point."""
    return None


def _decision_point(tokenizer, text: str):
    """Token ids of `text` up to where ' yes' and ' no' continuations first differ, and the yes / no token ids there
    (works whether the tokenizer has ' yes' as one token or splits it)."""
    a = tokenizer(text + " yes", add_special_tokens=False).input_ids
    b = tokenizer(text + " no", add_special_tokens=False).input_ids
    k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
    yes, no = {a[k]}, {b[k]}
    for w, bucket in ((" Yes", yes), (" No", no)):
        c = tokenizer(text + w, add_special_tokens=False).input_ids
        if len(c) > k and c[:k] == a[:k] and c[k] not in (yes | no):
            bucket.add(c[k])
    return a[:k], sorted(yes), sorted(no)


@torch.no_grad()
def score_candidates(llm, tokenizer, system: str, user: str, cands: List[str], yn=None) -> List[float]:
    """P(yes) for every candidate, reading the answer the model itself writes ('Section 302: yes/no', one per line).
    The prompt is read once (KV cache); each candidate then costs a few tokens."""
    from transformers import DynamicCache
    body, head = llm_body(llm), llm_head(llm)
    dev = head.weight.device
    _, _, prompt = chat_ids(tokenizer, system, user)
    cache, fed, answer, probs = DynamicCache(), [], "", []
    for i, c in enumerate(cands):
        answer += ("\n" if i else "") + f"{c}:"
        ids, yes, no = _decision_point(tokenizer, prompt + answer)
        if ids[:len(fed)] != fed or len(ids) <= len(fed):          # re-tokenisation changed earlier tokens: start over
            cache, fed = DynamicCache(), []
        out = body(input_ids=torch.tensor([ids[len(fed):]], device=dev), past_key_values=cache, use_cache=True)
        fed = ids
        lg = head(out.last_hidden_state[:, -1]).float()[0]
        p = torch.sigmoid(torch.logsumexp(lg[yes], 0) - torch.logsumexp(lg[no], 0)).item()
        probs.append(p)
        answer += " yes" if p >= 0.5 else " no"
    return probs


@torch.no_grad()
def generate_text(llm, tokenizer, system: str, user: str, max_new_tokens: int = 200) -> str:
    from transformers import DynamicCache
    body, head = llm_body(llm), llm_head(llm)
    dev = head.weight.device
    ids, _, _ = chat_ids(tokenizer, system, user)
    eos = llm.generation_config.eos_token_id if getattr(llm, "generation_config", None) is not None else None
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tokenizer.eos_token_id}
    cache = DynamicCache()
    out = body(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True)
    last, new = out.last_hidden_state[:, -1], []
    for _ in range(max_new_tokens):
        nxt = int(head(last).float().argmax(-1).item())
        if nxt in eos:
            break
        new.append(nxt)
        out = body(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
        last = out.last_hidden_state[:, -1]
    return tokenizer.decode(new, skip_special_tokens=True).strip()


def fuse(p_enc: float, p_llm: Optional[float], alpha: float) -> float:
    """Final score of a section: α·encoder + (1−α)·verifier (encoder only for sections outside the shortlist)."""
    return p_enc if p_llm is None else alpha * p_enc + (1 - alpha) * p_llm


# ───────────────────────────── loading a trained v7 model (laptop / server) ─────────────────────────────
def load_trained(out_dir: str, device="cuda:0", base_model: Optional[str] = None, load_4bit: bool = True,
                 legal_encoder: Optional[str] = None, embedder_device: Optional[str] = None):
    """Returns (tokenizer, llm, encoders, ck); the sentence embedder is ck["embedder"]."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    ck = torch.load(f"{out_dir}/hybrid_model.pt", map_location="cpu", weights_only=False)
    dev = torch.device(device)
    ampere = dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8
    dtype = (torch.bfloat16 if ampere else torch.float16) if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(f"{out_dir}/llama_lora")
    kw = dict(dtype=dtype)
    if load_4bit and dev.type == "cuda":
        kw.update(device_map={"": dev.index or 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    base = AutoModelForCausalLM.from_pretrained(base_model or ck["base_model"], **kw)
    if not (load_4bit and dev.type == "cuda"):
        base = base.to(dev)
    llm = PeftModel.from_pretrained(base, f"{out_dir}/llama_lora").eval()
    encoders = []
    for sd in ck["encoders"]:
        e = LegalHyenaEncoder(**ck["encoder_config"]); e.load_state_dict(sd); encoders.append(e.to(dev).eval())
    ck["dtype"] = dtype
    ck["embedder"] = SentenceEmbedder.load(legal_encoder or ck["legal_encoder"], embedder_device or dev)
    return tok, llm, encoders, ck


class LegalPredictor:
    """End-to-end v7 analysis of one judgment: InLegalBERT → Hyena encoders (whole judgment) → shortlist → Llama verifier → fusion."""

    def __init__(self, tokenizer, llm, encoders, ck):
        self.tok, self.llm, self.encoders, self.ck = tokenizer, llm, encoders, ck
        self.emb = ck["embedder"]
        self.dev = next(encoders[0].parameters()).device
        self.labels = ck["label_names"]
        self.yn = yes_no_ids(tokenizer)

    def _key(self, s):
        m = re.findall(r"\d+", s); return (int(m[0]) if m else 0, s)

    @torch.no_grad()
    def __call__(self, text: str, hide_cited: bool = True, max_tokens: int = 65536):
        ck, cfg = self.ck, self.ck["inference"]
        if hide_cited:
            text = hide_cited_sections(text)
        sents, flags = view_sentences(split_sentences(text), "full")
        doc = prepare_doc(sents, self.tok, max_tokens, flags)
        with torch.autocast("cuda", dtype=ck["dtype"], enabled=self.dev.type == "cuda"):
            p_enc, out = ensemble_probs(self.encoders, self.emb(doc.sentences).to(self.dev))
            cand_idx = shortlist(p_enc, cfg["k_min"], cfg["k_max"], cfg["min_p"])
            cands = sorted([self.labels[j] for j in cand_idx], key=self._key)
            passages = key_passages(doc, out["scores"], cfg["passage_tokens"], cfg["head_tokens"])
            evidence = {self.labels[j]: evidence_for(doc, out["attn"], j) for j in cand_idx}
            user = verify_user(passages, cands, ck["statute_desc"], evidence)
            acc = {c: 0.0 for c in cands}
            n_ord = max(1, cfg.get("verify_orders", 1))
            for k in range(n_ord):
                order = cands if k % 2 == 0 else cands[::-1]
                user_k = verify_user(passages, order, ck["statute_desc"], evidence)
                for c, p in zip(order, score_candidates(self.llm, self.tok, ck["verify_system"], user_k, order)):
                    acc[c] += p / n_ord
            p_llm = acc
        pe = {self.labels[j]: float(p_enc[j]) for j in range(len(self.labels))}
        final = {c: fuse(pe[c], p_llm.get(c), cfg["alpha"]) for c in self.labels}
        th = cfg.get("class_thresholds") or [cfg["threshold"]] * len(self.labels)
        secs = [c for j, c in enumerate(self.labels) if final[c] >= th[j]]
        if not secs and cfg.get("at_least_one"):
            secs = [max(final, key=final.get)]
        secs = sorted(secs, key=self._key)
        return {"sections": secs, "scores": {c: round(final[c], 3) for c in secs},
                "candidates": {c: {"encoder": round(pe[c], 3), "llm": round(p_llm[c], 3), "final": round(final[c], 3)} for c in cands},
                "evidence": evidence, "passages": passages,
                "doc_tokens": len(doc.ids), "sentences": len(doc.sentences),
                "llm_tokens": len(chat_ids(self.tok, ck["verify_system"], user)[0])}
