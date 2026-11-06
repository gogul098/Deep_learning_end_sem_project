"""
app.py · offline web page + API backend for lawyers and TUI.
  Runs fully offline on your NVIDIA RTX 3050.
  Proposes IPC sections, extracts evidence, explains court proceedings (CrPC/BNSS),
  provides interactive AI legal chat, and text embeddings via InLegalBERT.
"""
import argparse, html, json, os, sys
import gradio as gr
from local_model import load, read_document
from augment import format_conversational_briefing, conversational_qa
from procedural_kb import get_procedural_summary, format_proceedings_text

ap = argparse.ArgumentParser()
ap.add_argument("--lan", action="store_true", help="listen on the local network, not only this computer")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--per-section", action="store_true", help="per-section thresholds instead of one threshold")
a = ap.parse_args()

predict, info = load(per_section=a.per_section)
DESC = info["desc"]
TOK = info.get("tok")
LLM = info.get("llm")

# ── Text Embedding engine (lazy-loaded on first use) ──
_embed_engine = None

def _get_embed_engine():
    global _embed_engine
    if _embed_engine is None:
        from tui import EmbeddingEngine
        _embed_engine = EmbeddingEngine()
    return _embed_engine


def analyse_fn(pdf, text, hide_cited):
    """Core analysis function: runs Hyena + Llama and returns conversational chat briefing + tables."""
    if pdf is None and not (text or "").strip():
        return "Upload a PDF or paste the judgment text.", "", "No document loaded."
    
    doc = read_document(pdf if isinstance(pdf, str) else pdf.name) if pdf is not None else text
    r = predict(doc, hide_cited=hide_cited)
    
    # 1. Format conversational chat briefing using LLM & procedural KB
    chat_briefing = format_conversational_briefing(r, llm=LLM, tokenizer=TOK)

    # 2. Format HTML table for quick reference
    rows = []
    for s in r["sections"]:
        ev = html.escape((r["evidence"].get(s) or [""])[0][:300])
        rows.append(f"<tr><td><b>{html.escape(s)}</b></td><td>{html.escape(DESC.get(s, ''))}</td><td>{r['scores'][s]:.2f}</td>"
                    f"<td style='font-size:0.9em'>{ev}</td></tr>")
    table = ("<table><tr><th>Section</th><th>Offence</th><th>Score</th><th>Evidence sentence</th></tr>" + "".join(rows) + "</table>") if rows \
        else "<p>No section reached the threshold. See the candidates below.</p>"
    cand = "".join(f"<tr><td>{html.escape(c)}</td><td>{v['encoder']:.2f}</td><td>{v['llm']:.2f}</td><td>{v['final']:.2f}</td>"
                   f"<td>{'✓' if c in r['sections'] else ''}</td></tr>" for c, v in r["candidates"].items())
    stats = (f"{r['doc_tokens']:,} tokens · {r['sentences']} sentences read by the Hyena encoder · Llama read {r['llm_tokens']:,} tokens · "
             f"{r['seconds']} s" + (f" · peak VRAM {r['peak_vram_gb']} GB" if r["peak_vram_gb"] else ""))
    
    table_full = (table + "<h4>Candidates checked by Llama</h4><table><tr><th>Section</th><th>Encoder</th><th>Llama</th><th>Final</th><th></th></tr>"
                  + cand + f"</table><p style='color:gray'>{stats}</p><p><i>Decision support only: a qualified lawyer must verify every section.</i></p>")
    
    # Raw JSON string for TUI / API consumers
    raw_json = json.dumps({
        "sections": r["sections"],
        "scores": r["scores"],
        "candidates": r["candidates"],
        "evidence": r["evidence"],
        "passages": r["passages"],
        "doc_tokens": r["doc_tokens"],
        "sentences": r["sentences"],
        "seconds": r["seconds"],
        "peak_vram_gb": r.get("peak_vram_gb"),
        "chat_briefing": chat_briefing,
    })

    return chat_briefing, table_full, r["passages"], raw_json


def chat_fn(message, history, last_raw_json):
    """Conversational chat responding to questions about the loaded judgment or Indian criminal procedure."""
    if not message.strip():
        return history, ""
    
    context = None
    if last_raw_json and last_raw_json.strip():
        try:
            context = json.loads(last_raw_json)
        except Exception:
            pass

    reply = conversational_qa(message, context, llm=LLM, tokenizer=TOK)
    history = history or []
    history.append((message, reply))
    return history, ""


def embed_and_search(text, top_k):
    """Embed text using InLegalBERT and find the most similar IPC sections."""
    if not (text or "").strip():
        return "<p>Paste some legal text above.</p>", ""
    engine = _get_embed_engine()
    top_k = int(top_k)
    results = engine.find_similar_sections(text, top_k=top_k)
    vec_info = engine.embed_to_json(text)

    rows = []
    for i, r in enumerate(results, 1):
        sim = r["similarity"]
        bar_pct = int(sim * 100)
        color = "#22c55e" if sim >= 0.7 else ("#eab308" if sim >= 0.4 else "#ef4444")
        rows.append(
            f"<tr><td>{i}</td><td><b>{html.escape(r['section'])}</b></td>"
            f"<td style='color:{color};font-weight:bold'>{sim:.4f}</td>"
            f"<td>{html.escape(r['description'][:100])}</td>"
            f"<td><div style='background:linear-gradient(90deg,{color} {bar_pct}%,#333 {bar_pct}%);height:14px;border-radius:3px;width:120px'></div></td></tr>"
        )

    table_html = (
        f"<p style='color:gray'>Vector: {vec_info['dim']} dimensions (InLegalBERT) · "
        f"Input: \"{html.escape(vec_info['text'])}\"</p>"
        "<table><tr><th>#</th><th>Section</th><th>Similarity</th><th>Description</th><th>Score</th></tr>"
        + "".join(rows) + "</table>"
        "<p style='color:gray'><i>Similarity is cosine distance in InLegalBERT embedding space.</i></p>"
    )

    vec_preview = ", ".join(f"{v:.4f}" for v in vec_info["vector"][:20])
    vec_text = f"Vector ({vec_info['dim']} dims, showing first 20):\n[{vec_preview}, ...]"

    return table_html, vec_text


def embed_compare(text_a, text_b):
    """Compute cosine similarity between two text passages."""
    if not (text_a or "").strip() or not (text_b or "").strip():
        return "Enter text in both fields."
    engine = _get_embed_engine()
    sim = engine.cosine_similarity(text_a, text_b)
    color = "#22c55e" if sim >= 0.7 else ("#eab308" if sim >= 0.4 else "#ef4444")
    bar_pct = int(abs(sim) * 100)
    return (
        f"<div style='text-align:center;padding:20px'>"
        f"<h2 style='color:{color}'>Cosine Similarity: {sim:.4f}</h2>"
        f"<div style='background:linear-gradient(90deg,{color} {bar_pct}%,#333 {bar_pct}%);height:24px;border-radius:6px;max-width:400px;margin:auto'></div>"
        f"<p style='color:gray;margin-top:10px'>{'Very similar' if sim >= 0.8 else 'Similar' if sim >= 0.6 else 'Somewhat similar' if sim >= 0.4 else 'Different'}</p>"
        f"</div>"
    )


with gr.Blocks(title="Legal IPC Assistant (offline)") as demo:
    gr.Markdown(f"## ⚖️ IPC Section Assistant · runs offline on {info['device']}\n"
                "Upload a judgment PDF or paste text. Hyena proposes sections, Llama-3.2-3B verifies evidence, "
                "and an AI advocate explains sections, clauses, and court proceedings (CrPC/BNSS).")

    # State for holding raw analysis JSON
    state_raw_json = gr.State("")

    with gr.Tabs():
        # ── Tab 1: Conversational Analysis & Briefing ──
        with gr.TabItem("💬 Legal Briefing & Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    pdf = gr.File(label="Judgment (PDF or TXT)", file_types=[".pdf", ".txt"])
                    text = gr.Textbox(label="…or paste judgment text", lines=8)
                    hide = gr.Checkbox(value=True, label="Hide cited sections (infer from facts)")
                    go = gr.Button("⚡ Analyse Judgment", variant="primary")
                with gr.Column(scale=2):
                    briefing_md = gr.Markdown(label="Conversational AI Briefing")
                    with gr.Accordion("📊 Candidate Tables & Metrics", open=False):
                        table_html = gr.HTML()
                    with gr.Accordion("📜 Extracted Key Passages", open=False):
                        passages = gr.Textbox(lines=6)

            go.click(
                analyse_fn,
                inputs=[pdf, text, hide],
                outputs=[briefing_md, table_html, passages, state_raw_json],
                api_name="analyse"
            )

        # ── Tab 2: Interactive Advocate Chat ──
        with gr.TabItem("🏛️ Chat with Legal Advocate"):
            gr.Markdown("### Interactive Legal Q&A on Loaded Judgment\nAsk about trial stages, statutory clauses, bail strategy, or evidence.")
            chatbot = gr.Chatbot(height=450)
            with gr.Row():
                chat_in = gr.Textbox(placeholder="Ask anything (e.g. 'What are the grounds for bail?', 'Explain Section 34 role')...", lines=1, scale=8)
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            chat_btn.click(chat_fn, inputs=[chat_in, chatbot, state_raw_json], outputs=[chatbot, chat_in], api_name="chat")
            chat_in.submit(chat_fn, inputs=[chat_in, chatbot, state_raw_json], outputs=[chatbot, chat_in])

        # ── Tab 3: Text Embedding ──
        with gr.TabItem("🔤 Text Embedding"):
            gr.Markdown("### InLegalBERT Semantic Embeddings\nFind IPC sections whose statutory descriptions are closest to your input.")
            with gr.Row():
                with gr.Column(scale=1):
                    embed_text = gr.Textbox(label="Legal text to embed", lines=6,
                                            placeholder="e.g. The accused inflicted severe fatal blows with a sharp weapon...")
                    embed_topk = gr.Slider(minimum=5, maximum=30, value=10, step=1, label="Top-K results")
                    embed_go = gr.Button("🔍 Find Similar Sections", variant="primary")
                with gr.Column(scale=2):
                    embed_result = gr.HTML(label="Similar IPC Sections")
                    embed_vec = gr.Textbox(label="Embedding vector (preview)", lines=3)
            embed_go.click(embed_and_search, [embed_text, embed_topk], [embed_result, embed_vec], api_name="embed")

            gr.Markdown("---\n### Semantic Text Similarity Comparison")
            with gr.Row():
                cmp_a = gr.Textbox(label="Text A", lines=3, placeholder="First passage...")
                cmp_b = gr.Textbox(label="Text B", lines=3, placeholder="Second passage...")
            cmp_btn = gr.Button("📐 Compare Similarity", variant="secondary")
            cmp_result = gr.HTML()
            cmp_btn.click(embed_compare, [cmp_a, cmp_b], [cmp_result], api_name="compare")

demo.queue(max_size=8).launch(server_name="0.0.0.0" if a.lan else "127.0.0.1", server_port=a.port, share=False)
