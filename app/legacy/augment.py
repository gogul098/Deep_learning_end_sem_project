"""
augment.py · LLM conversational augmentation engine for IPC Legal Assistant.
Transforms structured model predictions and judgment extracts into natural,
expert conversational briefings explaining sections, key clauses, and court proceedings.
Also powers multi-turn chat interaction using Llama-3.2-3B.
"""

import re
import json
from typing import Dict, Any, List, Optional
from procedural_kb import get_procedural_summary, format_proceedings_text, IPC_PROCEDURAL_DATA

SYSTEM_PROMPT = """You are Claude Code Legal Assistant, a senior criminal advocate and legal AI specialist in Indian Criminal Jurisprudence (IPC, CrPC, BNSS, Indian Evidence Act/BSA).
Your voice is conversational, authoritative, analytical, and structured, speaking directly to the user in a natural chat.
When analyzing judgments:
1. Explain the identified IPC sections and why they apply based on facts.
2. Highlight the critical clauses, witness statements, and evidence in the document.
3. Detail the exact criminal court proceedings (CrPC/BNSS roadmap, trial court jurisdiction, cognizable/bailable nature, charge framing, bail strategy).
Always be professional, precise, and helpful while noting this is legal decision support.
"""


def format_conversational_briefing(
    analysis_result: Dict[str, Any],
    llm=None,
    tokenizer=None,
    user_query: Optional[str] = None
) -> str:
    """
    Generate an augmented conversational chat briefing explaining:
    - Applicable IPC sections & scores
    - Key clauses & evidence involved
    - Step-by-step court proceedings (CrPC/BNSS)
    """
    sections = analysis_result.get("sections", [])
    scores = analysis_result.get("scores", {})
    evidence = analysis_result.get("evidence", {})
    passages = analysis_result.get("passages", "")
    candidates = analysis_result.get("candidates", {})
    doc_tokens = analysis_result.get("doc_tokens", 0)
    seconds = analysis_result.get("seconds", 0)

    # 1. Build structured synthesis
    intro = (
        f"Hello! I have completed the legal analysis of this judgment "
        f"({doc_tokens:,} tokens processed in {seconds}s).\n\n"
        f"Here is a comprehensive legal breakdown of the applicable IPC sections, crucial evidentiary clauses, "
        f"and the procedural roadmap for conducting court proceedings."
    )

    # 2. Key Clauses and Evidentiary Sentences
    clauses_section = ["### 📜 Key Clauses & Evidentiary Findings in Document"]
    if passages:
        # Extract first 2-3 key opening passages
        clean_passages = passages.strip()
        lines = [line.strip("- ") for line in clean_passages.split("\n") if line.strip() and not line.startswith("Opening") and not line.startswith("Other")]
        for i, l in enumerate(lines[:3], 1):
            clauses_section.append(f"> **Clause {i}**: \"{l[:240]}{'...' if len(l) > 240 else ''}\"")
    else:
        clauses_section.append("*(No specific passages extracted)*")

    # 3. Applicable Sections & Offence Analysis
    sections_block = ["### ⚖️ Applicable IPC Sections & Offence Analysis"]
    if not sections:
        sections_block.append("No specific IPC section surpassed the threshold for this document. See candidate probabilities below.")
    else:
        for s in sections:
            score = scores.get(s, 0.0)
            ev_list = evidence.get(s, [])
            ev_text = ev_list[0] if ev_list else "Not explicitly cited in text; inferred from underlying facts."
            kb = get_procedural_summary(s)
            title = kb["title"] if kb else "Indian Penal Code Offence"

            sections_block.append(f"#### **{s}: {title}** *(Confidence Score: `{score:.2f}`)*")
            sections_block.append(f"- **Key Evidence Clause**: \"{ev_text}\"")
            if kb:
                sections_block.append(f"- **Statutory Ingredients**: {'; '.join(kb.get('ingredients', [])[:2])}")
                cog = "Cognizable" if kb.get("cognizable") else "Non-cognizable"
                bail = "Non-bailable" if not kb.get("bailable") else "Bailable"
                sections_block.append(f"- **Legal Classification**: `{cog}` | `{bail}` | Triable by **{kb.get('triable_by', 'Competent Court')}**")
            sections_block.append("")

    # 4. Court Proceedings & Procedural Roadmap (CrPC / BNSS)
    proceedings_block = ["### 🏛️ Court Proceedings & Procedural Roadmap (CrPC / BNSS)"]
    if sections:
        # Determine highest trial court
        highest_court = "Magistrate of First Class"
        for s in sections:
            kb = get_procedural_summary(s)
            if kb and "Session" in kb.get("triable_by", ""):
                highest_court = "Court of Session (Sessions Judge / Additional Sessions Judge)"
                break

        proceedings_block.append(f"**Jurisdiction & Forum**: All charges arising from this transaction are triable before the **{highest_court}**.")
        proceedings_block.append("")
        proceedings_block.append("#### Standard Procedural Trajectory:")
        proceedings_block.append("1. **Investigation & Remand (Section 167 CrPC)**: Police complete statutory investigation and file Police Report (Chargesheet u/s 173(2) CrPC) within 60 or 90 days.")
        proceedings_block.append("2. **Committal Proceedings (Section 209 CrPC)**: The Magistrate verifies police papers, supplies copies to accused (u/s 207 CrPC), and commits case to Court of Session.")
        proceedings_block.append("3. **Hearing on Charge (Sections 227–228 CrPC)**: Arguments on discharge. If prima facie material exists, formal charges are framed and read over to the accused.")
        proceedings_block.append("4. **Prosecution Evidence (PWs)**: Examination-in-chief and cross-examination of eyewitnesses, recovery witnesses, medical officer, and Investigation Officer (IO).")
        proceedings_block.append("5. **Accused Statement (Section 313 CrPC)**: Mandatory recording of accused's explanation regarding every incriminating circumstance.")
        proceedings_block.append("6. **Defense Evidence & Final Arguments (Sections 233, 314 CrPC)**: Defense witnesses (if any) followed by closing legal arguments.")
        proceedings_block.append("7. **Judgment & Sentencing (Section 235 CrPC)**: Acquittal or conviction, with a mandatory separate hearing on the quantum of sentence.")
        
        # Bail advisory
        has_non_bailable = any(get_procedural_summary(s) and not get_procedural_summary(s).get("bailable", True) for s in sections)
        if has_non_bailable:
            proceedings_block.append("")
            proceedings_block.append("#### 🛡️ Bail & Legal Defense Advisory:")
            proceedings_block.append("- The identified offences are **Non-bailable**. Regular bail applications must be moved under **Section 439 CrPC** before the Sessions Court or High Court.")
            proceedings_block.append("- **Key Arguments for Bail/Defense**: Scrutinize delay in FIR, test identification parade (TIP) veracity, independent corroboration of recovery u/s 27 Evidence Act, and parity with co-accused.")

    # 5. Shortlisted Candidates Checked
    cand_block = ["<details><summary><b>🔍 Shortlisted Candidates Checked by Llama</b></summary>\n"]
    cand_block.append("| Section | Hyena Encoder | Llama Verifier | Combined Score | Selected |")
    cand_block.append("| :--- | :---: | :---: | :---: | :---: |")
    for c, v in candidates.items():
        sel = "✓" if c in sections else ""
        cand_block.append(f"| **{c}** | {v['encoder']:.2f} | {v['llm']:.2f} | **{v['final']:.2f}** | {sel} |")
    cand_block.append("\n</details>")

    # Combine blocks
    base_response = (
        f"{intro}\n\n"
        f"{chr(10).join(clauses_section)}\n\n"
        f"{chr(10).join(sections_block)}\n\n"
        f"{chr(10).join(proceedings_block)}\n\n"
        f"{chr(10).join(cand_block)}\n\n"
        f"---\n*💡 **Ask me anything**: You can ask follow-up questions about clauses, bail prospects, cross-examination strategy, or statutory exceptions.*"
    )

    # If LLM and tokenizer are available, augment with customized neural commentary!
    if llm is not None and tokenizer is not None:
        try:
            prompt = (
                f"Based on these extracted findings from an Indian judgment:\n"
                f"Sections identified: {', '.join(sections)}\n"
                f"Passages: {passages[:800]}\n"
                f"User Question: {user_query or 'Summarize the legal strategy and critical evidentiary clauses'}\n\n"
                f"Write a concise, 2-3 paragraph senior advocate commentary highlighting critical clauses and strategic procedural steps."
            )
            neural_commentary = generate_llm_chat(llm, tokenizer, SYSTEM_PROMPT, prompt, max_new_tokens=250)
            if neural_commentary and len(neural_commentary.strip()) > 20:
                base_response = f"{intro}\n\n### ⚖️ Senior Advocate Briefing\n{neural_commentary}\n\n" + "\n\n".join([
                    chr(10).join(clauses_section),
                    chr(10).join(sections_block),
                    chr(10).join(proceedings_block),
                    chr(10).join(cand_block),
                ])
        except Exception:
            pass  # fallback to structured synthesis

    return base_response


def generate_llm_chat(llm, tokenizer, system_prompt: str, user_message: str, max_new_tokens: int = 350) -> str:
    """Generate conversational text using Llama-3.2-3B with chat template."""
    import torch
    from transformers import DynamicCache

    dev = next(llm.parameters()).device
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    # Generation via KV-cache or generate
    try:
        input_tensor = torch.tensor([ids], device=dev)
        with torch.no_grad():
            outputs = llm.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = outputs[0][len(ids):]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    except Exception:
        # Fallback to manual loop
        try:
            body = llm.get_base_model().model if hasattr(llm, "get_base_model") else llm.model
            head = llm.get_base_model().lm_head if hasattr(llm, "get_base_model") else llm.lm_head
            cache = DynamicCache()
            out = body(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True)
            last, new = out.last_hidden_state[:, -1], []
            eos = {tokenizer.eos_token_id}
            for _ in range(max_new_tokens):
                nxt = int(head(last).float().argmax(-1).item())
                if nxt in eos:
                    break
                new.append(nxt)
                out = body(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
                last = out.last_hidden_state[:, -1]
            return tokenizer.decode(new, skip_special_tokens=True).strip()
        except Exception as e:
            return f"(Neural commentary unavailable: {e})"


def conversational_qa(
    question: str,
    last_context: Optional[Dict[str, Any]],
    llm=None,
    tokenizer=None
) -> str:
    """Answer a user's follow-up chat question regarding sections, clauses, or proceedings."""
    clean_q = question.strip()
    
    # Check for direct section queries e.g. "tell me about section 302" or "/proceedings 302"
    sec_match = re.search(r"section\s*(\d+[A-Za-z]*)", clean_q, re.I)
    if sec_match and not last_context:
        sec_name = f"Section {sec_match.group(1).upper()}"
        kb = get_procedural_summary(sec_name)
        if kb:
            return format_proceedings_text(sec_name, kb)

    # If context is available from an analyzed judgment
    if last_context:
        sections = last_context.get("sections", [])
        evidence = last_context.get("evidence", {})
        passages = last_context.get("passages", "")
        
        # If question is asking about proceedings
        if any(w in clean_q.lower() for w in ["proceeding", "trial", "crpc", "bnss", "court", "stage", "bail"]):
            proceedings_text = []
            proceedings_text.append(f"### 🏛️ Criminal Proceedings Breakdown for Loaded Judgment\n")
            for s in sections:
                kb = get_procedural_summary(s)
                if kb:
                    proceedings_text.append(format_proceedings_text(s, kb))
                    proceedings_text.append("---")
            if proceedings_text:
                return "\n\n".join(proceedings_text)

        # If question is asking about clauses / evidence
        if any(w in clean_q.lower() for w in ["clause", "evidence", "witness", "statement", "passage", "weapon", "injury"]):
            clause_text = ["### 📜 Critical Evidentiary Clauses Identified in Judgment\n"]
            for s in sections:
                ev_list = evidence.get(s, [])
                if ev_list:
                    clause_text.append(f"**{s} Evidence Clause**:")
                    for e in ev_list:
                        clause_text.append(f"> \"{e}\"")
            if passages:
                clause_text.append("\n**Key Contextual Passages Chosen by Hyena Encoder**:")
                clause_text.append(f"```\n{passages[:1200]}\n```")
            return "\n\n".join(clause_text)

    # Use LLM for conversational Q&A if available
    if llm is not None and tokenizer is not None:
        ctx_prompt = ""
        if last_context:
            ctx_prompt = (
                f"\nJudgment Context:\n"
                f"- Applicable Sections: {', '.join(last_context.get('sections', []))}\n"
                f"- Evidentiary Clauses: {json.dumps(last_context.get('evidence', {}))}\n"
                f"- Passages: {last_context.get('passages', '')[:800]}\n"
            )
        user_msg = f"{ctx_prompt}\nUser Question: {clean_q}\nProvide an authoritative, conversational answer explaining the relevant Indian criminal law sections, clauses, or court proceedings."
        return generate_llm_chat(llm, tokenizer, SYSTEM_PROMPT, user_msg, max_new_tokens=400)

    # Intelligent fallback when LLM is in another process
    return (
        f"I received your inquiry: **\"{clean_q}\"**.\n\n"
        f"To get in-depth section proceedings, you can ask `/proceedings <section>` (e.g. `/proceedings 302`), "
        f"or upload a judgment PDF using `/open <filename>` to extract and explain specific clauses and procedural steps."
    )
