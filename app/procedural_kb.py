"""
procedural_kb.py · Indian Criminal Law Procedural Knowledge Base (CrPC / BNSS & IPC).
Provides statutory classifications, court jurisdiction, bail provisions, and procedural steps
for Indian Penal Code offences to augment LLM legal reasoning and chat briefings.
"""

from typing import Dict, Any, Optional, List

# Procedural mapping for major IPC offences
IPC_PROCEDURAL_DATA: Dict[str, Dict[str, Any]] = {
    "Section 302": {
        "title": "Punishment for Murder",
        "punishment": "Death, or imprisonment for life, and liable to fine",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "ingredients": [
            "Causing death of a human being",
            "Act done with the intention of causing death, or causing bodily injury sufficient in the ordinary course of nature to cause death, or knowing the act is so imminently dangerous that it must in all probability cause death",
            "Absence of General Exceptions (Section 300 Exceptions 1-5: Grave & sudden provocation, private defense, public servant, sudden fight, consent)",
        ],
        "proceedings": [
            "Investigation by Police Officer (u/s 156/157 CrPC)",
            "Arrest of accused; remand under Section 167 CrPC (90 days maximum for filing chargesheet)",
            "Recording of dying declaration (if applicable, u/s 32 IEA / BSA) and Section 161/164 witness statements",
            "Seizure of weapon/articles under Section 27 Evidence Act (recovery panchnama)",
            "Chargesheet filed u/s 173(2) CrPC before Judicial Magistrate of First Class",
            "Committal proceedings u/s 209 CrPC to the Court of Session",
            "Framing of Charges u/s 228 CrPC by Sessions Judge",
            "Trial: Examination of Prosecution Witnesses (PWs), Doctor (post-mortem), IO",
            "Statement of Accused u/s 313 CrPC (mandatory explanation of incriminating facts)",
            "Final Arguments and Judgment u/s 235 CrPC; Separate hearing on sentence",
        ],
        "bail_guidance": "Regular bail u/s 439 CrPC from Sessions Court or High Court. Rigorous scrutiny: prima facie case, gravity of offence, chance of tampering with witnesses, parity.",
    },
    "Section 304": {
        "title": "Punishment for Culpable Homicide not amounting to Murder",
        "punishment": "Part I: Life imprisonment or up to 10 years + fine; Part II: Up to 10 years or fine or both",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "ingredients": [
            "Causing death with intention (Part I) or with knowledge that act is likely to cause death (Part II)",
            "Case falls under one of the 5 exceptions to Section 300 IPC (e.g. sudden fight, provocation, sudden quarrel)",
        ],
        "proceedings": [
            "Committal to Court of Session u/s 209 CrPC",
            "Focus of trial: distinguishing knowledge from intention; provocation or absence of premeditation",
            "Doctor's testimony on nature of single vs multiple blows",
        ],
        "bail_guidance": "Bail can be sought u/s 439 CrPC, especially where Part II applies (knowledge without intention) or single-blow scenario.",
    },
    "Section 304B": {
        "title": "Dowry Death",
        "punishment": "Imprisonment for not less than 7 years, but which may extend to life",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "ingredients": [
            "Death of woman occurs within 7 years of marriage",
            "Death caused by burns, bodily injury, or under abnormal circumstances",
            "Shown that 'soon before her death' she was subjected to cruelty or harassment by husband or his relatives for dowry",
        ],
        "proceedings": [
            "Inquest by Executive Magistrate u/s 174/176 CrPC (mandatory for death within 7 years of marriage)",
            "Statutory presumption of dowry death under Section 113B Indian Evidence Act shifts initial burden",
            "Committal to Court of Session; rigorous witness examination of maternal family",
        ],
        "bail_guidance": "Strict scrutiny for regular bail; applicants must establish absence of harassment 'soon before death'.",
    },
    "Section 307": {
        "title": "Attempt to Murder",
        "punishment": "Imprisonment up to 10 years and fine; if hurt is caused, up to life imprisonment",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "ingredients": [
            "Act done with such intention or knowledge and under such circumstances that if death was caused, offender would be guilty of murder",
            "Execution of an overt act towards the commission of murder",
        ],
        "proceedings": [
            "FIR registration, medical examination of victim (MLC / injury report: grievous vs simple injury)",
            "Recovery of weapon used",
            "Trial before Court of Session",
        ],
        "bail_guidance": "Bail considerations hinge heavily on the injury report (vital vs non-vital organ) and discharge from hospital.",
    },
    "Section 376": {
        "title": "Punishment for Rape",
        "punishment": "Rigorous imprisonment not less than 10 years, which may extend to life, and fine",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session (presided over by a woman Judge where practicable)",
        "compoundable": False,
        "ingredients": [
            "Sexual acts specified in clauses (a)-(d) of Section 375 without consent or against will",
            "Absence of free consent (Section 375 Explanation 2)",
        ],
        "proceedings": [
            "Mandatory medical examination of victim u/s 164A CrPC within 24 hours",
            "Recording of statement by woman Judicial Magistrate u/s 164(5A) CrPC",
            "In-camera trial u/s 327(2) CrPC; bar on revealing victim's identity (Section 228A IPC)",
            "Presumption as to absence of consent under Section 114A Evidence Act",
        ],
        "bail_guidance": "Stringent conditions under Section 439 CrPC; victim notice and hearing is mandatory prior to grant of bail.",
    },
    "Section 395": {
        "title": "Punishment for Dacoity",
        "punishment": "Imprisonment for life, or rigorous imprisonment up to 10 years, and fine",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "ingredients": [
            "Conjoint commission of robbery by 5 or more persons (Section 391 IPC)",
            "Dishonest taking of property with wrongful restraint or putting person in fear of instant hurt/death",
        ],
        "proceedings": [
            "Test Identification Parade (TIP) of accused persons under supervision of Magistrate",
            "Recovery of looted property under Section 27 Evidence Act",
            "Trial before Sessions Court; proving presence of minimum 5 participants",
        ],
        "bail_guidance": "Non-bailable. Scrutiny focuses on whether accused was identified in TIP and recoveries made.",
    },
    "Section 397": {
        "title": "Robbery or Dacoity, with Attempt to Cause Death or Grievous Hurt",
        "punishment": "Rigorous imprisonment not less than 7 years",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "ingredients": [
            "At the time of committing robbery or dacoity, offender uses any deadly weapon, or causes grievous hurt, or attempts to cause death/grievous hurt",
            "Applies strictly to the individual offender who actually uses the weapon (individual liability, not vicarious)",
        ],
        "proceedings": [
            "Proving the exact recovery and brandishing/use of the 'deadly weapon'",
            "Trial by Court of Session with mandatory minimum sentence of 7 years upon conviction",
        ],
        "bail_guidance": "High threshold for bail due to statutory minimum sentence.",
    },
    "Section 420": {
        "title": "Cheating and Dishonestly Inducing Delivery of Property",
        "punishment": "Imprisonment up to 7 years and fine",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Magistrate of the First Class",
        "compoundable": True,  # with permission of court
        "ingredients": [
            "Deception of any person",
            "Fraudulently or dishonestly inducing the person deceived to deliver property",
            "Mens rea (fraudulent intent) existing at the inception of the transaction",
        ],
        "proceedings": [
            "Police investigation or filing of private complaint u/s 200 CrPC / Section 156(3) CrPC",
            "Forensic analysis of financial documents, contracts, bank statements",
            "Trial before Judicial Magistrate First Class (JMFC)",
            "Mediation / Compounding possible with leave of court u/s 320(2) CrPC",
        ],
        "bail_guidance": "Notice u/s 41A CrPC (Arnesh Kumar guidelines) before arrest. Anticipatory bail u/s 438 or regular bail readily considered if civil dispute or willing to deposit disputed amount.",
    },
    "Section 498A": {
        "title": "Husband or Relative of Husband of a Woman Subjecting Her to Cruelty",
        "punishment": "Imprisonment up to 3 years and fine",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Magistrate of the First Class",
        "compoundable": False,
        "ingredients": [
            "Woman subjected to cruelty by husband or his relative",
            "Cruelty defined: (a) Wilful conduct likely to drive her to suicide or cause grave injury; (b) Harassment to coerce unlawful property demand",
        ],
        "proceedings": [
            "Reference to Family Welfare Committee / Mediation Cell prior to arrest (Social Welfare guidelines)",
            "Strict compliance with Section 41A CrPC notice against automatic arrest",
            "Trial before JMFC; proving specific overt acts against each named relative",
        ],
        "bail_guidance": "Anticipatory bail u/s 438 CrPC widely granted unless specific allegations of physical violence exist. Quashing u/s 482 CrPC common for distant relatives.",
    },
    "Section 34": {
        "title": "Acts Done by Several Persons in Furtherance of Common Intention",
        "punishment": "Rule of joint/vicarious liability (punished according to substantive offence committed)",
        "cognizable": True,
        "bailable": False,  # dependent on main offence
        "triable_by": "Court having jurisdiction over substantive offence",
        "compoundable": False,
        "ingredients": [
            "Criminal act done by two or more persons",
            "Act must be done in furtherance of common intention of all",
            "Prior meeting of minds (pre-arranged plan) or intention formed on the spot",
            "Active physical participation or presence facilitating the crime",
        ],
        "proceedings": [
            "Section 34 is charged alongside substantive offences (e.g. 302/34, 392/34)",
            "Prosecution must establish shared intention and participation of each accused",
            "If individual role is distinct (e.g. standing guard vs stabbing), both are equally liable if common intention proved",
        ],
        "bail_guidance": "Bail can be argued on ground of parity or absence of overt fatal act if common intention is weak.",
    },
    "Section 120B": {
        "title": "Punishment of Criminal Conspiracy",
        "punishment": "If conspiracy to commit offence punishable with death/life/2+ yrs: same as abetment; otherwise up to 6 months or fine",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court having jurisdiction over substantive offence",
        "compoundable": False,
        "ingredients": [
            "Agreement between two or more persons to do or cause to be done an illegal act",
            "Direct evidence is rare; inferred from surrounding circumstantial evidence and conduct",
        ],
        "proceedings": [
            "Evidentiary reliance on Section 10 Indian Evidence Act (admissibility of statements of co-conspirators)",
            "Proof of meeting of minds, call detail records (CDR), money trails, common design",
        ],
        "bail_guidance": "Challenged on lack of prima facie agreement or lack of independent corroboration.",
    },
    "Section 149": {
        "title": "Every Member of Unlawful Assembly Guilty of Offence Committed in Prosecution of Common Object",
        "punishment": "Vicarious liability (punished as principal offender for offences in prosecution of common object)",
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court having jurisdiction over substantive offence",
        "compoundable": False,
        "ingredients": [
            "Assembly of five or more persons (Section 141 IPC)",
            "Commission of an offence by any member of that assembly",
            "Offence committed in prosecution of common object, or such as members knew to be likely to be committed",
        ],
        "proceedings": [
            "Proving identity and membership of at least 5 persons",
            "Proving common object of assembly (not necessarily prior meeting of minds like Section 34)",
        ],
        "bail_guidance": "Accused who did not commit overt violence can argue for bail citing passive presence or omnibus allegations.",
    },
}


def get_procedural_summary(section: str) -> Optional[Dict[str, Any]]:
    """Return procedural data for an IPC section if available."""
    clean_sec = section.strip()
    if clean_sec in IPC_PROCEDURAL_DATA:
        return IPC_PROCEDURAL_DATA[clean_sec]
    # Try finding without "Section "
    for k, v in IPC_PROCEDURAL_DATA.items():
        if clean_sec.lower() in k.lower() or k.lower() in clean_sec.lower():
            return v
    return None


def format_proceedings_text(section: str, data: Dict[str, Any]) -> str:
    """Format procedural roadmap for display."""
    cognizable_str = "Cognizable (Police can arrest without warrant)" if data.get("cognizable") else "Non-cognizable"
    bailable_str = "Non-bailable (Bail is discretion of Court)" if not data.get("bailable") else "Bailable (Bail as matter of right)"
    compoundable_str = "Compoundable" if data.get("compoundable") else "Non-compoundable"

    lines = [
        f"### 🏛️ {section}: {data.get('title', '')}",
        f"- **Statutory Punishment**: {data.get('punishment', 'As prescribed by law')}",
        f"- **Classification**: {cognizable_str} | {bailable_str} | {compoundable_str}",
        f"- **Trial Jurisdiction**: **{data.get('triable_by', 'Competent Court')}**",
        "",
        "#### 📌 Statutory Ingredients & Clauses to Prove:",
    ]
    for ing in data.get("ingredients", []):
        lines.append(f"  • {ing}")

    lines.append("")
    lines.append("#### ⚖️ Step-by-Step Criminal Proceedings Roadmap (CrPC / BNSS):")
    for step in data.get("proceedings", []):
        lines.append(f"  1. {step}")

    if data.get("bail_guidance"):
        lines.append("")
        lines.append(f"#### 🛡️ Bail & Defense Strategy:\n{data['bail_guidance']}")

    return "\n".join(lines)
