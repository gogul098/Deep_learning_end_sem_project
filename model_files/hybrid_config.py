# ═════════════ hybrid_config.py · v7 · Hyena encoder → shortlist → Llama verifier → fusion · edit settings here ═════════════
BASE_MODEL    = "unsloth/Llama-3.2-3B-Instruct"  # ungated mirror of meta-llama/Llama-3.2-3B-Instruct (Llama 3.2 licence applies)
LEGAL_ENCODER = "law-ai/InLegalBERT"             # pretrained on Indian legal text (MIT licence), frozen
WORK_DIR     = "/kaggle/working"
DATA_DIR     = f"{WORK_DIR}/data_v4"              # prepared datasets (written by prepare_data.py)
LEGACY_DATA_DIR = f"{WORK_DIR}/data"              # v3 data, upgraded to v4 if present (saves re-downloading the corpus)
CKPT_DIR     = f"{WORK_DIR}/checkpoints"          # local checkpoints (also pushed to the Hub)
CACHE_DIR    = "/tmp/emb_cache"                   # InLegalBERT sentence vectors, computed once (kept out of the notebook output; rebuilt if needed)
OUT_DIR      = f"{WORK_DIR}/hybrid_legal"         # final model + results
SEED         = 42

# ── checkpoint sync: Kaggle → private Hugging Face repo → your laptop (run sync_checkpoints.py locally) ──
PUSH_TO_HUB  = True                               # needs a WRITE token in the Kaggle secret HF_TOKEN
REPO_NAME    = "hyena-llama-legal-v7"             # checkpoint repo name on your Hugging Face account
HF_REPO      = ""                                 # "" = <your-hf-username>/<REPO_NAME> (checkpoints, private)
HF_DATA_REPO = ""                                 # "" = <your-hf-username>/hyena-llama-legal (prepared data; v4 lives in data_v4/)
CKPT_EVERY_UPDATES = 20                           # Stage B checkpoint every ~20 optimizer updates

# ── data ──
DATA_SOURCE  = "aws"                              # "aws" = Indian Supreme Court judgments (S3) | "iltur" = IL-TUR LSI
AWS_YEARS    = list(range(2005, 2025))
AWS_CACHE    = f"{WORK_DIR}/sc_corpus.parquet"
TRAIN_UNTIL, DEV_UNTIL = 2020, 2022               # temporal split: train ≤2020 · dev 2021–22 · test ≥2023
N_LABEL_CLASSES, MIN_LABEL_FREQ = 100, 15
FACTS_PROB   = 0.5                                # training: share of documents shown WITHOUT the sentences that cite sections
MAX_DOC      = 16384                              # tokens of a judgment read by InLegalBERT + Hyena (train and test)
TRAIN_LIMIT  = 0                                  # 0 = all training judgments (a smaller number makes a quick trial run)

# ── 1 · Hyena encoder (reads the whole judgment) ──
ENC_DIM, ENC_LAYERS, N_MODES = 512, 4, 8
BAND_CUTOFF  = 0.25
LABEL_MIX    = True                               # label-correlation layer: the 100 sections attend to each other
ENC_EPOCHS   = 8                                  # passes over the training judgments (best dev epoch is kept)
ENC_LR       = 3e-4
ENTROPY_FLOOR = 0.3

# ── 2 · shortlist + 3 · Llama verifier ──
K_MIN, K_MAX, MIN_P = 5, 10, 0.05                 # candidates per judgment: top-10 (at least 5; drop those below 5%)
PASSAGE_TOKENS, HEAD_TOKENS = 1500, 400           # what Llama reads: opening (400 tokens) + key passages (up to 1500 in total)
STAGE_B_DOCS = 2400                               # verifier fine-tuning examples (totals across GPUs; ~2 h on 2× T4)
GENERATE_SHARE = 0.2                              # share of Stage B examples in plain "list the sections" mode (baseline row)
MAX_NEW_TOKENS = 200
EFFECTIVE_BATCH = 8                               # examples per optimizer update (split across GPUs)

# ── accuracy boosters ──
VERIFY_ORDERS = 2                                 # the verifier reads the shortlist in 2 orders and averages (no position bias)
CLASS_THRESHOLDS = True                           # per-section thresholds tuned on dev (shrunk toward the global one)
OBJECTIVE    = "micro"                            # calibration maximises "micro" or "macro" F1 (the IL-TUR leaderboard uses macro)

# ── evaluation ──
VIEWS   = ["full", "facts"]                       # IL-TUR: ["full"] (its text is already facts only)
N_DEV_ENC = 400                                   # dev judgments for the encoders' best epoch + per-section thresholds
ENC_TEST_ALL = False                              # also score the encoder ensemble on the WHOLE test split (benchmark comparison)
TOP_LABELS_REPORT = 20                            # also report F1 on the 20 most frequent sections (smaller label space)
EVAL_LIST_ROWS = True                             # the two slow "Llama lists sections" rows
N_CALIB = 200                                     # dev judgments used to tune the fusion (α, threshold)
N_EVAL  = 150                                     # test judgments (each in 2 views: full text, facts-only)
BENCH_LENGTHS = [2048, 4096, 8192, 16384, 32768]

VERIFY_SYSTEM = ("You are a legal assistant for Indian criminal law. You are given the key passages of a Supreme Court judgment "
                 "and a shortlist of candidate Indian Penal Code (IPC) sections proposed by a document classifier. Some candidates "
                 "are wrong. Decide for each candidate whether it applies to the accused in this case.")
GENERATE_SYSTEM = ("You are a legal assistant for Indian criminal law. Read the key passages of the judgment and list every "
                   "Indian Penal Code (IPC) section that applies, one per line, as 'Section <number>'.")
