"""
label_checkpoint.py (from prob_reas checkpoint.py) - off-machine checkpoints for long labelling runs (Kaggle sessions can die at any moment).

The labels file is copied to a PRIVATE Hugging Face dataset repo every N questions, at the
end, and (best effort) when the process is told to stop. A new run first downloads that
copy, merges it with any local file, and continues from there - so at most N questions
are ever lost, whatever happens to the machine.

Needs a Hugging Face token with WRITE access in the HF_TOKEN environment variable
(on Kaggle: Add-ons -> Secrets). The repo "<your username>/ipc-router-labels" is created
automatically, private.
"""

import json
import os
import signal
import sys


class HFCheckpoint:
    def __init__(self, repo: str = "auto", every: int = 50):
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--checkpoint needs a Hugging Face WRITE token in HF_TOKEN (Kaggle: Add-ons -> Secrets)")
        self.api = HfApi(token=token)
        if repo == "auto":
            repo = f"{self.api.whoami()['name']}/ipc-router-labels"
        self.repo, self.every = repo, every
        self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"[checkpoint] labels are backed up to the private dataset https://huggingface.co/datasets/{repo} "
              f"every {every} questions")
        self.paths = []

    def pull(self, path: str) -> None:
        """Merge the backed-up copy of `path` (if any) into the local file, keyed by question id."""
        from huggingface_hub import hf_hub_download
        name = os.path.basename(path)
        try:
            remote = hf_hub_download(self.repo, name, repo_type="dataset", token=self.api.token,
                                     local_dir=os.path.join(os.path.dirname(path) or ".", ".ckpt"))
        except Exception:
            print(f"[checkpoint] no backup of {name} yet - starting from the local file")
            return
        rows = {}
        for p in (remote, path):
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    if line.strip():
                        r = json.loads(line)
                        rows.setdefault(r["id"], r)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[checkpoint] restored {len(rows)} labelled questions from the backup")

    def watch(self, *paths: str) -> None:
        """Files to upload; also back them up if the process is killed (SIGTERM / Ctrl-C)."""
        self.paths = list(paths)

        def on_stop(signum, frame):
            print(f"\n[checkpoint] stop signal {signum} - backing up before exiting")
            self.push()
            sys.exit(1)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, on_stop)

    def maybe_push(self, n_done_this_run: int) -> None:
        if n_done_this_run % self.every == 0:
            self.push()

    def push(self) -> None:
        for p in self.paths:
            if not os.path.exists(p):
                continue
            try:
                self.api.upload_file(path_or_fileobj=p, path_in_repo=os.path.basename(p), repo_id=self.repo,
                                     repo_type="dataset", commit_message=f"checkpoint {os.path.basename(p)}")
            except Exception as e:          # network blip: keep labelling, the next push will catch up
                print(f"[checkpoint] backup failed ({type(e).__name__}: {e}) - will retry at the next checkpoint")
                return
        print(f"[checkpoint] backed up {sum(1 for _ in open(self.paths[0], encoding='utf-8'))} rows")
