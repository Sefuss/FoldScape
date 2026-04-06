"""
Hugging Face Hub collector for FoldScape.
Discovers protein ML models/spaces on HF and merges into repos.json.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    exit(1)


class HFProteinCollector:
    """Collect protein ML models and spaces from Hugging Face Hub."""

    EXPAND_FIELDS = [
        "downloadsAllTime", "likes", "createdAt", "lastModified",
        "pipeline_tag", "library_name", "tags"
    ]

    def __init__(self, token=None):
        self.api = HfApi(token=token or os.getenv("HF_TOKEN"))
        self.seen = {}  # repo_id -> record
        self._load_config()

    def _load_config(self):
        """Load HF settings from config.json."""
        config_path = Path(__file__).parent.parent.parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        hf = config.get("huggingface", {})
        self.search_keywords = hf.get("search_keywords", [
            "protein", "ESM", "ProteinMPNN", "AlphaFold",
            "ProtT5", "protein design", "protein structure",
        ])
        self.relevance_keywords = hf.get("relevance_keywords", [
            "protein", "antibody", "folding", "alphafold", "esm",
            "rosetta", "amino acid", "pdb", "sequence", "binder",
            "proteinmpnn", "prott5", "molecular", "structure prediction",
        ])
        self.limit_per_query = hf.get("limit_per_query", 200)
        self.recent_days = hf.get("recent_days", 7)

    # Keywords that are too generic on their own — only count if accompanied
    # by another bio signal in the same record
    WEAK_KEYWORDS = {"esm", "sequence", "molecular", "pdb"}

    # Strong signals that confirm protein/bio context
    BIO_SIGNALS = {
        "protein", "antibody", "folding", "alphafold", "amino",
        "binder", "proteinmpnn", "prott5", "rosetta", "uniprot",
        "proteingpt", "protgpt", "progen", "esm2", "esmfold",
        "structure prediction", "inverse folding",
    }

    def _is_protein_related(self, model_id, tags, pipeline_tag, library_name):
        """Check if a model is protein/bio related. Uses strong/weak keyword distinction."""
        haystack = " ".join([
            model_id or "",
            " ".join(tags or []),
            pipeline_tag or "",
            library_name or "",
        ]).lower()

        # Check strong signals first (any one is enough)
        if any(sig in haystack for sig in self.BIO_SIGNALS):
            return True

        # Weak keywords only count if at least 2 are present
        weak_hits = sum(1 for kw in self.WEAK_KEYWORDS if kw in haystack)
        return weak_hits >= 2

    def _extract_record(self, model, hit_term=None):
        """Extract a normalized record from an HF model object."""
        model_id = getattr(model, "id", None) or getattr(model, "modelId", None)
        if not model_id:
            return None

        tags = list(getattr(model, "tags", []) or [])
        pipeline_tag = getattr(model, "pipeline_tag", None)
        library_name = getattr(model, "library_name", None)

        created = getattr(model, "created_at", None) or getattr(model, "createdAt", None)
        modified = getattr(model, "last_modified", None) or getattr(model, "lastModified", None)

        return {
            "hf_id": model_id,
            "author": getattr(model, "author", None) or model_id.split("/")[0] if "/" in model_id else None,
            "pipeline_tag": pipeline_tag,
            "library_name": library_name,
            "tags": tags,
            "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created) if created else None,
            "last_modified": modified.isoformat() if hasattr(modified, "isoformat") else str(modified) if modified else None,
            "downloads_30d": getattr(model, "downloads", None),
            "downloads_all_time": getattr(model, "downloads_all_time", None) or getattr(model, "downloadsAllTime", None),
            "likes": getattr(model, "likes", None),
            "trending_score": getattr(model, "trending_score", None),
            "is_protein_related": self._is_protein_related(model_id, tags, pipeline_tag, library_name),
            "search_hit_terms": [hit_term] if hit_term else [],
        }

    def _accumulate(self, models, hit_term=None, max_count=None):
        """Add models to seen dict, deduplicating."""
        count = 0
        for model in models:
            if max_count and count >= max_count:
                break

            rec = self._extract_record(model, hit_term)
            if not rec:
                continue

            hf_id = rec["hf_id"]
            if hf_id in self.seen:
                # Just add the search term if new
                if hit_term and hit_term not in self.seen[hf_id]["search_hit_terms"]:
                    self.seen[hf_id]["search_hit_terms"].append(hit_term)
                continue

            self.seen[hf_id] = rec
            count += 1

        return count

    def collect_all(self):
        """Run all collection passes."""
        print("Starting HF collection...")
        total_before = len(self.seen)

        # Pass 1: keyword searches (the main discovery mechanism)
        for kw in self.search_keywords:
            print(f"  Searching HF for: {kw}...")
            try:
                models = self.api.list_models(
                    search=kw,
                    sort="downloads",
                    limit=self.limit_per_query,
                    expand=self.EXPAND_FIELDS,
                )
                found = self._accumulate(models, hit_term=kw)
                print(f"    +{found} new models")
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(0.5)

        # Pass 2: trending protein models (likes7d)
        print("  Fetching trending models...")
        for kw in ["protein", "ESM", "structure prediction"]:
            try:
                models = self.api.list_models(
                    search=kw,
                    sort="likes7d",
                    limit=50,
                    expand=self.EXPAND_FIELDS,
                )
                found = self._accumulate(models, hit_term=f"trending:{kw}")
                print(f"    trending:{kw} +{found} new")
            except Exception as e:
                print(f"    ERROR: {e}")
            time.sleep(0.5)

        # Filter to protein-related only
        protein_models = {k: v for k, v in self.seen.items() if v["is_protein_related"]}
        total_new = len(protein_models) - total_before

        print(f"\nHF collection complete: {len(protein_models)} protein models "
              f"(from {len(self.seen)} total scanned)")
        return list(protein_models.values())

    def merge_into_repos(self, repos_path="data/repos.json"):
        """Merge HF data into existing repos.json."""
        repos_path = Path(repos_path)

        # Load existing repos
        with open(repos_path, "r", encoding="utf-8") as f:
            repos = json.load(f)

        # Build lookups for cross-linking
        existing_by_id = {}
        existing_by_name = {}
        for repo in repos:
            existing_by_id[repo["repo_id"].lower()] = repo
            existing_by_name[repo["metadata"]["name"].lower()] = repo

        # Load manual crosslinks from config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            crosslinks = json.load(f).get("huggingface", {}).get("crosslinks", {})

        # Collect HF models
        hf_models = self.collect_all()

        linked = 0
        added = 0

        for hf_rec in hf_models:
            hf_id = hf_rec["hf_id"]
            hf_name = hf_id.split("/")[-1].lower() if "/" in hf_id else hf_id.lower()

            # Try to find matching GitHub repo:
            # 1) Manual crosslink (hf_id -> github repo_id)
            # 2) Exact name match
            target = None
            gh_id = crosslinks.get(hf_id, "").lower()
            if gh_id and gh_id in existing_by_id:
                target = existing_by_id[gh_id]
            elif hf_name in existing_by_name:
                target = existing_by_name[hf_name]

            if target:
                # Link HF data to existing GitHub repo (keep best match per GH repo)
                existing = target.get("hf_metadata")
                new_dl = hf_rec["downloads_all_time"] or 0
                if not existing or new_dl > (existing.get("downloads_all_time") or 0):
                    target["hf_metadata"] = {
                        "hf_id": hf_rec["hf_id"],
                        "pipeline_tag": hf_rec["pipeline_tag"],
                        "library_name": hf_rec["library_name"],
                        "downloads_30d": hf_rec["downloads_30d"],
                        "downloads_all_time": hf_rec["downloads_all_time"],
                        "likes": hf_rec["likes"],
                        "tags": hf_rec["tags"],
                        "last_modified": hf_rec["last_modified"],
                    }
                linked += 1
            else:
                # Add as HF-only entry
                repos.append({
                    "repo_id": f"hf:{hf_id}",
                    "source": "huggingface",
                    "metadata": {
                        "name": hf_id.split("/")[-1] if "/" in hf_id else hf_id,
                        "description": None,  # HF API doesn't return descriptions in list
                        "url": f"https://huggingface.co/{hf_id}",
                        "stars": hf_rec["likes"] or 0,
                        "forks": 0,
                        "last_updated": hf_rec["last_modified"],
                        "created_at": hf_rec["created_at"],
                        "language": hf_rec["library_name"],
                        "license": None,
                        "topics": hf_rec["tags"][:10],  # Cap tags
                    },
                    "hf_metadata": {
                        "hf_id": hf_rec["hf_id"],
                        "pipeline_tag": hf_rec["pipeline_tag"],
                        "library_name": hf_rec["library_name"],
                        "downloads_30d": hf_rec["downloads_30d"],
                        "downloads_all_time": hf_rec["downloads_all_time"],
                        "likes": hf_rec["likes"],
                        "tags": hf_rec["tags"],
                        "last_modified": hf_rec["last_modified"],
                    },
                    "classification": {
                        "category": None,
                        "subcategory": None,
                        "layer": None,
                    },
                    "domain_specific": {
                        "experimental_validation": None,
                        "expression_systems": [],
                        "gpu_requirement": None,
                        "input_types": [],
                        "output_formats": [],
                    },
                    "tracking": {
                        "first_tracked": datetime.now().isoformat(),
                        "star_velocity_7d": 0,
                        "star_velocity_30d": 0,
                        "trending": False,
                    },
                })
                added += 1

        # Atomic write
        temp_path = repos_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(repos, f, indent=2, ensure_ascii=False)
        temp_path.replace(repos_path)

        print(f"\nMerge complete: {linked} linked to GitHub repos, {added} new HF-only entries")
        print(f"Total repos now: {len(repos)}")

        # Log the run
        self._save_run_log(len(hf_models), linked, added, len(repos))

    def _save_run_log(self, scanned, linked, added, total):
        """Save collection run metadata for debugging."""
        log_dir = Path("data/run_logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "huggingface",
            "models_scanned": scanned,
            "linked_to_github": linked,
            "new_hf_only": added,
            "total_repos_after": total,
            "keywords_used": self.search_keywords,
        }

        log_path = log_dir / f"hf_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        print(f"Run log saved to {log_path}")


if __name__ == "__main__":
    collector = HFProteinCollector()
    collector.merge_into_repos("data/repos.json")
