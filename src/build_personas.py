"""
Build persona ground-truth relevance sets from real MIND user click logs,
and sample the content pool the recommenders will retrieve from.

This is the step that keeps evaluation honest: relevance labels come from
real users' actual reading behavior, filtered to users whose interests
match a persona's category profile -- not from us deciding what "should"
be relevant.

Usage:
    python src/build_personas.py
"""
import os
import json
import random
from collections import Counter, defaultdict

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "personas.json")

NEWS_COLS = [
    "news_id", "category", "subcategory", "title", "abstract",
    "url", "title_entities", "abstract_entities",
]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

SAMPLE_SIZE = 800          # target size of the final content pool (within 500-1000 target)
PER_PERSONA_POOL_SAMPLE = 60  # cap on how many of each persona's real relevant articles go into the pool
MIN_MATCHING_USERS = 15    # minimum users required to trust a persona's ground truth
RANDOM_SEED = 42


def load_news(split_dir: str) -> pd.DataFrame:
    path = os.path.join(split_dir, "news.tsv")
    df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS, index_col=False)
    return df


def load_behaviors(split_dir: str) -> pd.DataFrame:
    path = os.path.join(split_dir, "behaviors.tsv")
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS, index_col=False)
    return df


def user_category_profile(history: str, news_cat: dict) -> Counter:
    """Category counts for a user's clicked history (space-separated news IDs)."""
    if not isinstance(history, str) or not history.strip():
        return Counter()
    ids = history.strip().split()
    cats = [news_cat.get(nid) for nid in ids if news_cat.get(nid)]
    return Counter(cats)


def clicked_ids_from_impressions(impressions: str) -> list:
    """Impressions field is 'newsID-label' space separated; label=1 means clicked."""
    if not isinstance(impressions, str):
        return []
    clicked = []
    for token in impressions.strip().split():
        if "-" in token:
            nid, label = token.rsplit("-", 1)
            if label == "1":
                clicked.append(nid)
    return clicked


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        persona_config = json.load(f)
    threshold = persona_config["match_threshold"]
    personas = persona_config["personas"]

    train_dir = os.path.join(RAW_DIR, "MINDsmall_train")
    news_df = load_news(train_dir)
    behaviors_df = load_behaviors(train_dir)

    news_cat = dict(zip(news_df["news_id"], news_df["category"]))

    # Build each user's category profile from their click history.
    user_profiles = {}
    user_history_ids = {}
    for _, row in behaviors_df.iterrows():
        uid = row["user_id"]
        if uid in user_profiles:
            continue
        profile = user_category_profile(row["history"], news_cat)
        if profile:
            user_profiles[uid] = profile
            user_history_ids[uid] = row["history"].strip().split()

    # Also collect each user's clicked impression articles (separate from history)
    # to use as additional held-out ground-truth signal.
    user_clicked_impressions = defaultdict(set)
    for _, row in behaviors_df.iterrows():
        uid = row["user_id"]
        clicked = clicked_ids_from_impressions(row["impressions"])
        if clicked:
            user_clicked_impressions[uid].update(clicked)

    ground_truth = {}
    persona_stats = []

    for persona in personas:
        pid = persona["id"]
        cats = set(persona["categories"])
        matching_users = []
        for uid, profile in user_profiles.items():
            total = sum(profile.values())
            if total == 0:
                continue
            in_cat = sum(cnt for cat, cnt in profile.items() if cat in cats)
            if (in_cat / total) >= threshold:
                matching_users.append(uid)

        relevant_ids = set()
        for uid in matching_users:
            # Only count clicks that fall within the persona's own target
            # categories. Using a matching user's FULL history here would
            # leak generic/globally-popular clicks into "relevant," which
            # inflates the popularity baseline's apparent precision for
            # reasons that have nothing to do with topical relevance.
            hist = user_history_ids.get(uid, [])
            relevant_ids.update(nid for nid in hist if news_cat.get(nid) in cats)
            clicked = user_clicked_impressions.get(uid, [])
            relevant_ids.update(nid for nid in clicked if news_cat.get(nid) in cats)

        ground_truth[pid] = sorted(relevant_ids)
        persona_stats.append({
            "persona_id": pid,
            "name": persona["name"],
            "matching_users": len(matching_users),
            "relevant_articles": len(relevant_ids),
            "sufficient_data": len(matching_users) >= MIN_MATCHING_USERS,
        })

    stats_df = pd.DataFrame(persona_stats)
    print(stats_df.to_string(index=False))
    insufficient = stats_df[~stats_df["sufficient_data"]]
    if len(insufficient):
        print(
            "\nWARNING: these personas have fewer than "
            f"{MIN_MATCHING_USERS} matching users; treat their eval numbers "
            "as directional only, not statistically reliable:"
        )
        print(insufficient["name"].tolist())

    # Build the content pool: a CAPPED sample of each persona's real
    # ground-truth articles (so no single persona's huge click volume can
    # blow past the target pool size), plus random distractors to reach
    # SAMPLE_SIZE and give retrieval genuine noise to filter out.
    pool_ids = set()
    for pid, ids in ground_truth.items():
        if not ids:
            continue
        sample_n = min(PER_PERSONA_POOL_SAMPLE, len(ids))
        pool_ids.update(random.sample(ids, sample_n))

    all_ids = news_df["news_id"].tolist()
    remaining_needed = max(0, SAMPLE_SIZE - len(pool_ids))
    distractor_candidates = [i for i in all_ids if i not in pool_ids]
    random.shuffle(distractor_candidates)
    pool_ids.update(distractor_candidates[:remaining_needed])

    pool_df = news_df[news_df["news_id"].isin(pool_ids)].reset_index(drop=True)

    # Sanity check: how many of each persona's relevant articles actually
    # made it into the final pool (this is the ceiling on that persona's
    # achievable Precision@10/NDCG@10 later).
    print("\nRelevant articles included in final pool, per persona:")
    for pid, ids in ground_truth.items():
        in_pool = len(set(ids) & pool_ids)
        print(f"  {pid}: {in_pool} / {len(ids)} relevant articles in pool")

    pool_path = os.path.join(PROCESSED_DIR, "content_pool.csv")
    gt_path = os.path.join(PROCESSED_DIR, "persona_ground_truth.json")
    stats_path = os.path.join(PROCESSED_DIR, "persona_stats.csv")

    pool_df.to_csv(pool_path, index=False)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    stats_df.to_csv(stats_path, index=False)

    print(f"\nContent pool: {len(pool_df)} articles -> {pool_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Persona stats: {stats_path}")


if __name__ == "__main__":
    main()
