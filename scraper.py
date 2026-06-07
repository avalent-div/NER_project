# scraper.py
import requests
import pandas as pd
import re
import time
import csv
from datetime import datetime

TARGET_CLEAN = 22500

SUBREDDITS = [
    "africa",
    "anime",
    "anime_irl",
    "animememes",
    "animenews",
    "AskReddit",
    "asia",
    "business",
    "climate",
    "CombatFootage",
    "economics",
    "economy",
    "education",
    "energy",
    "entertainment",
    "environment",
    "europe",
    "finance",
    "geopolitics",
    "globalnews",
    "history",
    "indonesia",
    "interestingasfuck",
    "investing",
    "Military",
    "news",
    "Olympics",
    "politics",
    "PublicFreakout",
    "science",
    "soccer",
    "sports",
    "technology",
    "todayilearned",
    "UnitedNations",
    "worldnews",
]

HEADERS = {"User-Agent": "NER_Project/1.0"}


def clean_text(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\[deleted\]|\[removed\]", "", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) > 10 else None


def scrape_subreddit(subreddit, category="hot", limit=100, after=None):
    url = f"https://www.reddit.com/r/{subreddit}/{category}.json?limit={limit}"
    if after:
        url += f"&after={after}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            print(f"  kena rate limit, istirahat 60 detik dulu...")
            time.sleep(60)
            return scrape_subreddit(subreddit, category, limit, after)
        else:
            print(f"  r/{subreddit}/{category} balik status {r.status_code}, skip")
    except Exception as e:
        print(f"  gagal request: {e}")
    return None


def scrape_until_target():
    all_data    = []
    seen_ids    = set()
    seen_texts  = set()
    clean_count = 0

    print(f"oke mulai scraping, target {TARGET_CLEAN} data bersih")
    print(f"total subreddit: {len(SUBREDDITS)}")
    print("-" * 50)

    for sub in SUBREDDITS:
        if clean_count >= TARGET_CLEAN:
            break

        sub_added = 0

        for category in ["hot", "new", "top"]:
            if clean_count >= TARGET_CLEAN:
                break

            after = None
            page  = 0

            while clean_count < TARGET_CLEAN:
                print(f"  ambil r/{sub}/{category} halaman {page + 1}...")
                data = scrape_subreddit(sub, category, limit=100, after=after)

                if not data or "data" not in data:
                    print(f"  r/{sub}/{category} kosong, lanjut yang lain")
                    break

                posts = data["data"]["children"]
                if not posts:
                    break

                added = 0
                for post in posts:
                    p = post["data"]

                    if p["id"] in seen_ids:
                        continue
                    seen_ids.add(p["id"])

                    if p.get("selftext") in ["[deleted]", "[removed]"]:
                        continue

                    raw   = p["title"] + " " + (p.get("selftext") or "")
                    clean = clean_text(raw)

                    if not clean or clean in seen_texts:
                        continue
                    seen_texts.add(clean)

                    all_data.append({
                        "id":            p["id"],
                        "title":         p["title"],
                        "text":          p.get("selftext", ""),
                        "content_clean": clean,
                        "subreddit":     sub,
                        "created":       datetime.utcfromtimestamp(
                                            p["created_utc"]).strftime("%Y-%m-%d")
                    })
                    added       += 1
                    sub_added   += 1
                    clean_count += 1

                    if clean_count % 500 == 0:
                        print(f"\n  udah {clean_count} dari {TARGET_CLEAN}, terus jalan...\n")

                    if clean_count >= TARGET_CLEAN:
                        break

                print(f"  r/{sub}/{category} hal {page+1}: nambah {added}, total {clean_count}")

                after = data["data"].get("after")
                if not after:
                    break

                page  += 1
                time.sleep(3)

        print(f"\nselesai r/{sub}, kontribusi: {sub_added} data, total sekarang: {clean_count}")
        print("-" * 50)

    df = pd.DataFrame(all_data)

    # simpan dengan quoting yang benar biar CSV tidak corrupt
    df.to_csv("data/reddit_raw.csv", index=False, quoting=csv.QUOTE_ALL)

    print(f"\nselesai semua!")
    print(f"terkumpul : {len(df)} data")
    print(f"target    : {TARGET_CLEAN} data")
    if len(df) >= TARGET_CLEAN:
        print(f"target tercapai, mantap")
    else:
        print(f"kurang {TARGET_CLEAN - len(df)} data, coba tambah subreddit lagi")

    return df


if __name__ == "__main__":
    df = scrape_until_target()

    print("\n5 data pertama:")
    print(df[["id", "title", "subreddit", "created"]].head())

    print("\ndistribusi subreddit:")
    print(df["subreddit"].value_counts())

    print("\ndata disimpan ke data/reddit_raw.csv")