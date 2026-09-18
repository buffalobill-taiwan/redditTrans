import random
import time
from pathlib import Path
from typing import List, Dict

import requests

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,zh-TW;q=0.8,zh;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.reddit.com/",
    "DNT": "1",
}

REDDIT_DOMAINS = ["www.reddit.com", "api.reddit.com", "old.reddit.com"]

COOKIE_PATH = Path(__file__).resolve().parent.parent / "cookie.txt"


def load_cookies() -> Dict[str, str]:
    """Load the Netscape cookie file (cookie.txt) into a dict."""
    if not COOKIE_PATH.exists():
        return {}
    cookies = {}
    for line in COOKIE_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        cookies[parts[5]] = parts[6]
    return cookies


def get_hot_posts(subreddit: str, limit: int = 10, sort: str = "hot") -> List[Dict]:
    cookies = load_cookies()
    if cookies:
        print(f"使用 cookie.txt 進行抓取 ({len(cookies)} 個 cookies)")
    else:
        print("警告：未找到 cookie.txt，Reddit 可能回傳 403/302")

    for domain in REDDIT_DOMAINS:
        session = requests.Session()
        session.headers.update(HEADERS)
        if cookies:
            session.cookies.update(cookies)

        url = f"https://{domain}/r/{subreddit}/{sort}.json"
        params = {"limit": min(max(limit + 3, 10), 100), "raw_json": 1}

        time.sleep(random.uniform(0.3, 0.9))

        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type:
                continue
        except requests.RequestException:
            continue

        data = response.json()
        posts = []
        for item in data["data"]["children"]:
            post = item["data"]
            if post.get("stickied", False):
                continue
            if len(posts) >= limit:
                break
            posts.append({
                "title": post.get("title", ""),
                "selftext": post.get("selftext", ""),
                "author": str(post.get("author", "[deleted]")),
                "score": post.get("score", 0),
                "permalink": f"https://reddit.com{post.get('permalink', '')}",
                "url": post.get("url", ""),
                "id": post.get("id", ""),
                "num_comments": post.get("num_comments", 0),
                "is_self": post.get("is_self", True),
            })
        return posts

    raise RuntimeError(
        f"所有 Reddit 網域皆回傳 403 或是無法載入，r/{subreddit} 暫時無法存取"
    )


def get_post_title_and_content(post: Dict) -> tuple:
    content = post.get("selftext", "")
    if content:
        return (post["title"], f"{content}")
    return (post["title"], post["title"])


def get_post_content(post: Dict) -> str:
    if post["selftext"]:
        return f"{post['title']}\n\n{post['selftext']}"
    return post["title"]
