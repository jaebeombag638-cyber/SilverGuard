"""
baseline(XGBoost/RandomForest) 학습용 lexical feature 추출 스크립트.
data/url_dataset.csv -> data/features.csv
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

IN_CSV = "url_dataset.csv"
OUT_CSV = "features.csv"

KNOWN_SHORTENERS = {
    "t.ly", "bit.ly", "han.gl", "c11.kr", "tinyurl.com", "goo.gl",
    "is.gd", "buff.ly", "rebrand.ly", "shorturl.at", "url.kr",
}

IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    netloc = parsed.netloc.split(":")[0]  # 포트 번호 제거
    path = parsed.path or ""
    labels = netloc.split(".")
    domain = ".".join(labels[-2:]) if len(labels) >= 2 else netloc
    subdomain = ".".join(labels[:-2]) if len(labels) > 2 else ""

    return {
        "url_len": len(url),
        "domain_len": len(netloc),
        "path_len": len(path),
        "num_dots": url.count("."),
        "subdomain_count": max(len(labels) - 2, 0),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / len(url) if url else 0,
        "has_ip": int(bool(IP_RE.match(netloc))),
        "has_at_symbol": int("@" in url),
        "is_https": int(parsed.scheme == "https"),
        "tld": labels[-1].lower() if labels else "",
        "subdomain_entropy": shannon_entropy(subdomain),
        "has_path": int(len(path) > 1),
        "has_query": int(bool(parsed.query)),
        "special_char_count": sum(url.count(c) for c in "-_%=&?"),
        "is_shortener": int(domain.lower() in KNOWN_SHORTENERS),
    }


df = pd.read_csv(IN_CSV)
feat = df["url"].apply(extract).apply(pd.Series)
out = pd.concat([df[["url", "label", "source"]], feat], axis=1)
out.to_csv(OUT_CSV, index=False)

print(f"rows: {len(out)}")
print(out.groupby("label")[
    ["url_len", "subdomain_count", "subdomain_entropy", "has_ip", "is_shortener"]
].mean())
print(f"saved -> {OUT_CSV}")
