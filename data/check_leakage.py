"""
benign 데이터의 두 소스(tranco vs github_benign_corpus) 간
lexical feature 분포 차이를 확인하는 스크립트.
"""
import pandas as pd
from urllib.parse import urlparse

df = pd.read_csv("url_dataset.csv")
benign = df[df["label"] == "benign"].copy()


def features(url: str) -> pd.Series:
    parsed = urlparse(url)
    path = parsed.path or ""
    return pd.Series({
        "url_len": len(url),
        "has_path": int(len(path) > 1),  # "/" 하나만 있는 경우는 경로 없음으로 취급
        "path_len": len(path),
        "subdomain_count": max(parsed.netloc.count(".") - 1, 0),
        "special_char_count": sum(url.count(c) for c in "-_%=&?"),
        "has_query": int(bool(parsed.query)),
    })


feat = benign["url"].apply(features)
benign = pd.concat([benign, feat], axis=1)

summary = benign.groupby("source")[
    ["url_len", "has_path", "path_len", "subdomain_count", "special_char_count", "has_query"]
].agg(["mean", "std"])

pd.set_option("display.width", 120)
print(summary)
