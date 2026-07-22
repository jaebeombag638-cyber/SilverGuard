"""
URL 위험도 판별기 학습용 데이터셋 생성 스크립트.

소스:
  - phishing: 한국인터넷진흥원 피싱사이트 URL (2023, 전량)
  - benign:   Tranco top-1M 도메인 (경로 없음, 상위 N개) +
              GitHub 공개 legitimate URL corpus (경로 포함, N개 샘플)
              -> Tranco만 쓰면 "경로 유무"로 모델이 편법 학습할 위험이 있어 혼합
"""
import csv
import random

random.seed(42)

KISA_CSV = r"C:\Users\hi\Downloads\한국인터넷진흥원_피싱사이트 URL_20231231.csv"
TRANCO_CSV = r"C:\SilverGuard\data\tranco_tmp\top-1m.csv"
BENIGN_PATH_CSV = r"C:\SilverGuard\data\benign_paths_raw.csv"
OUT_CSV = r"C:\SilverGuard\data\url_dataset.csv"

N_TRANCO = 14000
N_BENIGN_PATH = 14000

rows = []

# 1) phishing (KISA, 전량)
with open(KISA_CSV, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        url = r["홈페이지주소"].strip()
        if url:
            rows.append((url, "phishing", "KISA"))

n_phishing = len(rows)

# 2) benign - Tranco 상위 도메인 (경로 없음)
with open(TRANCO_CSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    tranco_domains = [r[1].strip() for r in reader if len(r) >= 2]

tranco_sample = random.sample(tranco_domains, min(N_TRANCO, len(tranco_domains)))
for domain in tranco_sample:
    rows.append((f"https://{domain}/", "benign", "tranco"))

# 3) benign - 경로 포함 legitimate URL corpus
benign_path_urls = []
with open(BENIGN_PATH_CSV, encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("label", "").strip() == "good":
            url = r["url"].strip()
            if url:
                benign_path_urls.append(url)

benign_path_sample = random.sample(
    benign_path_urls, min(N_BENIGN_PATH, len(benign_path_urls))
)
for url in benign_path_sample:
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    rows.append((url, "benign", "github_benign_corpus"))

random.shuffle(rows)

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label", "source"])
    writer.writerows(rows)

n_benign = len(rows) - n_phishing
print(f"phishing: {n_phishing}")
print(f"benign:   {n_benign}  (tranco={len(tranco_sample)}, path_corpus={len(benign_path_sample)})")
print(f"total:    {len(rows)}")
print(f"saved -> {OUT_CSV}")
