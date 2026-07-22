# SilverGuard 스텝 결과

## Step 1. 데이터 준비 — 완료 (2026-07-22)

### 한 일

- **피싱 URL**: `한국인터넷진흥원_피싱사이트 URL_20231231.csv` (2023-01-01 ~ 2023-12-31 전량, 27,582건, 날짜 결측 없음) 확보 검증
- **정상 URL 2종 병합**
  - Tranco top-1M 도메인 중 14,000개 랜덤 샘플 → `https://domain/` 형태로 사용
  - GitHub 공개 legitimate/malicious URL corpus(faizann24/Using-machine-learning-to-detect-malicious-URLs)에서 `label=good` 14,000건 랜덤 샘플 → 경로/파라미터 포함 URL 확보
  - 이유: Tranco 도메인만 쓰면 "경로 유무"만으로 모델이 편법 학습(cheating)할 위험이 있어 경로 포함 benign 데이터로 보강
- 세 소스를 `url,label,source` 스키마로 통합, 셔플 후 `data/url_dataset.csv` 저장

### 산출물

- `data/url_dataset.csv` — 통합 라벨링 데이터셋 (phishing 27,582 / benign 28,000, 총 55,582건)
- `data/build_dataset.py` — 재현 가능한 생성 스크립트 (seed=42)
- `data/tranco_tmp/top-1m.csv` — Tranco 원본 (재현용, 필요 시 삭제 가능)
- `data/benign_paths_raw.csv` — GitHub benign corpus 원본 (재현용, 420,465건 중 good만 사용)

### 열린 질문 / 다음 스텝에서 확인 필요

- Tranco 상위 도메인 자체는 사실상 100% "안전"으로 간주했으나, 실제로는 일부 만료/재등록 도메인이 섞였을 가능성 — Step 2 학습 전 간단한 sanity check 고려
- 라벨 소스가 다른 두 benign 그룹(tranco vs github_benign_corpus) 간 lexical feature 분포 차이가 클 경우, 모델이 "source"를 학습하는 leakage 가능성 — feature 설계 시 확인 필요

## Step 2. 모델 학습 및 비교 — 진행 중 (2026-07-22)

### Leakage 점검 결과

`data/check_leakage.py`로 benign 내 tranco vs github_benign_corpus, 그리고 benign vs phishing(KISA) 간 lexical feature(url_len, has_path, path_len, subdomain_count, special_char_count, has_query) 분포를 비교.

**발견**: KISA 피싱 데이터는 대부분 경로 없이 도메인만 등록되어 있어(`url_len` 평균 22.5, std 3.26), tranco benign(22.9, std 4.4)과 거의 동일한 분포. 반면 github_benign_corpus는 실제 경로 포함 URL이라 url_len이 훨씬 김(54.1). subdomain_count는 phishing 0.69 vs benign 0.20으로 뚜렷이 구분됨(진짜 신호로 판단).

**결론/의사결정**: 이번 데이터셋은 "경로 없는 도메인형 피싱" 탐지에 집중하기로 함. 경로가 붙는 피싱(예: `가짜은행.com/login?id=123`) 사례가 KISA 데이터에 거의 없어 지금은 학습 범위에서 제외 — 이 한계를 최종 결과 해석 시 명시. 시간이 남으면 이후 단계에서 경로 포함 피싱 URL 데이터를 추가로 수집해 보강하는 것을 향후 과제로 남김.

### Feature 추출

`data/extract_features.py`로 17개 lexical feature 추출 → `data/features.csv` (55,582행). TLD 분석 결과 benign(tranco/github)은 글로벌 유명 도메인 위주, phishing(KISA)은 `.pro/.buzz/.wtf/.hair` 등 저가·특이 TLD 위주로 확인. 또한 KISA 도메인 실사례 확인 결과 브랜드 유사(typosquat) 패턴이 아니라 **랜덤 문자열 서브도메인(공유 부모 도메인 반복, 예: `yahwagsc.pro`) + URL 단축서비스(`han.gl`, `c11.kr`, `t.ly`)** 위주로 확인 — 이에 따라 브랜드 유사도 feature는 제외하고 `subdomain_entropy`(서브도메인 문자열 랜덤성), `is_shortener`(단축URL 여부) feature로 대체.

최종 feature 17개: url_len, domain_len, path_len, num_dots, subdomain_count, num_hyphens, num_digits, digit_ratio, has_ip, has_at_symbol, is_https, tld, subdomain_entropy, has_path, has_query, special_char_count, is_shortener

### Baseline 학습 결과 (`data/train_baseline.py`)

- Split: stratified 70/15/15 (train 38,907 / val 8,337 / test 8,338), random_state=42
- **RandomForest** (val): accuracy 99.88%, phishing recall 99.83%, F1 99.88%
- **XGBoost** (val): accuracy 99.87%, phishing recall 99.90%, F1 99.87%

**Feature importance 상위**: `path_len`(17.0%), `url_len`(10.2%), `subdomain_entropy`(7.15%), `domain_len`(7.03%), `tld_com`(6.89%) — 상위권에 path/length 관련 feature가 많아, 앞서 확인한 "KISA=경로 없음" 편향이 실제로 모델에 영향을 주고 있는지 ablation으로 검증.

**Ablation (path_len/url_len/domain_len/has_path 4개 제거 후 재학습)**: RandomForest accuracy 98.98%/recall 98.91%, XGBoost accuracy 98.88%/recall 98.79% — 약 1%p 하락에 그침. → path 관련 feature가 성능을 일부 끌어올리지만 전적으로 그것에 의존하는 건 아님을 확인 (순수 신호만으로도 98.9% 수준 유지). artifact 제거 후 top feature: tld_com, subdomain_entropy, subdomain_count, special_char_count, digit_ratio, num_digits, is_shortener — 예상한 "진짜 신호" 위주로 재배치됨.

**결정**: 17개 feature 전체를 최종 baseline으로 채택. 최종 보고서에는 "일부 feature(path_len 등)가 데이터 수집 방식(KISA=도메인 단위 등록) 영향을 받을 수 있으나, ablation 결과 이를 제외해도 Recall 98.9% 유지"라는 한계와 근거를 함께 명시하기로 함.

### 다음 할 일

- character-level CNN(PyTorch) 학습 → baseline과 비교
