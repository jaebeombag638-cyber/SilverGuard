# SilverGuard 스텝 계획

`계획.md`의 범위를 3개 큰 스텝으로 나눈다. 스텝 완료마다 결과를 `step결과.md`에 누적 기록한다.

## Step 1. 데이터 준비

- KISA 피싱 URL 데이터(2023, 전량) 확보 및 검증
- Tranco 상위 도메인 + 경로 포함 benign URL corpus로 정상 URL 데이터 구성
  (Tranco 도메인만 쓰면 "경로 유무"로 모델이 편법 학습할 위험이 있어 병합)
- phishing/benign 라벨링된 통합 데이터셋 CSV 생성

## Step 2. 모델 학습 및 비교

- 진행 방식: 코드/흐름/결과 해석을 사용자가 완전히 이해하도록, 각 단계마다 자세히 인터뷰하며 진행. 결과가 나오면 왜 그렇게 나왔는지 토론하며 확인 (속도보다 이해 우선)
- **Leakage 점검**: baseline 학습 전, benign 데이터의 두 소스(tranco vs github) 간 주요 lexical feature 분포를 간단히 비교 확인. 큰 차이 없으면 진행, 있으면 보정
- **1단계 (baseline)**: 표준 lexical feature 세트(15~20개 — URL 길이, 특수문자 수, IP 직접사용 여부, 서브도메인 개수, 유명 브랜드 유사도(edit distance), TLD, HTTPS 여부 등) 추출 → XGBoost/RandomForest 학습
- **2단계 (딥러닝)**: character-level CNN 하나만 학습 (Transformer 생략 — 비용 대비 이득 낮다고 판단). PyTorch로 구현, 로컬 내장 GPU 사용
- **비교/완료 기준**: 절대 수치 목표 대신 baseline vs CNN 중 Recall이 더 높고 False Negative가 적은 쪽을 선택하는 상대 비교로 결론. 처음엔 간단한 기준으로 시작해 여러 번 실험하며 기준을 점진적으로 구체화
- 미정 사항(구현하며 확정): train/val/test split 비율·전략, CNN 세부 아키텍처(임베딩 차원, 레이어 수 등)

## Step 3. 데모 앱 구축

- 문자 메시지 텍스트 입력 → URL 추출 → 분류기로 위험도 판정
- LLM 프롬프팅으로 판정 근거 + 대응 가이드 생성
- end-to-end 데모 화면 구성
