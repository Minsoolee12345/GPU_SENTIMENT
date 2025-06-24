# GPU_SENTI

1. 데이터 수집
  - GPU 게시판 : 퀘이사존, 쿨&조이.
  - 수집 데이터셋 : 63만개.

2. 데이터 라벨링
  - Gemma3 사용해 긍정/중립/부정 라벨링.


4. 데이터 전처리
   - CSV 필터링 : 긍정/중립/부정 만 추출 후 불필요 라벨 제거.
   - 레이블 인코딩 : 문자열 라벨을 정수로 매핑.
   - 텍스트 정제 : 정규화를 사용해 한글/숫자/공백/!,? 외 모든 문자 삭제.
   - 형태소 토큰화 : KoNLPy Okt 분석기로 어간(원형)까지 분리.
   - 불용어 제거 : korean_stopwords.txt(사용자 제작) 읽어 불용어 제거.
   - 캐싱 : 1회 전처리 결과를 pkl(피클)로 저장해 재실행 시 로딩 속도 개선.
   - 문장 길이 통계 : 95-백분위 길이를 max_len으로 설정해 과도한 패딩 방지.
   - 토크나이저 피팅 : 상위 30k 단어만 인덱싱.
   - 패딩 : 모든 샘플을 동일 길이 맞춤.


5. 모델 학습
   - 데이터 분할 : train/validation/test -> 8:1:1
   - Word2Vec 300 d 학습 -> 임베딩 매트릭스 생성
   - BI-LSTM(128 -> 64) + Attention + Dense → Softmax
   - 워밍업 & 본 학습 : 3 epoch 워밍업 후 EarlyStopping+Checkpoint 로 100 epoch 탐색
   - 하이퍼파라미터 : batch 4096, dropout 0.5, L2 1e-4
   - 모델 저장 : final_bilstm_attention.h5 저장
