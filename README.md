# GPU_SENTI

1. 데이터 수집
  - GPU 게시판 : 쿨&조이.(Selenium)
  - ![image](https://github.com/user-attachments/assets/635e4e3f-c52e-4b13-88d9-4c4c27ba0d64)
  - 수집 데이터셋 : 63만개.

-------------------------------------------------------------------------------------------------

2. 데이터 라벨링
  - Gemma3 사용해 긍정/중립/부정 라벨링.

-------------------------------------------------------------------------------------------------

4. 데이터 전처리
   - CSV 필터링 : 긍정/중립/부정 만 추출 후 불필요 라벨 제거.
   - ![image](https://github.com/user-attachments/assets/01ce16e8-40b6-4374-be7b-2d624df351df)

   - 레이블 인코딩 : 문자열 라벨을 정수로 매핑.
   - 텍스트 정제 : 정규화를 사용해 한글/숫자/공백/!,? 외 모든 문자 삭제.
   - ![image](https://github.com/user-attachments/assets/69a61734-ade5-4899-a998-dceddea0b1fc)

   - 형태소 토큰화 : KoNLPy Okt 분석기로 어간(원형)까지 분리.
   - 불용어 제거 : korean_stopwords.txt(사용자 제작) 읽어 불용어 제거.
   - 캐싱 : 1회 전처리 결과를 pkl(피클)로 저장해 재실행 시 로딩 속도 개선.
   - 문장 길이 통계 : 95-백분위 길이를 max_len으로 설정해 과도한 패딩 방지.
   - 토크나이저 피팅 : 상위 30k 단어만 인덱싱.
   - 패딩 : 모든 샘플을 동일 길이 맞춤.

-------------------------------------------------------------------------------------------------

5. 모델 학습
   - 데이터 분할 : train/validation/test -> 8:1:1
   - Word2Vec 300 d 학습 -> 임베딩 매트릭스 생성
   - BI-LSTM(128 -> 64) + Attention + Dense → Softmax
   - 워밍업 & 본 학습 : 3 epoch 워밍업 후 EarlyStopping+Checkpoint 로 100 epoch 탐색
   - 하이퍼파라미터 : batch 4096, dropout 0.5, L2 1e-4
   - 모델 저장 : final_bilstm_attention.h5 저장

-------------------------------------------------------------------------------------------------

6. 스트림릿 구성
   - 사이드바 구성 : GPU 모델 및 필터 설정
  
![image](https://github.com/user-attachments/assets/29993d59-81e6-4756-81a2-0f1dd24e9fb1)
![image](https://github.com/user-attachments/assets/4d7daf70-88b1-437e-8005-807f84e1385c)


   - 감성 분포 시각화
       - Plotly 막대 차트
       - ![image](https://github.com/user-attachments/assets/2e019094-c63a-4583-8009-014d7a72edd1)
       - Plotly 도넛 차트
       - ![image](https://github.com/user-attachments/assets/29738870-5c46-44f2-a2db-06e45dacb673)
  - 키워드 분석 : WordCloud + Counter.most_common(20) 감성별 키워드 빈도
  - ![image](https://github.com/user-attachments/assets/efdae6f9-3173-4219-83ee-d2209b14ffce)
  - ![image](https://github.com/user-attachments/assets/4b30b49d-f977-4497-85c4-56fb9af7c939)


