# Dynamic Augmentation

## 실험 환경 

실험 모델 : VGGnet16, ShakeShake(26 2x32d, 2x96d)

학습 데이터 세트 : CIFAR10

## 학습 모드

|모드|설명|
|------|---|
|ORIGINAL| 원본 데이터만 사용 (4만개/에폭)  |
|DYNAMIC_AUG_ONLY| 동적 증강 데이터만 사용 (4만개/에폭) |
|ORIG_PLUS_DYNAMIC_AUG_1X| 원본 데이터 일부 + 동적 증강 데이터 일부 (비율 만큼) = 4만개/에폭 |
|ORIG_PLUS_DYNAMIC_AUG_2X| 원본 데이터 (4만개) + 증강 데이터 일부 (비율 만큼) = 4만개 이상(최대 8만개)/에폭|
|ORIG_PLUS_VALID_AUG_1X| 원본 데이터 일부 + validation data 포함 동적 증강 데이터 일부 (비율 만큼) = 약 4만개/에폭 |
|ORIG_PLUS_VALID_AUG_2X| 원본 데이터 (4만개) + validation data 포함 증강 데이터 일부 (비율 만큼) = 4만개 이상(최대 9만개)/에폭|

## 모델 학습

    bash train.sh

## 학습 모델 평가

    bash eval.sh

