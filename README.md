# Dynamic Augmentation

## 실험 환경 

실험 모델 : VGGnet16

학습 데이터 세트 : CIFAR10

## 학습 모드

|모드|설명|
|------|---|
|DYNAMIC_AUG| 원본 데이터 (4만개) + 증강 데이터 (비율) |
|DYNAMIC_AUG_ONLY| 증강 데이터 (4만개) |

## 모델 학습

    bash train.sh

## 학습 모델 평가

    bash eval.sh
