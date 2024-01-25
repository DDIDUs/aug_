for mode in 1 2
do
    python3 evaluate.py \
        -train_mode $mode \
        -repeat_num 2 \
        &> log.$mode
done
