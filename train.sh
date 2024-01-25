for mode in 1 2
do
    python3 main.py \
        -train_mode $mode \
        -repeat_num 2 \
        &> log.$mode
done
