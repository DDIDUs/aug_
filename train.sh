for mode in 1 2 3 4 5
do
    python3 main.py \
        -train_mode $mode \
        -repeat_num 5 \
        &> train_log.$mode
done
