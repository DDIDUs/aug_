for mode in 1,2
do
    python3 main.py \
        -train_mode $mode \
        &> log.$mode
done