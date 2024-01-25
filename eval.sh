for mode in 1 2
do
    python3 evaluate.py \
        -train_mode $mode \
        &> log.$mode
done
