for w_tbase in 32 96
do 
    for mode in 5 6
    do
        python3 main.py \
            -train_mode $mode \
            -w_base $w_tbase \
            -train_model "shakeshake" \
            -repeat_num 3 \
            &> train_log.$mode
    done
done