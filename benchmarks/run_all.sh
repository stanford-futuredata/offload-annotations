TRIALS=5

blackscholes_cupy () {
    name="blackscholes_cupy"
    start=16
    for size in $(seq $start 1 31);
        do python run.py -b $name -m naive -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 27);
        do python run.py -b $name -m cuda -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 31);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

blackscholes_torch () {
    name="blackscholes_torch"
    start=16
    for size in $(seq $start 1 31);
        do python run.py -b $name -m naive -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 27);
        do python run.py -b $name -m cuda -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 31);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

blackscholes_torch

# Post-processing
# ===============
# cat results/blackscholes_torch_cpu | grep Med | grep Total | awk '{print $3}'
# OR ./parse.sh blackscholes_torch
