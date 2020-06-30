TRIALS=5

blackscholes_cupy () {
    name="blackscholes_cupy"
    start=16
    for size in $(seq $start 1 31);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 27);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 31);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

blackscholes_torch () {
    name="blackscholes_torch"
    start=16
    for size in $(seq $start 1 31);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 27);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 31);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

crime_index () {
    name="crime_index"
    start=13
    for size in $(seq $start 1 31);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 27);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 31);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

haversine_cupy () {
    name="haversine_cupy"
    start=20
    for size in $(seq $start 1 32);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 28);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 32);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

haversine_torch () {
    name="haversine_torch"
    start=20
    for size in $(seq $start 1 32);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 28);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 32);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

dbscan () {
    name="dbscan"
    start=8
    for size in $(seq $start 1 16);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 22);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 21);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

pca () {
    name="pca"
    start=4
    for size in $(seq $start 1 18);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 12);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 19);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

tsvd () {
    name="tsvd"
    start=10
    for size in $(seq $start 1 22);
        do python run.py -b $name -m cpu -s $size --trials $TRIALS >> "results/"$name"_cpu"
    done
    for size in $(seq $start 1 20);
        do python run.py -b $name -m gpu -s $size --trials $TRIALS >> "results/"$name"_gpu"
    done
    for size in $(seq $start 1 20);
        do python run.py -b $name -m bach -s $size --trials $TRIALS >> "results/"$name"_bach"
    done
}

# Clean
# =====
# rm results/*

# Run
# ===
blackscholes_cupy
blackscholes_torch
crime_index
haversine_cupy
haversine_torch
dbscan
tsvd
# pca

# Post-processing
# ===============
# cat results/blackscholes_torch_cpu | grep Med | grep Total | awk '{print $3}'
# OR ./parse.sh blackscholes_torch

# ./parse.sh blackscholes_cupy
# ./parse.sh blackscholes_torch
# ./parse.sh crime_index
# ./parse.sh haversine_cupy
# ./parse.sh haversine_torch
# ./parse.sh dbscan
# ./parse.sh tsvd
# ./parse.sh pca
