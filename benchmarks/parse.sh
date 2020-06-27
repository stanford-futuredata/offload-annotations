workload=$1
echo $workload
for mode in cpu gpu bach;
do
    echo $mode
    cat "results/"$workload"_"$mode | grep Med | grep Total | awk '{print $3}'
done
