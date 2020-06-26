
import argparse
import pandas as pd
import sys
import time

def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[0:1000]

def analyze(top1000):
    start1 = time.time()
    all_names = pd.Series(top1000.name.unique())
    lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
    filtered = top1000[top1000.name.isin(lesley_like)]
    table = filtered.pivot_table('births', index='year',
                                 columns='sex', aggfunc='sum')

    table = table.div(table.sum(1), axis=0)
    end1 = time.time()
    print("Analysis:", end1 - start1)
    return table

def run(filename):
    years = range(1880, 2011)
    pieces = []
    columns = ['year', 'sex', 'name', 'births']

    start = time.time()
    sys.stdout.write("Reading data...")
    sys.stdout.flush()
    names = pd.read_csv(filename, names=columns)
    init_time = time.time() - start
    print("done:", init_time)
    sys.stdout.flush()

    print("Size of names:", len(names))

    e2e_start = time.time()

    # Time preprocessing step
    start0 = time.time()
    grouped = names.groupby(['year', 'sex'])
    end0 = time.time()
    print("GroupBy:", end0 - start0)
    start0 = end0

    top1000 = grouped.apply(get_top1000)
    top1000.reset_index(inplace=True, drop=True)

    end0 = time.time()
    print("Apply:", end0-start0)
    print("Elements in top1000:", len(top1000))

    result = analyze(top1000)

    e2e_end = time.time()
    print("Total time:", e2e_end - e2e_start)

    print(top1000['births'].sum())
    return init_time, e2e_end - e2e_start

def median(arr):
    arr.sort()
    m = int(len(arr) / 2)
    if len(arr) % 2 == 1:
        return arr[m]
    else:
        return (arr[m] + arr[m-1]) / 2

def main():
    parser = argparse.ArgumentParser(
        description="Birth Analysis."
    )
    parser.add_argument('-f', "--filename", type=str, default="../datasets/birth_analysis/_data/babynames.txt", help="Input file")
    parser.add_argument('--trials', type=int, default=1, help='Number of trials.')
    args = parser.parse_args()

    filename = args.filename

    print("File:", filename)
    init_times = []
    runtimes = []
    for _ in range(args.trials):
        init_time, runtime = run(filename)
        init_times.append(init_time)
        runtimes.append(runtime)
    if args.trials > 1:
        print('Median Init:', median(init_times))
        print('Median Runtime:', median(runtimes))
        print('Median Total:', median(init_times) + median(runtimes))


if __name__ == "__main__":
    main()
