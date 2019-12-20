
import sys

sys.path.append("../../lib/")
sys.path.append("../../pycomposer/")

import argparse
import sa.annotated.pandas as pd
import time
from sa.annotation import Backend

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

def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[0:1000]

def run(filename, threads, batch_size, force_cpu):
    years = range(1880, 2011)
    columns = ['year', 'sex', 'name', 'births']

    sys.stdout.write("Reading data...")
    sys.stdout.flush()
    names = pd.read_csv(filename, names=columns)
    print("done")

    print("Size of names:", len(names))

    e2e_start = time.time()

    start0 = time.time()
    grouped = pd.dfgroupby(names, ['year', 'sex'])
    top1000 = pd.gbapply(grouped, get_top1000)
    top1000.dontsend = False
    pd.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    top1000 = top1000.value
    top1000.reset_index(inplace=True, drop=True)
    print(len(top1000))

    """
    grouped: Dag Operation
    GBApply Takes a DAG operation and stores it in its type. The operation must be a GroupBy
    GBApply has type ApplySplit. It's combiner:
        1. Combines the results of the dataFrame
        2. Resets index
        3. Gets the keys from the DAG operation
        4. Calls groupBy again
        5. Calls apply again.
    """

    localreduce_start = time.time()
    top1000 = top1000.groupby(['year', 'sex']).apply(get_top1000)
    localreduce_end = time.time()
    print("Local reduction:", localreduce_end - localreduce_start)
    top1000.reset_index(inplace=True, drop=True)
    end0 = time.time()

    print("Apply:", end0-start0)
    print("Elements in top1000:", len(top1000))

    result = analyze(top1000)

    e2e_end = time.time()
    print("Total time:", e2e_end - e2e_start)

    print(top1000['births'].sum())

def main():
    parser = argparse.ArgumentParser(
        description="Birth Analysis with Composer."
    )
    parser.add_argument('-f', "--filename", type=str, default="../datasets/birth_analysis/_data/babynames.txt", help="Input file")
    parser.add_argument('-t', "--threads", type=int, default=1, help="Number of threads.")
    parser.add_argument('-cpu', "--cpu_piece_size", type=int, default=14, help="Log size of each CPU piece.")
    parser.add_argument('-gpu', "--gpu_piece_size", type=int, default=19, help="Log size of each GPU piece.")
    parser.add_argument('--force_cpu', action='store_true', help='Whether to force composer to execute CPU only.')
    args = parser.parse_args()

    filename = args.filename
    threads = args.threads
    cpu_piece_size = 1<<args.cpu_piece_size
    gpu_piece_size = 1<<args.gpu_piece_size
    batch_size = {
        Backend.CPU: cpu_piece_size,
        Backend.GPU: gpu_piece_size,
    }
    force_cpu = args.force_cpu

    print("File:", filename)
    print("Threads:", threads)
    print("GPU Piece Size:", gpu_piece_size)
    print("CPU Piece Size:", cpu_piece_size)
    mi = run(filename, threads, batch_size, force_cpu)


if __name__ == "__main__":
    main()
