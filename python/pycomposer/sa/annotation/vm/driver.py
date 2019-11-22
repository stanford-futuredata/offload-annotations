
from collections import defaultdict

import logging
import torch.multiprocessing as multiprocessing

import threading
import time

STOP_ITERATION = "stop"

# Global reference to values. These should be in read-only shared memory with
# all child processes.
_VALUES = None
# Global reference to program currently being executed.
_PROGRAM = None
# Batch size to use.
_BATCH_SIZE = None

# Size of the L2 Cache (TODO read this from somewhere)
CACHE_SIZE = 252144

def _worker(worker_id, index_range, max_batch_size):
    """
    A multiprocessing worker.

    The index indicates the "thread ID" of the worker, and the queue is a
    channel with which the driver program sends this worker programs to
    execute.

    Parameters
    ----------

    worker_id : the thread ID of this worker.
    index_range : A range
    and the master.
    max_batch_size: the maximum batch size in the program.

    """
    context = defaultdict(list)
    _run_program(worker_id, index_range, context, max_batch_size, worker_id)
    return context

def _run_program(
    worker_id,
    index_range,
    context,
    batch_size: int,
    batch_index: int = 0,
    initial_i: int = 0,
    replace_original: bool = False,
):
    """Runs the global program to completion and return partial values.

    Parameters
    ----------

    worker_id : the ID of this worker.
    program : the program to execute.
    batch_size : the current batch size of the program.
    batch_index : the index of the current split batch.
    initial_i : the index of the instruction to start execution at.
    replace_original: boolean
        Whether to replace the object at a value's original pointer with the
        merged object. Typically if the merge that occurs at the end is a
        top-level merge.
    """
    global _VALUES
    global _PROGRAM
    global _BATCH_SIZE

    # logging.debug("Thread", worker_id, "range:", index_range, "batch size:", _BATCH_SIZE)
    from ..backend import Backend
    print("Thread {} range: {} batch size: {} instruction: {} replace_original: {}".format(
        worker_id, index_range, batch_size, initial_i, replace_original))
    start = time.time()

    just_parallel = False
    if just_parallel:
        _BATCH_SIZE = { Backend.CPU: index_range[1] - index_range[0] }
        piece_start = index_range[0]
        piece_end = index_range[1]
    else:
        piece_start = index_range[0]
        piece_end = piece_start + batch_size

    _PROGRAM.set_range_end(index_range[1])

    early_exit_i = set()
    batch_subindex = 0
    while piece_start < index_range[1]:
        # There are three possible scenarios for executing a program instruction
        # with regards to batch size.
        #
        # (1) The batch size stays the same: Execute instructions until the program ends or the
        #     batch size changes.
        # (2) The batch size decreases: Pipeline the program within the current index range on
        #     the smaller index subrange until the batch size increases again. Then resume at
        #     that instruction at the current batch size.
        # (3) The batch size increase: Exit early.
        #
        # Repeat the pipeline for each index subrange.
        i = initial_i
        while i < len(_PROGRAM.insts):
            inst = _PROGRAM.insts[i]
            from .instruction import To
            if isinstance(inst, To) or inst.batch_size == batch_size:
                print('EVALUATE ' + str(inst))
                result = inst.evaluate(worker_id, batch_subindex, _VALUES, context)
                i += 1
                if isinstance(result, str) and result == STOP_ITERATION:
                    break
            elif batch_size > inst.batch_size:
                index_subrange = (piece_start, piece_end)
                i = _run_program(
                    worker_id,
                    index_subrange,
                    context,
                    inst.batch_size,
                    batch_subindex,
                    initial_i=i,
                )
                continue
            else:
                early_exit_i.add(i)
                break

        piece_start += batch_size
        piece_end += batch_size

        # Clamp to the range assigned to this thread.
        if piece_end > index_range[1]:
            piece_end = index_range[1]

    process_end = time.time()

    # Free non-shared memory on this worker.
    # Replace the data in the original pointer if we are the top level thread.
    # Merge the data if we are the top level thread or are returning execution to the top level.
    if replace_original:
        _merge(_PROGRAM, context, replace_original=replace_original)

    merge_end = time.time()

    logging.debug("Thread {}\t processing: {:.3f}\t merge: {:.3f}\t total:{:.3f}\t".format(
            worker_id,
            process_end - start,
            merge_end - process_end,
            merge_end - start))

    if len(early_exit_i) == 0:
        return i
    else:
        assert len(early_exit_i) == 1
        return early_exit_i.pop()

def _merge(program, context, replace_original):
    """
    Merge a context that was generated with the given program.

    Parameters
    ----------

    program : The executed program instructions.
    context : The context of program values at the end of the execution.
    replace_original : boolean
        Whether to replace the object at a value's original pointer with the
        merged object. Necessary if the object wasn't operated on shared memory.
    """
    merged = set()
    for inst in reversed(program.insts):
        # Reverse list to obtain the last split type assigned to each target.
        # For now, given the way pipelines are set up, this shouldn't matter
        # since the split type of each target won't change.
        if inst.target in merged or inst.target is None:
            continue
        else:
            merged.add(inst.target)
            if inst.ty is not None:
                if inst.ty.mutable:
                    from .. import dag
                    if isinstance(_VALUES[inst.target], dag.Operation) or not replace_original:
                        context[inst.target] = inst.ty.combine(context[inst.target])
                    else:
                        # Since we operated on a copy of the original data, we need to
                        # replace the original pointer with the new data in combine()
                        original = _VALUES[inst.target]
                        context[inst.target] = inst.ty.combine(context[inst.target], original=original)
                else:
                    # No need to merge values and send the result back: it's immutable,
                    # and should not have changed on the master process.
                    context[inst.target] = None

class Driver:
    """
    Parallel driver and scheduler for the virtual machine.
    """

    __slots__ = [ "workers", "batch_size", "optimize_single", "profile" ]

    def __init__(self, workers, batch_size, optimize_single=True, profile=False):
        self.workers = workers
        self.batch_size = batch_size
        self.optimize_single = optimize_single
        self.profile = profile
        if self.profile:
            assert self.workers == 1, "Profiling only supported on single thread"
            assert self.optimize_single, "Profiling only supported with optimize_single=True"

    def get_partitions(self, total_elements):
        """ Returns a list of index ranges to process for each worker. """
        ranges = []
        for tid in range(self.workers):
            elements = total_elements // self.workers
            if elements == 0 and tid != 0:
                ranges.append(None)
                continue

            if elements == 0:
                # Thread 0
                elements = total_elements
            else:
                # Round up
                elements = total_elements //\
                        self.workers + int(total_elements % self.workers != 0)

            thread_start = elements * tid
            thread_end = min(total_elements, elements * (tid + 1))
            ranges.append((thread_start, thread_end))

        return ranges

    def run(self, program, backends, values):
        """ Executes the program with the provided values. """
        elements = program.elements(values)
        ranges = self.get_partitions(elements)

        # Make the values accessible to child processes.
        global _VALUES
        global _PROGRAM
        global _BATCH_SIZE

        _VALUES = values
        _PROGRAM = program
        _BATCH_SIZE = dict([(backend, self.batch_size[backend]) for backend in backends])

        max_batch_size = max(_BATCH_SIZE.values())
        result = defaultdict(list)
        if self.workers == 1 and self.optimize_single:
            _run_program(0, ranges[0], result, max_batch_size, replace_original=True)
        elif self.workers > 1 and ranges[1] is None:
            # We should really dynamically adjust the number of processes
            # (i.e., self.workers should be the maximum allowed workers), but
            # for now its 1 or all to make evaluation easier.
            _run_program(0, ranges[0], result, max_batch_size, replace_original=True)
        else:
            # This needs to go after the assignment to _VALUES, so
            # the process snapshot sees the updated variable. The advantage of
            # this approach is copy-on-write semantics on POSIX systems for
            # (potentially large) inputs. I'm not sure what the Python
            # interpreter does, but assuming it's sane, the underlying objects
            # should never be written to, hence preventing the copy. The big
            # disadvantage of this approach is that we need to incur a
            # process-start overhead every time...
            pool = multiprocessing.Pool(self.workers)

            # TODO Just use Pool.imap instead?
            partial_results = []
            for (i, index_range) in enumerate(ranges):
                # import pdb; pdb.set_trace()
                partial_results.append(pool.apply_async(_worker, args=(i, index_range, max_batch_size)))
            for i in range(len(partial_results)):
                partial_results[i] = partial_results[i].get()

            result = defaultdict(list)
            start = time.time()
            if self.workers > 1:
                for partial_result in partial_results:
                    if partial_result is not None:
                        for (key, value) in partial_result.items():
                            # Don't add unnecessary None values.
                            if value is not None:
                                result[key].extend(value)

                _merge(program, result, replace_original=True)

                # Reinstate non-mutable values, broadcast values, etc.
                for value_key in _VALUES:
                    if value_key not in result:
                        result[value_key] = _VALUES[value_key]
            else:
                result = partial_results[0]

            end = time.time()
            logging.debug("Final merge time:", end - start)
            pool.terminate()

        _VALUES = None
        _PROGRAM = None
        _BATCH_SIZE = None

        return result

