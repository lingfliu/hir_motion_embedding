import multiprocessing
from queue import Empty


def tuple_sort(list):
    return [x[1] for x in sorted(list, key=lambda x: x[0])]


def _tag_task(idx, task, params, queue):
    # print('task param=', params)
    result = task(*params)
    queue.put((idx, result))


"""
支持任务排序的多进程池
"""
class OrderedPool:
    def __init__(self, num_procs=4, queue_max=5000):
        cpu_count = multiprocessing.cpu_count()
        num_procs = num_procs if num_procs < cpu_count else cpu_count
        self.num_procs = int(num_procs)
        self.task_pool = multiprocessing.Pool(processes=self.num_procs)
        self.queue = multiprocessing.Manager().Queue(maxsize=queue_max)

    def submit(self, task, order, params):
        self.task_pool.apply_async(_tag_task, (order, task, params, self.queue))

    def subscribe(self):
        self.task_pool.close()
        # blocking call
        self.task_pool.join()
        return

    def fetch_results(self):
        q = self.queue
        results = []
        try:
            r = q.get_nowait()
            while r is not None:
                results.append(r)
                r = q.get_nowait()
        except Empty:
            if len(results) > 0:
                return tuple_sort(results)
            else:
                return []

    def cleanup(self):
        # self.queue = None .clear()
        self.queue = None
        self.task_pool.close()
        self.task_pool = multiprocessing.Pool(processes=self.num_procs)
