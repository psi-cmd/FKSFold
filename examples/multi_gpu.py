import os
import torch
import multiprocessing as mp
from functools import wraps

def _worker_fn(fn, gpu_id, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"gpu {gpu_id} is used")
    torch.cuda.set_device(0)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, arg = item
        try:
            result = fn(arg)
            result_queue.put((idx, result))
        except Exception as e:
            result_queue.put((idx, e))

def gpu_map(fn):
    @wraps(fn)
    def wrapper(arg_list):
        n_gpus = torch.cuda.device_count()
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # 启动每个 GPU 上的 worker 进程
        processes = []
        for i in range(n_gpus):
            p = mp.Process(target=_worker_fn, args=(fn, i, task_queue, result_queue))
            p.start()
            processes.append(p)

        # 分配任务（轮询式）
        for i, arg in enumerate(arg_list):
            task_queue.put((i, arg))

        # 发送结束信号
        for _ in range(n_gpus):
            task_queue.put(None)

        # 收集结果
        results = [None] * len(arg_list)
        for _ in range(len(arg_list)):
            idx, res = result_queue.get()
            results[idx] = res

        for p in processes:
            p.join()

        return results
    return wrapper

def gpu_map_debug(single_parameter):
    """debug the single parameter in this thread"""
    def wrapper(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            return f(single_parameter)
        return wrapped_f
    return wrapper