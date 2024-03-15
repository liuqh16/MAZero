from collections import defaultdict
from queue import Queue
import numpy as np
import ray
from ray.util.queue import Queue as RemoteQueue


class QueueStorage(object):
    def __init__(self, threshold=15, size=20):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def get_len(self):
        return self.queue.qsize()


class RemoteQueueStorage(QueueStorage):

    def __init__(self, threshold=15, size=20):
        self.threshold = threshold
        self.queue = RemoteQueue(maxsize=size)


class SharedStorage(object):
    def __init__(self, model, target_model):
        """Shared storage for models and others
        Parameters
        ----------
        model: any
            models for self-play (update every checkpoint_interval)
        target_model: any
            models for reanalyzing (update every target_model_interval)
        """
        self.model = model
        self.model_index = 0
        self.target_model = target_model
        self.target_model_index = 0

        self.start = False
        self.step_counter = 0
        self.worker_logs = defaultdict(list)
        self.test_logs = defaultdict(float)

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        return (self.model_index, self.model.get_weights())

    def set_weights(self, step_count, weights):
        self.model_index = step_count
        return self.model.set_weights(weights)

    def get_target_weights(self):
        return (self.target_model_index, self.target_model.get_weights())

    def set_target_weights(self, step_count, weights):
        self.target_model_index = step_count
        return self.target_model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def add_worker_logs(self, log_dict: dict):
        for k, v in log_dict.items():
            self.worker_logs[k].append(v)

    def get_worker_logs(self):
        if self.worker_logs:
            worker_logs = self.worker_logs.copy()
            for k, v in self.worker_logs.items():
                if k == 'eps_len':
                    worker_logs['eps_len_max'] = np.max(v)
                elif k == 'eps_reward':
                    worker_logs['eps_reward_max'] = np.max(v)
                    worker_logs['eps_reward_min'] = np.min(v)
                    worker_logs['eps_reward_std'] = np.std(v)
                worker_logs[k] = np.mean(v)
            self.worker_logs.clear()
        else:
            worker_logs = None
        return worker_logs

    def add_test_logs(self, log_dict: dict):
        for k, v in log_dict.items():
            self.test_logs[k] = v

    def get_test_logs(self):
        if self.test_logs:
            test_logs = self.test_logs.copy()
            self.test_logs.clear()
        else:
            test_logs = None
        return test_logs


@ray.remote
class RemoteShareStorage(SharedStorage):

    def __init__(self, model, target_model):
        super().__init__(model, target_model)
