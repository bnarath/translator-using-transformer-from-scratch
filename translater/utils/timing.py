import time
from functools import wraps


def timeit(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = method(self, *args, **kwargs)
        end_time = time.time()
        self.execution_time = end_time - start_time
        return result

    return wrapper
