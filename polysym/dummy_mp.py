import multiprocessing
from contextlib import contextmanager

class DummyPool:
    def map(self, func, iterable):
        # fall back to the builtin map if only one worker
        return list(map(func, iterable))
    def imap(self, func, iterable):
        return map(func, iterable)
    def close(self): pass
    def join(self):  pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

@contextmanager
def get_pool(workers):
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            yield pool
    else:
        # yield a dummy that behaves like Pool
        yield DummyPool()
