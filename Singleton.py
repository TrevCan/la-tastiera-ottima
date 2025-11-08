# Source - https://stackoverflow.com/questions/50566934/why-is-this-singleton-implementation-not-thread-safe
# https://stackoverflow.com/a/55629949
# Posted by se7entyse7en
# Retrieved 2025-11-05, License - CC BY-SA 4.0
# 
# def synchronized(lock)
# https://stackoverflow.com/revisions/0ce4c0aa-c6b1-4086-a224-3181cddfd9f5/view-source

import functools
import threading

lock = threading.Lock()


def synchronized(lock):
    """ Synchronization decorator """
    def wrapper(f):
        @functools.wraps(f)
        def inner_wrapper(*args, **kw):
            with lock:
                return f(*args, **kw)
        return inner_wrapper
    return wrapper

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._locked_call(*args, **kwargs)
        return cls._instances[cls]

    @synchronized(lock)
    def _locked_call(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

#class SingletonClassOptimized(metaclass=Singleton):
#    pass
