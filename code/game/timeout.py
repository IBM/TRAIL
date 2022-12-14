from functools import wraps
import errno
import os
import signal
import traceback
import sys
# from game.execute_episode import ProblemData
from torch.multiprocessing import Pool, TimeoutError

# hmm read this: https://anonbadger.wordpress.com/2018/12/15/python-signal-handlers-and-exceptions/

# class TimeoutError(Exception):
#     pass
#limitation: it expects the passed argument to have a field time_limit which is acheived in PlayData and ProblemData
def timeout(seconds, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            # if args[0] is isinstance(ProblemData):
            if args[0] is None:
                seconds = 30
            else:
                seconds = int(args[0].time_limit)
            print('******decorator: time limit set to ', seconds, '*****')
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

