import time
from functools import wraps

class decorators:
    @staticmethod
    def running_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            START_time = time.time()
            print(f"\033[94m{func.__name__} started on {time.ctime(START_time)}. \033[0m")
            result = func(*args, **kwargs)
            END_time = time.time()
            print(f"\033[92m{func.__name__} ended on {time.ctime(END_time)} "
                  f"({round(END_time - START_time, 2)}s).\033[0m")
            return result
        return wrapper
