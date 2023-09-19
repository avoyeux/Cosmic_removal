import multiprocessing as mp
import numpy as np

def multiprocessing_fun():
    pool = mp.Pool(processes=64)
    args = [(useless, ) for useless in np.arange(1, 80)]
    pool.starmap(functioning, args)
    pool.close()
    pool.join()

def functioning(useless):
    print('yes')
    

if __name__ == '__main__':
    multiprocessing_fun()
