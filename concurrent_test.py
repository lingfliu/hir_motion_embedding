from utils import OrderedPool
import random
import time
import matplotlib.pyplot as plt


def task(x):
    print('submitted task', x)
    time.sleep(0.001)
    return x*x


if __name__ == '__main__':
    pool = OrderedPool(queue_max=10000)

    xs = [i for i in range(1000)]
    xs1 = [i for i in range(1000)]

    random.shuffle(xs)


    for i,x in enumerate(xs):
        pool.submit(task, x, (x,))

    pool.subscribe()

    results = pool.fetch_results()


    plt.plot(xs1, results)
    plt.show()
