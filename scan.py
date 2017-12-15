import numpy as np
import tensorflow as tf


def show(t):
    v = t.eval()
    print(type(t), type(v), v)


elems = np.array([1, 2, 3, 4, 5, 6])
sum2 = tf.scan(lambda a, x: a + x, elems)
# sum == [1, 3, 6, 10, 15, 21]

elems = np.array([1, 2, 3, 4, 5, 6])
initializer = np.array(0)
sum_one = tf.scan(lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
# sum_one == [1, 2, 3, 4, 5, 6]

elems = np.array([1, 0, 0, 0, 0, 0])
initializer = (np.array(0), np.array(1))
fibonaccis = tf.scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
# fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])

sess = tf.Session()
with sess.as_default():
    show(sum2)
    show(sum_one)
    for fib in fibonaccis:
        show(fib)
