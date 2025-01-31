import numpy as np
from pathlib import WindowsPath
import random
from network import Network


p = WindowsPath()

names = ['velocity', 'angle', 'distance', 'chet']


data = np.loadtxt(
    f'{p.cwd()}\HW_2\data.csv',
    delimiter=','
)

# во входных данный очень большой разброс признаков, поэтому их надо нормализовать -
# из значения вычитаем среднее арифметическое по столбцу и делим на стандартное отклонение по этому столбцу
# категориальные переменные (последний столбец) нормализовывать не надо
#
# если я правильно понял, то для него будет применено one-hot encoding
# то есть категории 0, 1, 2 должны превратиться в векторы [1, 0, 0].T, [0, 1, 0].T, [0, 0, 1].T
# и судя по всему у нейронки в выходном слое должно быть три нейрона
# это пока только предположения, TODO: дальше разберёмся
#


def normalize(data):
    means = data.mean(axis=0)
    means[-1] = 0
    stds = data.std(axis=0)
    stds[-1] = 1

    return (data - means) / stds


nn = Network([3, 2])
nn.biases = [np.array([[-1], [-1]])]
nn.weights = [np.array([[-1, 1, -1], [1, -1, 1]])]
x = np.array([[1], [2], [3]])
y = np.array([[0], [1]])

# nn.backprop(x, y)

a = np.array([1, 2, 3, 1, 2, 3]).reshape(3, 2)
print(a.shape)
a = a[:, np.newaxis]
print(a.shape)
