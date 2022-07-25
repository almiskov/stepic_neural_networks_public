from network import Network
import numpy as np
import doctest


def sample_backprop1():
    """
      Пример тестового случая. Создает сеть, запускает функцию backprop и
      проверяет возвращенные значения.

      >>> nabla_b, nabla_w = sample_backprop1()
      >>> print(nabla_b[0])
      [[ 0.00214254]
       [-0.05287709]]
      >>> print(nabla_w[0])
      [[ 0.00214254  0.00428509  0.00642763]
       [-0.05287709 -0.10575419 -0.15863128]]
    """
    nn = Network([3, 2])
    nn.biases = [np.array([[-1], [-1]])]
    nn.weights = [np.array([[-1, 1, -1], [1, -1, 1]])]
    x = np.array([[1], [2], [3]])
    y = np.array([[0], [1]])

    return nn.backprop(x, y)


doctest.run_docstring_examples(sample_backprop1, globals(), verbose=True)
