import tensorflow as tf

class Ann:

    def __init__(self, X):
        self.X = X
        pass

    def affine(self):
        print('aa')
        print(self.X)
        pass


if __name__ == '__main__':
    X = tf.Variable([[1.0, 2.0]], name='X')
    obj = Ann(X)
    obj.affine()
