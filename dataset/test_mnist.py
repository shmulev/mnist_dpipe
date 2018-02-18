from mnist import MNIST


def test():
    mnist = MNIST('../../MNIST_data/')

    print(len(mnist.ids))


if __name__ == '__main__':
    test()
