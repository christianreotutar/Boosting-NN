import mnist_loader
from network import Network

training,validation,testing=mnist_loader.load_data_wrapper()
net = Network([784, 25, 10])
res = net.SGD(training, 20, 15, 2.5, testing)
print(res)
