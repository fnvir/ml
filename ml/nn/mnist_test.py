import numpy as np
from neural_network import NeuralNetwork
from loss_functions import MSE,CrossEntropy


def main():
    x_train,x_test,y_train,y_test=preprocess(*load_mnist())

    print(x_train.shape,y_train.shape)
    print(x_test.shape,y_test.shape)
    print(x_test[0].shape,y_train[0].shape)

    import time

    st=time.time()
    nn=NeuralNetwork((28*28,40,10),MSE)
    nn.gradient_descent_stochastic(x_train,y_train,max_iter=140,show=True,regularization=0.01,batch_size=1000)
    print('\n',time.time()-st,'s\n')

    
    cnt=0
    for i in range(len(x_test)):
        z=nn.predict(x_test[i])
        if np.argmax(z)==np.argmax(y_test[i]): cnt+=1

    print(f'{cnt} / {len(x_test)}  -->  accuracy: {100*cnt/len(x_test)} %')




def preprocess(x_train,x_test,y_train,y_test):
    return x_train/255,x_test/255,np.eye(10)[y_train].reshape(-1,10,1),np.eye(10)[y_test].reshape(-1,10,1)

def load_mnist():
    import zipfile,gzip

    data_sources = {
        "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
        "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
        "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
        "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
    }
    mnist_dataset = {}

    zf=zipfile.ZipFile('mnist.zip','r')
    # Images
    for key in ("training_images", "test_images"):
        with gzip.open(zf.open(data_sources[key]),"rb") as mnist_file:
            mnist_dataset[key] = np.frombuffer(
                mnist_file.read(), np.uint8, offset=16
            ).reshape(-1,28*28,1)
    # Labels
    for key in ("training_labels", "test_labels"):
        with gzip.open(zf.open(data_sources[key]),"rb") as mnist_file:
            mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)
    x_train, y_train, x_test, y_test = (
        mnist_dataset["training_images"],
        mnist_dataset["training_labels"],
        mnist_dataset["test_images"],
        mnist_dataset["test_labels"],
    )
    return x_train,x_test,y_train,y_test


if __name__=='__main__':
    main()

