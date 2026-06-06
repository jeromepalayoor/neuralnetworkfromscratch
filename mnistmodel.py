from neuralnetwork import *
import numpy as np
from scipy.ndimage import rotate, shift, zoom
import pickle

with np.load('mnist.npz') as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

x_train= x_train.reshape(60000, -1) / 255.0
x_test= x_test.reshape(10000, -1) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# augment images so more accurate model
def augment_image(image_784):
    img_2d = image_784.reshape(28, 28)
    scale_factor = np.random.uniform(0.75, 1.20)
    zoomed_img = zoom(img_2d, scale_factor, order=1)
    h, w = zoomed_img.shape
    if scale_factor < 1.0:
        pad_y = (28 - h) // 2
        pad_x = (28 - w) // 2
        img_2d = np.zeros((28, 28))
        img_2d[pad_y:pad_y+h, pad_x:pad_x+w] = zoomed_img
    else:
        start_y = (h - 28) // 2
        start_x = (w - 28) // 2
        img_2d = zoomed_img[start_y:start_y+28, start_x:start_x+28]
    random_angle = np.random.uniform(-15, 15)
    img_2d = rotate(img_2d, random_angle, reshape=False, order=1)
    random_shift_x = np.random.uniform(-2, 2)
    random_shift_y = np.random.uniform(-2, 2)
    img_2d = shift(img_2d, shift=[random_shift_y, random_shift_x], order=1)
    return img_2d.reshape(784)

nn = NeuralNetwork([
    DenseLayer(input_dim=784, output_dim=512),
    ReLULayer(),
    DenseLayer(input_dim=512, output_dim=10),
    SoftmaxLayer()
])

loss_fn = CrossEntropyLoss()

epochs = 20
batch_size = 128
lr = 0.01

for epoch in range(1, epochs+1):
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)

    total_loss = 0
    num_batches = x_train.shape[0] / batch_size

    for i in range(0, x_train.shape[0], batch_size):
        batch_indices = indices[i:i+batch_size]
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]

        augmented_x_batch = np.zeros_like(x_batch)
        for idx in range(x_batch.shape[0]):
            augmented_x_batch[idx] = augment_image(x_batch[idx])
        
        y_pred = nn.forward(augmented_x_batch)
        
        loss = loss_fn.forward(y_pred, y_batch)
        total_loss += loss
        
        loss_gradient = loss_fn.backward()
        nn.backward(loss_gradient)
        nn.update_params(lr)
        
    average_epoch_loss = total_loss / (x_train.shape[0] / batch_size)
    print(f"Epoch {epoch} completed | Average Loss: {average_epoch_loss:.4f}")

test_preds = nn.forward(x_test)
final_acc = accuracy(test_preds, y_test)
print(f"Final Test Accuracy: {final_acc:.2f}%")

with open('mnist_nn_model.pkl', 'wb') as f:
    pickle.dump(nn, f)