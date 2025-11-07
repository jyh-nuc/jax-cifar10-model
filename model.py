from flax import nnx
import jax.numpy as jnp

class CNN(nnx.Module):
    def __init__(self, num_classes=10, rngs=None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="same", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="same", rngs=rngs)
        self.fc1 = nnx.Linear(64 * 8 * 8, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, num_classes, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, (2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, (2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.fc1(x))
        x = self.fc2(x)
        return x
