# tools
Self-built tools for research

## pytorch related

### Split\_layer
This is a simple package to print each layer forward and backward time of PyTorch models.

#### Usage

You can use this tool by three steps:

0. Install split_layer by running `pip3 install split_layer -U --user`

1. Find the file which defines the structure of a Network. Add the following code:
  ```
  from split_layer import split_layer_dec
  @split_layer_dec(__file__)
  class Net():
  ```
  **Notice:** Make sure the forward function input parameter and itermediate output should be `x`. For example:
	```
  x = F.relu(self.conv1(x))
  return x
  ```
2. Replace `loss.backward()` with something like `net.backward(outputs)`. Then you can run your training code as usual.

#### Others

+ It is built according to the accepted answer of this [question](https://discuss.pytorch.org/t/how-to-split-backward-process-wrt-each-layer-of-neural-network/7190'). Now, it is not flexible enough, and **DO NOT support DP or DDP models**. We will develop it further in the future.

+ It works both on CPU and GPU.

#### Requirements

Make sure `inspect` and `torch` has been installed.


