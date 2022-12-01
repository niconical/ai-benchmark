[AI Benchmark Alpha](http://ai-benchmark.com/alpha) is an open source python library for evaluating AI performance of various hardware platforms, including CPUs, GPUs and TPUs. The benchmark is relying on [TensorFlow](https://www.tensorflow.org) machine learning library, and is providing a lightweight and accurate solution for assessing inference and training speed for key Deep Learning models.</br></br>

In total, AI Benchmark consists of <b>42 tests</b> and <b>19 sections</b> provided below:</br>

1. MobileNet-V2&nbsp; `[classification]`
2. Inception-V3&nbsp; `[classification]`
3. Inception-V4&nbsp; `[classification]`
4. Inception-ResNet-V2&nbsp; `[classification]`
5. ResNet-V2-50&nbsp; `[classification]`
6. ResNet-V2-152&nbsp; `[classification]`
7. VGG-16&nbsp; `[classification]`
8. SRCNN 9-5-5&nbsp; `[image-to-image mapping]`
9. VGG-19&nbsp; `[image-to-image mapping]`
10. ResNet-SRGAN&nbsp; `[image-to-image mapping]`
11. ResNet-DPED&nbsp; `[image-to-image mapping]`
12. U-Net&nbsp; `[image-to-image mapping]`
13. Nvidia-SPADE&nbsp; `[image-to-image mapping]`
14. ICNet&nbsp; `[image segmentation]`
15. PSPNet&nbsp; `[image segmentation]`
16. DeepLab&nbsp; `[image segmentation]`
17. Pixel-RNN&nbsp; `[inpainting]`
18. LSTM&nbsp; `[sentence sentiment analysis]`
19. GNMT&nbsp; `[text translation]`

For more information and results, please visit the project website: [http://ai-benchmark.com/alpha](http://ai-benchmark.com/alpha)</br></br>

#### Installation Instructions </br>

The benchmark requires TensorFlow machine learning library to be present in your system.

On systems that <b>do not have Nvidia GPUs</b>, run the following commands to install AI Benchmark:

```bash
pip install tensorflow
pip install ai-benchmark
```
</br>

If you want to check the <b>performance of Nvidia graphic cards</b>, run the following commands:

```bash
pip install tensorflow-gpu
pip install ai-benchmark
```
</br>

If you want to check the <b>performance of Nvidia graphic cards</b> on <b>aarch64</b> platform, you should install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and use [NVIDIA NGC container](https://www.nvidia.com/en-us/gpu-cloud/).

<b>`Note 1:`</b> If Tensorflow is already installed in your system, you can skip the first command.

<b>`Note 2:`</b> For running the benchmark on Nvidia GPUs, <b>`NVIDIA CUDA`</b> and <b>`cuDNN`</b> libraries should be installed first. Please find detailed instructions [here](https://www.tensorflow.org/install/gpu). </br></br>

#### Getting Started </br>

To run AI Benchmark, use the following code:

```bash
from ai_benchmark import AIBenchmark
benchmark = AIBenchmark()
results = benchmark.run()
```

Alternatively, on Linux systems you can type `ai-benchmark` in the command line to start the tests.

To run inference or training only, use `benchmark.run_inference()` or `benchmark.run_training()`. </br></br>

To run on the aarch64 platform, use the following code:
```bash
DLFRAMEWORK=tf2 make images
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm ai-benchmark-tf2:bf0f727
```

Currently, we support `PyTorch`,`TensorFlow 1.x`and`TensorFlow 2.x` framworks, two cpu arch(`arm64` and `amd64`).`bf0f727` is the container tag when you build image.
 

#### Advanced settings </br>

```bash
AIBenchmark(use_CPU=None, verbose_level=1):
```
> use_CPU=`{True, False, None}`:&nbsp;&nbsp; whether to run the tests on CPUs&nbsp; (if tensorflow-gpu is installed)

> verbose_level=`{0, 1, 2, 3}`:&nbsp;&nbsp; run tests silently | with short summary | with information about each run | with TF logs

```bash
benchmark.run(precision="normal"):
```

> precision=`{"normal", "high"}`:&nbsp;&nbsp; if `high` is selected, the benchmark will execute 10 times more runs for each test.

</br>

### Additional Notes and Requirements </br>

GPU with at least 2GB of RAM is required for running inference tests / 4GB of RAM for training tests.

### TODO
[ ] PyTorch support.