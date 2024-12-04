# Learning-Rate-Quantization

Included some resnet34 and resnet50 notebooks for additional training if necessary


To run the code in Terminal ```cd src``` then run ```python main.py -model {model architecture u would like to use} -b 4 -bs 256 -s 1.16 -ds CIFAR10 -wf {weights file you would like to quantize}```

docker run -it -v C:\Users\daniel\dsc180z\Learning-Rate-Quantization\logs:/app/logs quant_nnets python main.py -model resnet18 -ds 'CIFAR10' -wf 'resnet18_lr_0.01.pth' -dc true
