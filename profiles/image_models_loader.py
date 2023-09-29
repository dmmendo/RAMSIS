import torch
from PIL import Image
from torchvision import transforms

img_to_tensor = transforms.ToTensor()

def get_image_data_input(batch_size):
    tensor_list = []
    
    for i in range(batch_size):
        #img = Image.open("test_img.jpg")
        #tensor = torch.unsqueeze(img_to_tensor(img),dim=0)
        #torch.save(torch.ones(1,3,224,224),'test_img.pt')
        tensor = torch.load('test_img.pt')
        tensor_list.append(tensor)
    input_tensor = torch.cat(tensor_list)
    """
    torch.save(torch.ones(batch_size,3,224,224,dtype=torch.uint8),'test_img.pt')
    input_tensor = torch.load('test_img.pt')
    """
    return input_tensor

def load_efficientnetb0():
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb1():
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb2():
    from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb3():
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb4():
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb5():
    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B5_Weights.DEFAULT
    model = efficientnet_b5(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb6():
    from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B6_Weights.DEFAULT
    model = efficientnet_b6(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetb7():
    from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B7_Weights.DEFAULT
    model = efficientnet_b7(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetv2l():
    from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = efficientnet_v2_l(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetv2m():
    from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_V2_M_Weights.DEFAULT
    model = efficientnet_v2_m(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_efficientnetv2s():
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_googlenet():
    from torchvision.models import googlenet, GoogLeNet_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = GoogLeNet_Weights.DEFAULT
    model = googlenet(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_inceptionv3():
    from torchvision.models import inception_v3, Inception_V3_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_mobilenetv2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_mobilenetv3large():
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnext101_32x8d():
    from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNeXt101_32X8D_Weights.DEFAULT
    model = resnext101_32x8d(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnext101_64x4d():
    from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNeXt101_64X4D_Weights.DEFAULT
    model = resnext101_64x4d(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x05():
    from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = shufflenet_v2_x0_5(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x10():
    from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x15():
    from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X1_5_Weights.DEFAULT
    model = shufflenet_v2_x1_5(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x20():
    from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    model = shufflenet_v2_x2_0(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_googlenet_quantized():
    from torchvision.models.quantization import googlenet, GoogLeNet_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = GoogLeNet_QuantizedWeights.DEFAULT
    model = googlenet(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_inceptionv3_quantized():
    from torchvision.models.quantization import inception_v3, Inception_V3_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = Inception_V3_QuantizedWeights.DEFAULT
    model = inception_v3(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_mobilenetv2_quantized():
    from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = MobileNet_V2_QuantizedWeights.DEFAULT
    model = mobilenet_v2(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_mobilenetv3large_quantized():
    from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = MobileNet_V3_Large_QuantizedWeights.DEFAULT
    model = mobilenet_v3_large(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnext101_32x8d_quantized():
    from torchvision.models.quantization import resnext101_32x8d, ResNeXt101_32X8D_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNeXt101_32X8D_QuantizedWeights.DEFAULT
    model = resnext101_32x8d(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnext101_64x4d_quantized():
    from torchvision.models.quantization import resnext101_64x4d, ResNeXt101_64X4D_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNeXt101_64X4D_QuantizedWeights.DEFAULT
    model = resnext101_64x4d(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x05_quantized():
    from torchvision.models.quantization import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X0_5_QuantizedWeights.DEFAULT
    model = shufflenet_v2_x0_5(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x10_quantized():
    from torchvision.models.quantization import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X1_0_QuantizedWeights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x15_quantized():
    from torchvision.models.quantization import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X1_5_QuantizedWeights.DEFAULT
    model = shufflenet_v2_x1_5(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_shufflenetv2x20_quantized():
    from torchvision.models.quantization import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ShuffleNet_V2_X2_0_QuantizedWeights.DEFAULT
    model = shufflenet_v2_x2_0(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnet50_quantized():
    from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet50_QuantizedWeights.DEFAULT
    model = resnet50(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnet18_quantized():
    from torchvision.models.quantization import resnet18, ResNet18_QuantizedWeights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet18_QuantizedWeights.DEFAULT
    model = resnet18(weights=weights,quantize=True).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnet152():
    from torchvision.models import resnet152, ResNet152_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnet101():
    from torchvision.models import resnet101, ResNet101_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnet50():
    from torchvision.models import resnet50, ResNet50_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

def load_resnet18():
    from torchvision.models import resnet18, ResNet18_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess


def load_resnet34():
    from torchvision.models import resnet34, ResNet34_Weights
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights).to(device)

    # Set model to eval mode
    frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(model.eval())).to(device)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    return frozen_mod, preprocess

quantized_image_loader_dict = {
    "googlenet":load_googlenet_quantized, #quantized
    "inception_v3":load_inceptionv3_quantized, #quantized
    "mobilenet_v2":load_mobilenetv2,
    "mobilenet_v3_large":load_mobilenetv3large_quantized, #quantized
    "resnext101_32x8d":load_resnext101_32x8d_quantized, #quantized
    "resnext101_64x4d":load_resnext101_64x4d_quantized, #quantized
    "shufflenet_v2_x05":load_shufflenetv2x05_quantized, #quantized
    "shufflenet_v2_x10":load_shufflenetv2x10_quantized, #quantized
    "shufflenet_v2_x15":load_shufflenetv2x15_quantized, #quantized
    "shufflenet_v2_x20":load_shufflenetv2x20_quantized, #quantized
    "resnet18":load_resnet18_quantized, #quantized
    "resnet50":load_resnet50_quantized, #quantized
    "resnet34":load_resnet34,
    "resnet101":load_resnet101,
    "resnet152":load_resnet152,
    "efficientnet_v2l":load_efficientnetv2l,
    "efficientnet_v2m":load_efficientnetv2m,
    "efficientnet_v2s":load_efficientnetv2s,
    "efficientnet_b0":load_efficientnetb0,
    "efficientnet_b1":load_efficientnetb1,
    "efficientnet_b2":load_efficientnetb2,
    "efficientnet_b3":load_efficientnetb3,
    "efficientnet_b4":load_efficientnetb4,
    "efficientnet_b5":load_efficientnetb5,
    "efficientnet_b6":load_efficientnetb6,
    "efficientnet_b7":load_efficientnetb7,
}

image_loader_dict = {
    "googlenet":load_googlenet, 
    "inception_v3":load_inceptionv3, 
    "mobilenet_v2":load_mobilenetv2,
    "mobilenet_v3_large":load_mobilenetv3large, 
    "resnext101_32x8d":load_resnext101_32x8d, 
    "resnext101_64x4d":load_resnext101_64x4d, 
    "shufflenet_v2_x05":load_shufflenetv2x05, 
    "shufflenet_v2_x10":load_shufflenetv2x10, 
    "shufflenet_v2_x15":load_shufflenetv2x15, 
    "shufflenet_v2_x20":load_shufflenetv2x20, 
    "resnet18":load_resnet18, 
    "resnet50":load_resnet50, 
    "resnet34":load_resnet34,
    "resnet101":load_resnet101,
    "resnet152":load_resnet152,
    "efficientnet_v2l":load_efficientnetv2l,
    "efficientnet_v2m":load_efficientnetv2m,
    "efficientnet_v2s":load_efficientnetv2s,
    "efficientnet_b0":load_efficientnetb0,
    "efficientnet_b1":load_efficientnetb1,
    "efficientnet_b2":load_efficientnetb2,
    "efficientnet_b3":load_efficientnetb3,
    "efficientnet_b4":load_efficientnetb4,
    "efficientnet_b5":load_efficientnetb5,
    "efficientnet_b6":load_efficientnetb6,
    "efficientnet_b7":load_efficientnetb7,
}
