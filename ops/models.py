# model
import networks
from torchsummary import summary

def load_model(model_name,in_channels,num_classes,aux_layer,device,sum_on):
    if model_name=="VGG11" or model_name=="VGG13" or model_name=="VGG16" or model_name=="VGG19":
        model = networks.vgg.VGGnet(model_name, in_channels=in_channels, num_classes=num_classes, init_weights=True).to(device)
    elif model_name=="InceptionV1":
        model = networks.InceptionV1.InceptionV1(in_channels=in_channels, num_classes=num_classes, init_weights=True, aux_layer=aux_layer).to(device)
    elif model_name=="InceptionV2":
        model = networks.InceptionV2.InceptionV2(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="InceptionV3":
        # the test size is 299
        model = networks.InceptionV3.InceptionV3(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="InceptionResNet":
        model = networks.InceptionV4.InceptionResNetV2(A=10, B=20, C=10, in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="ResNet18":
        model = networks.ResNet.resnet18(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="ResNet34":
        model = networks.ResNet.resnet34(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="ResNet50":
        model = networks.ResNet.resnet50(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="ResNet101":
        model = networks.ResNet.resnet101(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    elif model_name=="ResNet152":
        model = networks.ResNet.resnet152(in_channels=in_channels,num_classes=num_classes,init_weights=True).to(device)
    else:
        print(f'There is no {model_name}')

    # summary는 cpu 모드에서만 가능함
    if sum_on:
        summary(model, input_size=(in_channels, 224, 224)) # for torchsummary
        # summary(model, input_size=(in_channels, 299, 299)) # for over inceptionv3
    
    return model    
