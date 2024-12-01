import torch
from AMSIN import InvDDNet
from utils import load_model, set_requires_grad
import os
from PIL import Image
from torchvision import transforms
import cv2

if __name__ == '__main__':
    net = InvDDNet().cuda()
    set_requires_grad(net, False)
    last_epoch = load_model(net, 'save/replicate/0', epoch=None)
    count = 0
    total_time = 0

    input_path = 'your test data'
    img_names = os.listdir(input_path)
    print(img_names)
    for i in range(len(img_names)):
        name = 'your test data/' + img_names[i]
        input_img = Image.open(name)
        input_img = transforms.ToTensor()(input_img)
        input_img = input_img.unsqueeze(0).cuda()
        x1 = torch.cat([input_img[:, :1], input_img[:, :1], input_img[:, :1]], dim=1)
        output_img, _, _, _ = net(x1, input_img)
        output_img = torch.clamp(output_img[:, 3:], min=1e-5, max=1.0)
        output_img = torch.squeeze(output_img)
        output_img = torch.flip(output_img, dims=[0])
        output_img = output_img.clone().detach().to(torch.device('cpu'))  # 到cpu
        ndarr = output_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv2.imwrite('your path to save/' + img_names[i], ndarr)
        print(str(i + 1) + '/' + str(177))

