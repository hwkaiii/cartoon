import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
#从dataloater.u2net_data_loader模块中导入SalObjDataset类。这个类用于加载和处理图像数据。
from dataloater.u2net_data_loader import SalObjDataset
#从dataloater.u2net_data_loader模块中导入RescaleT类。这个类用于对图像进行缩放操作。
from dataloater.u2net_data_loader import RescaleT
#从dataloater.u2net_data_loader模块中导入ToTensorLab类。这个类用于将图像转换为张量。
from dataloater.u2net_data_loader import ToTensorLab
from torchvision import transforms
from utils.face_detect import FaceDetect
from model.u2net import U2NET
from torch.utils.data import DataLoader
from PIL import Image
from model.photo2cartoon import ResnetGenerator as Photo2Cartoon_ResnetGenerator



def get_face(image,scale,img_name):
    """剪切出人脸"""
    device = 'cpu'
    detector = 'dlib'
    detect = FaceDetect(device, detector)
    face_info = detect.align(image)
    image_align, landmarks_align = face_info  # 旋转后人脸得到矫正的图片(1079,999,3) 人脸关键点的坐标(68,2)
    face = detect.crop(image_align, landmarks_align,scale)  # 剪切旋转后的原图，得到人脸(429,429,3)
    # face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    cv2.imwrite('./dataset/face/'+img_name, face[:,:,::-1])
    return face


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')#
    img_name = image_name.split(os.sep)[-1]
    image = cv2.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    # imo.save(d_dir+imidx+'.png')
    imo.save(d_dir)
    imo = cv2.imread(d_dir)
    imo_gray = cv2.cvtColor(imo,cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(imo_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(d_dir,binary)

    #plt.title('face_mask')
    #plt.imshow(binary)
    #plt.show()
    return binary

def get_mask_U2net(face,img_name):
    # 模型路径
    model_dir = os.path.join(os.getcwd(), 'save_model', 'u2net_human_seg.pth')
    # 测试路径
    img_name_list = [os.path.join(os.getcwd(), 'dataset','face', img_name),]
    # 预测路径
    prediction_dir = os.path.join(os.getcwd(), 'dataset', 'result_seg' , img_name)

    # Dataset
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    u2net = U2NET(3,1)
    u2net.load_state_dict(torch.load(model_dir,map_location='cpu'))
    u2net = u2net
    u2net.eval()

    # 测试
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        d1, d2, d3, d4, d5, d6, d7 = u2net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder

        pred = save_output(img_name_list[i_test], pred, prediction_dir)
        return pred


def get_face_white_bg_photo2cartoon(face_rgba, img_name):
    face = face_rgba[:, :, :3].copy()  # 取出前三个通道
    mask = face_rgba[:, :, 3].copy()[:, :, np.newaxis] / 255.  # 取出mask，并转换为0-1之间的值
    face_white_bg = (face * mask + (1 - mask) * 255).astype(np.uint8)  # 背景变白色，并转换为uint8类型
    return face_white_bg  # 直接返回处理后的图像

def get_cartoon_face_photo2cartoon(face_white_bg,mask,img_name):
    mask = cv2.resize(mask, (256, 256))
    mask =  mask[:, :, np.newaxis].copy() / 255.


    face_white_bg = cv2.resize(face_white_bg, (256, 256), interpolation=cv2.INTER_AREA)
    face_white_bg = np.transpose(face_white_bg[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
    face_white_bg = torch.from_numpy(face_white_bg)

    model_dir = os.path.join(os.getcwd(), 'save_model', 'photo2cartoon_weights.pt')

    net = Photo2Cartoon_ResnetGenerator(ngf=32, img_size=256, light=True)
    params = torch.load(model_dir,map_location='cpu')
    net.load_state_dict(params['genA2B'])
    cartoon = net(face_white_bg)[0][0]
    # 处理过程
    cartoon = np.transpose(cartoon.cpu().detach().numpy(), (1, 2, 0))
    cartoon = (cartoon + 1) * 127.5
    cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
    cartoon = cv2.resize(cartoon,(512,512))[:,:,::-1]

    #plt.imshow(cartoon)
    #plt.title('cartoon')
    #plt.show()

    cv2.imwrite(os.path.join(os.getcwd(), 'dataset', 'cartoon_face', img_name)
               ,cartoon[:,:,::-1])
    return cartoon


#参数设置
def parse_opt():
    parser = argparse.ArgumentParser(description='Photo2Cartoon')
    # 人像图片名字
    parser.add_argument('--img-name',type=str,default='bb3.jpg',help='Image name')
    # 头部剪切比例，值越大剪切比例也大
    parser.add_argument('--shear-rate',type=int,default=0.8,help='Head cut rate')
    parser.add_argument('--segment-model',type=str,default='Unet',help='[U2net]')
    # 卡通风格迁移方法，用Photo2cartoon模型
    parser.add_argument('--migration-method',type=str,default='Photo2cartoon',help='[Photo2cartoon')

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # 读取图片,并且显示
    img = cv2.cvtColor(cv2.imread(os.path.join('dataset','img', opt.img_name)), cv2.COLOR_BGR2RGB)
    #plt.title('img')
    #plt.imshow(img)
    #plt.show()

    # 识别人脸关键点->人脸摆正->分割人脸
    # 参数： 原图，裁剪比例，图片名字
    face = get_face(img,opt.shear_rate,opt.img_name)
    # 分割图像
    # 得到切割后的黑白图
    mask = get_mask_U2net(face, opt.img_name)
    # 通过分割图像将人像的背景变为白色
    # 使用photo2cartoon模型进行卡通风格迁移
    if opt.migration_method == 'Photo2cartoon':
        face_white_bg = get_face_white_bg_photo2cartoon(np.dstack((face, mask)),opt.img_name) # 分割后的人脸
        cartoon_face = get_cartoon_face_photo2cartoon(face_white_bg,mask,opt.img_name) # 卡通图













