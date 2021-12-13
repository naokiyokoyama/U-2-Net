import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms#, utils

from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import cv2
import tqdm
import argparse
import os.path as osp
import numpy as np

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(osp.join(d_dir, imidx+'.png'))


def contour_mask(mask):
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    max_area = 0
    max_index = 0
    for idx, c in enumerate(cnt):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_index = idx
    cv2.drawContours(new_mask, cnt, max_index, 255, cv2.FILLED)

    return new_mask


def extract_object(img, mask):
    ''' Crop image to only feature the object '''
    # Remove small values
    # _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_TOZERO)
    # Extract only largest contour
    mask = contour_mask(mask)
    # Use mask as alpha channel for PNG
    bgra_img = cv2.merge([*cv2.split(img), mask])
    # Get bbox and use to crop
    x, y, w, h = cv2.boundingRect(mask)
    bgra_object = bgra_img[y : y + h, x : x + w]

    return bgra_object


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_path", help="/path/to/video from which object will be extracted"
    )
    parser.add_argument(
        "out_dir", help="/path/to/out_dir where object masks will be saved"
    )
    args = parser.parse_args()
    video_path = args.video_path
    out_dir = args.out_dir

    frames_dir = osp.join(out_dir, "frames")
    prediction_dir = osp.join(out_dir, "masks")
    objects_dir = osp.join(out_dir, "objects")
    model_name = 'u2net'#u2netp
    model_dir = osp.join('saved_models', model_name, model_name + '.pth')
    for d in [frames_dir, prediction_dir, objects_dir]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # --------- 1. save every frame of video ---------
    print("Saving video frames as images")
    vid = cv2.VideoCapture(video_path)
    total_num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm.trange(total_num_frames):
        _, frame = vid.read()
        frame_path = osp.join(frames_dir, f'{frame_id:04}.png')
        cv2.imwrite(frame_path, frame)


    img_name_list = glob.glob(osp.join(frames_dir, '*'))
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

    # --------- 5. extract object from inferred masks ---------
    print("Extracting objects..")
    for impath in tqdm.tqdm(sorted(glob.glob(osp.join(frames_dir, "*.png")))):
        basename = osp.basename(impath)
        mask_path = osp.join(prediction_dir, basename)
        img = cv2.imread(impath)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bgra_object = extract_object(img, mask)

        cv2.imwrite(osp.join(objects_dir, basename), bgra_object)

if __name__ == "__main__":
    main()
