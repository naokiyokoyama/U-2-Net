import argparse
import cv2
from os import path as osp
import numpy as np
import tqdm
import torch
from torchvision import transforms

from model import U2NET  # full size version 173.6 MB
from u2net_test import normPRED

MODEL_PATH = "saved_models/u2net.pth"


def save_output(predict, height, width, output_path):
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    pred_img = np.array(predict_np * 255, dtype=np.uint8)
    pred_img = cv2.resize(pred_img, (width, height))
    cv2.imwrite(output_path, pred_img)


def normalize_image(img):
    img = cv2.resize(img, (320, 320))
    tmp_img = np.zeros_like(img)
    img = img / np.max(img)
    tmp_img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    tmp_img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    tmp_img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    return tmp_img


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

    """1. Load model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()

    """2. Define data transforms"""
    data_transforms = transforms.Compose([normalize_image, transforms.ToTensor()])

    """3. Parse frames of video"""
    vid = cv2.VideoCapture(video_path)
    total_num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_id in tqdm.trange(total_num_frames):
        _, frame = vid.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_tensor = data_transforms(rgb_frame).unsqueeze(0).to(device)
        output = net(rgb_tensor)[0]

        # Normalize
        pred = normPRED(output[:, 0, :, :])

        # Save prediction
        height, width = frame.shape[:2]
        output_path = osp.join(out_dir, f"{frame_id:04}.png")
        save_output(pred, height, width, output_path)


if __name__ == '__main__':
    main()
