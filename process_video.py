import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import tqdm
from data_loader import RescaleT, SalObjDataset, ToTensorLab
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def get_mask(height, width, pred):
    predict = pred
    predict = predict.squeeze()
    mask = predict.cpu().data.numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (width, height))
    mask = np.reshape(mask, (*mask.shape[:2], 1))
    # Remove small values
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_TOZERO)
    # Extract only largest contour
    mask = contour_mask(mask)
    return mask


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


def extract_object(img, mask, crop=True):
    """Crop image to only feature the object"""
    # Use mask as alpha channel for PNG
    bgra_object = cv2.merge([*cv2.split(img), mask])
    # Get bbox and use to crop
    if crop:
        x, y, w, h = cv2.boundingRect(mask)
        bgra_object = bgra_object[y : y + h, x : x + w]

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

    objects_dir = osp.join(out_dir, "objects")
    viz_video_path = osp.join(out_dir, "visualized.mp4")
    model_name = "u2net"  # u2netp
    model_dir = osp.join("saved_models", model_name, model_name + ".pth")
    if not os.path.exists(objects_dir):
        os.makedirs(objects_dir, exist_ok=True)

    # --------- 1. save every frame of video ---------
    print("Extracting each frame of video...")
    vid = cv2.VideoCapture(video_path)
    total_num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = vid.get(cv2.CAP_PROP_FPS)
    orig_frames = []
    rgb_frames = []
    for _ in tqdm.trange(total_num_frames):
        _, frame = vid.read()
        orig_frames.append(frame)
        rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    height, width = orig_frames[0].shape[:2]
    # --------- 2. create dataloaders ---------
    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    test_salobj_dataset = SalObjDataset(rgb_frames, [], transform)
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # --------- 3. model define ---------
    if model_name == "u2net":
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == "u2netp":
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    else:
        raise NotImplementedError(f"Model name {model_name} not recognized.")

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location="cpu"))
    net.eval()

    # --------- 4. inference for each image ---------
    print("Inferring salient pixels for each frame...")
    masks = []
    iter_dataloader = iter(test_salobj_dataloader)
    for _ in tqdm.trange(len(orig_frames)):
        inputs_test = next(iter_dataloader)["image"].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1 = net(inputs_test)[0]

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        masks.append(get_mask(height, width, pred))

    # --------- 5. extract object from inferred masks ---------
    print("Extracting objects..")
    for idx in tqdm.trange(len(orig_frames)):
        img, mask = orig_frames[idx], masks[idx]
        try:
            bgra_object = extract_object(img, mask)
        except TypeError:
            continue
        out_path = osp.join(objects_dir, f"{idx:04}.png")
        cv2.imwrite(out_path, bgra_object)

    # --------- 6. generate video for visualization ---------
    four_cc = cv2.VideoWriter_fourcc(*"MP4V")
    out_vid = None
    for idx in tqdm.trange(len(orig_frames)):
        img, mask = orig_frames[idx], masks[idx]
        object_only = img.copy()
        object_only[mask==0] = (255,255,255)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        viz_frame = np.vstack([img, object_only, mask_bgr])
        # Make the video have a max height of 720
        scale = 720 / (height * 3)
        viz_frame = cv2.resize(
            viz_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        viz_height, viz_width = viz_frame.shape[:2]
        if out_vid is None:
            out_vid = cv2.VideoWriter(
                viz_video_path, four_cc, orig_fps, (viz_width, viz_height)
            )
        out_vid.write(viz_frame)
    print(f"Created visualization video at {viz_video_path}!")


if __name__ == "__main__":
    main()
