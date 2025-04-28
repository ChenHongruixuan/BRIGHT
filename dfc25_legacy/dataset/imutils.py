import random
import numpy as np


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label


def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_crop(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, 3), dtype=np.float32)

    pad_post_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label
