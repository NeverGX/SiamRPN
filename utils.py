import numpy as np
import cv2

def get_center(x):
    return (x - 1.) / 2.

def xyxy2cxcywh(bbox):
    return get_center(bbox[0]+bbox[2]), \
           get_center(bbox[1]+bbox[3]), \
           (bbox[2]-bbox[0]), \
           (bbox[3]-bbox[1])

def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    xmin = cx - original_sz // 2
    xmax = cx + original_sz // 2 + original_sz % 2
    ymin = cy - original_sz // 2
    ymax = cy + original_sz // 2 + original_sz % 2
    im_h, im_w, _ = img.shape

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]
    if left != 0 or right !=0 or top!=0 or bottom!=0:
        if img_mean is None:
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=img_mean)
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch

def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    exemplar_img = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z

def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w+h)
    hc_z = h + context_amount * (w+h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad  # 2pad = (instance_size/exemplar_size)*s_z -s_z
    scale_x = size_x / s_x
    instance_img = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    return instance_img, scale_x, s_x

def get_pyramid_instance_image(img, center, size_x, size_x_scales, img_mean=None):
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, center[0], center[1], size_x, size_x_scale, img_mean)
            for size_x_scale in size_x_scales]
    return pyramid


def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # (5,4x225) to (225x5,4)
    ori = - (score_size // 2) * total_stride
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def compute_iou(anchors, box):
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou

def box_transform(anchors, gt_box):
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w
    target_y = (gt_cy - anchor_yctr) / anchor_h
    target_w = np.log(gt_w / anchor_w)
    target_h = np.log(gt_h / anchor_h)
    regression_target = np.hstack((target_x, target_y, target_w, target_h))
    return regression_target

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    """
    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    return im_patch

def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)
