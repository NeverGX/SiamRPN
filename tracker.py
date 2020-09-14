import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
from siamrpn import SiamRPN
from config import config
from custom_transforms import ToTensor
from utils import generate_anchors, get_subwindow_tracking, python2round, compute_iou

torch.set_num_threads(1) # otherwise pytorch will take all cpus

def extreme_point_detection(r):
    sz = r.shape[0]
    r_pad = np.ones((sz+4,sz+4))
    for i in range(sz):
        for j in range(sz):
            r_pad[i+2,j+2] = r[i,j]
    r = r_pad
    P = []
    for i in range(2, sz - 2):
        for j in range(2,sz-2):
            if r[i,j]> 0.5:
                if (r[i,j]>=r[i-2,j-2] and r[i,j]>=r[i-2,j-1] and r[i,j]>=r[i-2,j] and r[i,j]>=r[i-2,j+1] and r[i,j]>=r[i-2,j+2] and
                r[i,j]>=r[i-1,j-2] and r[i,j]>=r[i-1,j-1] and r[i,j]>=r[i-1,j] and r[i,j]>=r[i-1,j+1] and r[i,j]>=r[i-1,j+2] and
                r[i,j]>=r[i,j-2] and r[i,j]>=r[i,j-1]  and r[i,j]>=r[i,j+1] and r[i,j]>=r[i, j+2] and
                r[i,j]>=r[i+1,j-2] and r[i,j]>=r[i+1,j-1] and r[i,j]>=r[i+1,j] and r[i,j]>=r[i+1,j+1] and r[i, j]>=r[i+1,j+2] and
                r[i,j]>=r[i+2,j-2] and r[i,j]>=r[i+2,j-1] and r[i,j]>=r[i+2,j] and r[i,j]>=r[i+2,j+1] and r[i, j]>=r[i+2,j+2]
                ):
                    P.append([i,j])

    return P


def vis_heatmap(map, max_value):
    map = cv2.resize(map, (136,136), cv2.INTER_CUBIC)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.title("max_value: "+str(max_value))
    ax.imshow(map, interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.01)
    plt.clf()

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class SiamRPNTracker:
    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id):
            self.model = SiamRPN()
            self.model.load_model(model_path)
            self.model = self.model.cuda()
            self.model.eval()
        self.response_sz = config.response_sz
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size,
                                        config.anchor_scales, config.anchor_ratios, self.response_sz)
        self.transforms = transforms.Compose([ToTensor()])

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: zero-based bounding box [x, y, width, height]
        """
        self.pos = np.array([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2])  # center x, center y
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height

        wc_z = self.target_sz[0] + 0.5 * sum(self.target_sz)
        hc_z = self.target_sz[1] + 0.5 * sum(self.target_sz)
        self.s_z = np.sqrt(wc_z * hc_z)
        self.s_x = self.s_z * config.instance_size / config.exemplar_size


        # get exemplar img
        img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img = get_subwindow_tracking(frame, self.pos, config.exemplar_size, python2round(self.s_z), img_mean)
        exemplar_img = self.transforms(exemplar_img)[None,:,:,:]

        # get exemplar feature
        with torch.cuda.device(self.gpu_id):
            exemplar_img = Variable(exemplar_img.cuda(), requires_grad=False)
            self.model(exemplar_img, None)

        # create hanning window
        self.hann_window = np.outer(np.hanning(self.response_sz), np.hanning(self.response_sz))
        self.hann_window = np.tile(self.hann_window.flatten(), len(config.anchor_ratios) * len(config.anchor_scales))
        self.counter_re = 0


    def update(self, frame, gt, clf, random_shift, frame_num):
        """track object based on the previous frame
        Args:
            frame: an BGR image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """

        #######################
        if random_shift:
            pos_ = np.array([gt[0] + gt[2] / 2, gt[1] + gt[3] / 2])  # center x, center y, zero based
            max_translate = 2 * (self.s_x / config.instance_size)*config.total_stride
            self.pos[0] = np.random.uniform(pos_[0] - max_translate,
                                                pos_[0] + max_translate)
            self.pos[1] = np.random.uniform(pos_[1] - max_translate,
                                                pos_[1] + max_translate)
        #########################
        # get instance img
        img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        instance_img = get_subwindow_tracking(frame, self.pos, config.instance_size, python2round(self.s_x), img_mean)
        instance_img = self.transforms(instance_img)[None, :, :, :]

        # get instance feature
        with torch.cuda.device(self.gpu_id):
            instance_img = Variable(instance_img.cuda(), requires_grad=False)
            pred_cls, pred_reg = self.model(None, instance_img)

        #offsets
        offsets = pred_reg.squeeze().view(4, -1).detach().cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)

        # response
        max_value = pred_cls.max().detach().cpu().numpy()
        response = torch.sigmoid(pred_cls).squeeze().view(-1).detach().cpu().numpy()
        response_raw = response
        response = response * penalty
        response = (1 - config.window_influence) * response + config.window_influence * self.hann_window
        best_id = np.argmax(response)

        # anomaly detection
        anchor_id = best_id // 289
        response_map = response_raw[anchor_id*289:anchor_id*289+289]
        vis_heatmap(response_map.reshape(17,17), max_value)

        clf_output = clf(torch.from_numpy(response_map).float().cuda()).data.cpu().numpy()
        state = np.argmax(clf_output)
        # print(state)

        # response_label = self.create_response_label(response_map.reshape(17, 17), self.s_x, anchor_id)
        # dae_output = sigmoid(dae(torch.from_numpy(response_map).float().cuda()).data.cpu().numpy())
        # loss = np.mean((dae_output-response_label.flatten())**2)

        update_flag = 1
        if state == 0 :
            # print(' frame:'+str(frame_num)+'  '+str(response_raw.max()))
            update_flag = 0
            # self.counter_re += 1
            # window_influence_re = 0.26
            # response_re = response_raw * penalty
            # response_re = (1 - window_influence_re) * response_re + window_influence_re * self.hann_window
            # best_id = np.argmax(response_re)
        #     if self.counter_re % 12 == 0 and self.counter_re != 0:
        #         best_candidate_pos, best_candidate_id, best_candidate_anchor_id, best_candidate_offsets, \
        #         best_candidate_response_map = self.redection(frame, self.s_x, img_mean, 2)
        #         if best_candidate_pos is not None:
        #             clf_output_re = clf(torch.from_numpy(best_candidate_response_map).float().cuda()).data.cpu().numpy()
        #             state_re = np.argmax(clf_output_re)
        #             if state_re ==1 :
        #                 self.pos = best_candidate_pos
        #                 offsets = best_candidate_offsets
        #                 best_id = best_candidate_id
        #                 self.counter_re = 0
        # else:
        #     self.counter_re = 0

        # peak location
        offset = offsets[:, best_id] * self.s_z / config.exemplar_size

        # update center
        self.pos += np.array([offset[0], offset[1]])
        self.pos = np.clip(self.pos, 0, [frame.shape[1], frame.shape[0]])

        # update scale
        lr = response[best_id] * config.scale_lr
        self.target_sz = (1 - lr) * self.target_sz + lr * np.array([offset[2], offset[3]])
        self.target_sz = np.clip(self.target_sz, 10, [frame.shape[1], frame.shape[0]])
        wc_z = self.target_sz[1] + 0.5 * sum(self.target_sz)
        hc_z = self.target_sz[0] + 0.5 * sum(self.target_sz)
        self.s_z = np.sqrt(wc_z * hc_z)
        self.s_x = self.s_z * config.instance_size / config.exemplar_size

        #update_model
        if update_flag :
            exemplar_img = get_subwindow_tracking(frame, self.pos, config.exemplar_size, python2round(self.s_z), img_mean)
            exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
            with torch.cuda.device(self.gpu_id):
                exemplar_img = Variable(exemplar_img.cuda(), requires_grad=False)
                self.model.update_model(exemplar_img)


        # return 1-indexed and left-top based bounding box
        bbox = np.array([
            self.pos[0] - (self.target_sz[0]) / 2,
            self.pos[1] - (self.target_sz[1]) / 2,
            self.pos[0] + (self.target_sz[0]) / 2,
            self.pos[1] + (self.target_sz[1]) / 2])

        return bbox, response_map

    def _create_penalty(self, target_sz, offsets):

        def padded_size(w, h):

            context = config.context_amount * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)

        s_c = larger_ratio(padded_size(offsets[2], offsets[3]) / (padded_size(target_sz[0], target_sz[1])))

        r_c = larger_ratio((target_sz[0] / target_sz[1]) / (offsets[2] / offsets[3]))

        penalty = np.exp(-(r_c * s_c - 1) * config.penalty_k)

        return penalty


    def redection(self, frame, s_x, img_mean, ratio):
        # get global instance img
        instance_img_global = get_subwindow_tracking(frame, self.pos, config.instance_size*ratio, ratio*s_x, img_mean)
        instance_img_global = self.transforms(instance_img_global)[None, :, :, :]
        # get global instance feature
        with torch.cuda.device(self.gpu_id):
            instance_img_global = Variable(instance_img_global.cuda())
            pred_cls, pred_reg = self.model(None, instance_img_global)

        # global response
        score_size = int((config.instance_size*ratio - config.exemplar_size) / config.total_stride + 1)
        global_response = torch.sigmoid(pred_cls).squeeze().view(-1).detach().cpu().numpy()
        global_best_id = np.argmax(global_response)
        global_anchor_id = global_best_id // (score_size*score_size)

        extreme_points = extreme_point_detection(global_response.reshape(config.anchor_num,score_size,score_size)[global_anchor_id])

        # print(extreme_points)
        if len(extreme_points) <=0:
            return None,None,None,None,None

        max_value = 0.5
        for p in extreme_points:
            p[0] = float(p[0]-8)*(config.total_stride * s_x * ratio)/config.instance_size
            p[1] = float(p[1]-8)*(config.total_stride * s_x * ratio)/config.instance_size
            candidate_pos = self.pos + p
            # get candidate instance img
            instance_img = get_subwindow_tracking(frame, candidate_pos, config.instance_size, s_x, img_mean)
            instance_img = self.transforms(instance_img)[None, :, :, :]
            # get candidate instance feature
            with torch.cuda.device(self.gpu_id):
                instance_img = Variable(instance_img.cuda())
                pred_cls, pred_reg = self.model(None, instance_img)

            # candidate offsets
            candidate_offsets = pred_reg.squeeze().view(4, -1).detach().cpu().numpy()
            candidate_offsets[0] = candidate_offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
            candidate_offsets[1] = candidate_offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
            candidate_offsets[2] = np.exp(candidate_offsets[2]) * self.anchors[:, 2]
            candidate_offsets[3] = np.exp(candidate_offsets[3]) * self.anchors[:, 3]

            candidate_response = torch.sigmoid(pred_cls).squeeze().view(-1).detach().cpu().numpy()
            candidate_response_raw = candidate_response
            candidate_response = (1 - config.window_influence) * candidate_response + config.window_influence * self.hann_window
            candidate_best_id = np.argmax(candidate_response)
            candidate_anchor_id = candidate_best_id // 289
            candidate_response_map = candidate_response_raw[candidate_anchor_id * 289:candidate_anchor_id * 289 + 289]

            if candidate_response.max() > max_value:
                best_candidate_pos = candidate_pos
                best_candidate_id = candidate_best_id
                best_candidate_anchor_id = candidate_anchor_id
                best_candidate_offsets = candidate_offsets
                best_candidate_response_map = candidate_response_map
                max_value = candidate_response.max()

        if max_value == 0.5:
            return  None,None,None,None,None

        return  best_candidate_pos, best_candidate_id, best_candidate_anchor_id, best_candidate_offsets, best_candidate_response_map


    def create_response_label(self, response_map, s_x, anchor_id):
        scale_x = config.instance_size / s_x
        max_x, max_y = np.unravel_index(response_map.argmax(), response_map.shape)
        max_x = max_x - 8
        max_y = max_y - 8
        bbox = [max_y*config.total_stride, max_x*config.total_stride, self.target_sz[0]*scale_x, self.target_sz[1]*scale_x]
        iou = compute_iou(self.anchors, bbox).flatten()
        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou <= config.neg_threshold)[0]
        classification_label = np.tile(response_map.flatten(), config.anchor_num)
        # classification_label = np.ones_like(iou, dtype=np.float32)*-1
        classification_label[pos_index] = 1
        classification_label[neg_index] = 0
        return classification_label[anchor_id*289:anchor_id*289+289].reshape(17,17)
