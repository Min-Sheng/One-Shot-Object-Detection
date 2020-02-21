import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model.roi_layers import nms


class Logger:
    """The base class for all loggers.
    Args:
        log_dir (str): The saved directory.
    """
    def __init__(self, log_dir):
        
        self.writer = SummaryWriter(log_dir)

    def write(self, step, shot, session, log, sample_batched, pred_boxes, scores):
        """Plot the network architecture and the visualization results.
        Args:
            shot (int): The number of shots.
            session (int): The number of the session.
            step (int): The number of the step.
            log (dict): The log information.
            sample_batched (List of torch.Tensor): The sample batch.
            pred_boxes (torch.Tensor): The prediction bounding boxes.
            scores (torch.Tensor): The scores of each bbox.

        """
        self._add_scalars(step, shot, session, log)
        self._add_images(step, shot, session, sample_batched, pred_boxes, scores)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_scalars(self, step, shot, session, log):
        """Plot the training curves.
        Args:
            step (int): The number of the step.
            shot (int): The number of shots.
            session (int): The number of the session.
            log (dict): The log information.
        """
        self.writer.add_scalars("logs_{}shot_sess{}/losses".format(shot, session), log, step)

    def _add_images(self, step, shot, session, sample_batched, pred_boxes, scores):
        """Plot the visualization results.
        Args:
            step (int): The number of the step.
            shot (int): The number of shots.
            session (int): The number of the session.
            sample_batched (dict): The sample batch.
            pred_boxes (torch.Tensor): The prediction bounding boxes.
            scores (torch.Tensor): The scores of each bbox.
        """
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        to_tensor = transforms.ToTensor()

        im_batched = sample_batched[0].float()
        query_batched = sample_batched[1]
        im_info_batched = sample_batched[2].float()
        gt_boxes_batched = sample_batched[3].float()
        num_boxes_batched = sample_batched[4].float()
        im_name_batched = sample_batched[5]
        pred_boxes = pred_boxes[0].squeeze().float()
        scores = scores[0].squeeze().float()

        gt_boxes = gt_boxes_batched[0]
        im_name = im_name_batched[0]
        im_name = os.path.join(im_name.split('/')[4], im_name.split('/')[5])

        im = inv_normalize(im_batched[0]).permute(1, 2, 0).data.numpy()
        im = (im - im.max()) / (im.max() - im.min())
        im = (im *255).astype(np.uint8)
        im = Image.fromarray(im)

        querys = []
        for i in range(shot):

            query = inv_normalize(query_batched[i][0].float()).permute(1, 2, 0).data.numpy()
            query = (query - query.max()) / (query.max() - query.min())
            query = (query *255).astype(np.uint8)
            query = Image.fromarray(query)
            querys.append(to_tensor(query))

        querys_grid = make_grid(querys, nrow=shot//2, normalize=True, scale_each=True, pad_value=1)
        querys_grid = transforms.ToPILImage()(querys_grid).convert("RGB")
        query_w, query_h = querys_grid.size
        query_bg = Image.new('RGB', (im.size), (255, 255, 255))
        bg_w, bg_h = query_bg.size
        offset = ((bg_w - query_w) // 2, (bg_h - query_h) // 2)
        query_bg.paste(querys_grid, offset)

        im_gt_bbox = im.copy()
        im_pred_bbox = im.copy()

        for bbox in gt_boxes:
            if bbox.sum().item()==0:
                break
            bbox = tuple(list(map(int,bbox[0:4].tolist())))
            draw = ImageDraw.Draw(im_gt_bbox)
            draw.rectangle(bbox, fill=None, outline=(0, 110, 255), width=2)

        im = ImageOps.expand(im, border=(0,24,0,0), fill=(0,0,0))
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 16)
        draw.text((0,0), im_name, (255,255,255), font=font)
        im_gt_bbox = ImageOps.expand(im_gt_bbox, border=(0,24,0,0), fill=(0,0,0))
        query_bg = ImageOps.expand(query_bg, border=(0,24,0,0), fill=(0,0,0))
        
        thresh = 0.5
        test_nms = 0.3
        # Post processing
        inds = torch.nonzero(scores>thresh).view(-1)
        if inds.numel() > 0:
          # remove useless indices
          cls_scores = scores[inds]
          cls_boxes = pred_boxes[inds, :]
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

          # rearrange order
          _, order = torch.sort(cls_scores, 0, True)
          cls_dets = cls_dets[order]

          # NMS
          keep = nms(cls_boxes[order, :], cls_scores[order], test_nms)
          cls_dets = cls_dets[keep.view(-1).long()]
                    
          for dets in cls_dets:
              bbox = tuple(list(map(int,dets[0:4].tolist())))
              draw = ImageDraw.Draw(im_pred_bbox)
              draw.rectangle(bbox, fill=None, outline=(255, 0, 110), width=2)

        im_pred_bbox = ImageOps.expand(im_pred_bbox, border=(0,24,0,0), fill=(0,0,0))

        train_grid = [to_tensor(im), to_tensor(query_bg), to_tensor(im_gt_bbox), to_tensor(im_pred_bbox)]
        train_grid = make_grid(train_grid, nrow=2, normalize=True, scale_each=True, pad_value=1)
        self.writer.add_image("logs_{}shot_sess{}/train".format(shot, session), train_grid, step)
