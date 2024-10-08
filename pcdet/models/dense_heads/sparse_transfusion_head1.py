import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from ...utils.spconv_utils import replace_feature, spconv
from easydict import EasyDict



class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, preds, targets):
        # 示例：计算预测和目标之间的MSE损失
        return F.mse_loss(preds, targets)


class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict



class SparseTransFusionHead1(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(SparseTransFusionHead1, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        hidden_channel=self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        #self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap = loss_utils.FocalLossSparse()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']
        self.custom_loss = CustomLoss()
        self.code_size = 10
        self.kv_nums = 5000
        self.gaussian_ratio = 1.0

        # a shared convolution
        # self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        #layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        #layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        # transform to sparse
        layers.append(
            spconv.SparseSequential(
                spconv.SubMConv2d(128, 128, kernel_size=[1, 1], padding=[0, 0], bias=True, indice_key='hm'),
                nn.BatchNorm1d(128),
                nn.ReLU()
        ))
        layers.append(spconv.SubMConv2d(128, 10, kernel_size=[1, 1], padding=[0, 0], dilation=[1, 1], bias=True, indice_key='hmout'))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        # self.bev_pos = self.create_2D_grid(x_size, y_size) 不使用

        self.forward_ret_dict = {}

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def predict(self, inputs):
        # batch_size = inputs.shape[0]
        # lidar_feat = self.shared_conv(inputs)
        '''
        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        '''
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(inputs)
        self.forward_ret_dict['spatial_shape'] = spatial_shape
        self.forward_ret_dict['spatial_indices'] = spatial_indices
        self.forward_ret_dict['num_voxels'] = num_voxels
        # bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # query initialization
        dense_heatmap = self.heatmap_head(inputs) # shape = [num_voxels, 128]
        heatmap = dense_heatmap.features.detach().sigmoid() # SparseTensor类型的
        # 做稀疏的maxpool
        '''
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.dataset_name == "nuScenes":
            local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # for Pedestrian & Cyclist in Waymo
        elif self.dataset_name == "Waymo":
            local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        '''
        # top num_proposals among all classes
        # import pdb;pdb.set_trace()
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(inputs)
        top_proposals = []
        lidar_feat_flatten = [] # 加速小技巧，不用额外写for 循环
        start_idx = 0
        for idx in range(len(num_voxels)):
            lidar_feat_flatten.append(heatmap[start_idx:start_idx + num_voxels[idx]].view(-1).argsort(descending=True)[:self.kv_nums])
            top_proposals.append(lidar_feat_flatten[-1].argsort(descending=True)[:self.num_proposals])
            start_idx += num_voxels[idx]
        top_proposals = torch.stack(top_proposals) # shape = [batch_size, 200] # 需要注意BEV-pos的编码
        lidar_feat_flatten = torch.stack(lidar_feat_flatten) # [batch_size, 5000] # 每一个存储的是相对那个batch的坐标
        # top_proposals = heatmap.view(-1).argsort(descending=True)[:self.num_proposals] batch_size = 1
        #top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        #    ..., : self.num_proposals
        # ]
        top_rows = top_proposals // heatmap.shape[1] # 相对lidar_feat_flatten坐标
        top_proposals_class = top_proposals % heatmap.shape[1]
        top_lidar_feat_flatten_rows = lidar_feat_flatten // heatmap.shape[1] # 和原始的代码是相反的
        top_lidar_feat_flatten_cols = lidar_feat_flatten % heatmap.shape[1]
        # top_proposals_class = top_proposals // heatmap.shape[-1]
        # top_proposals_index = top_proposals % heatmap.shape[-1]
        # select kv_feats
        kv_feat = []
        kv_pos = []
        top_scores = []
        #q_pos = [] 
        start_idx = 0
        inputs = inputs.features # 转换为特征
        for idx in range(len(num_voxels)): # [27526, 128]
            kv_feat.append(inputs[start_idx:start_idx + num_voxels[idx]].permute(1,0).gather(
                index = top_lidar_feat_flatten_rows[idx].expand(128, -1),
                dim=-1,
            ))
            # get_pos spatial_indices[x,y]的列表
            kv_pos.append((spatial_indices[idx]*0.5 + 0.5).permute(1,0).gather(
                index = top_lidar_feat_flatten_rows[idx].expand(2, -1),
                dim = -1,
            ))
            top_scores.append(heatmap[start_idx:start_idx + num_voxels[idx]].permute(1,0).gather(
                index = top_lidar_feat_flatten_rows[idx].expand(10, -1),
                dim = -1,
            ))
            start_idx += num_voxels[idx]
            # q_pos.append(kv_pos)
        kv_feat = torch.stack(kv_feat) # [2, 128, 5000]
        kv_pos = torch.stack(kv_pos) # [2,2, 5000]
        top_scores = torch.stack(top_scores) # [2,10, 1000]
        # query_feat.shape  = [2, 128, 200]
        query_feat = kv_feat.gather(
            index=top_rows[:, None, :].expand(-1, 128, -1),
            dim=-1,
        )
         
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding
        # 需要注意BEV-pos的编码，一个是query的编码 一个是lidar_features的编码--- liulin
        #query_pos = bev_pos.gather(
        #    index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
        #    dim=1,
        #) query_pos的代码格式应该是[2, 200, 2] , 这里 kv_pos [2,2,5000] BEV_pos  = [2, 32400, 2], top_proposals_index [2,200]
        # import pdb;pdb.set_trace()
        query_pos = kv_pos.permute(0,2,1).gather(
            index=top_rows[:, None, :].permute(0, 2, 1).expand(-1, -1, 2),
            dim=1,
        )
        # convert to xy
        query_pos = query_pos.flip(dims=[-1])
        # bev_pos = bev_pos.flip(dims=[-1])
        bev_pos = kv_pos.flip(dims=[-1])

        #query_feat = self.decoder(
        #    query_feat, lidar_feat_flatten, query_pos, bev_pos
        #)kv_feat
        #import pdb; pdb.set_trace()
        query_feat = self.decoder(
            query_feat, kv_feat, query_pos, bev_pos.permute(0,2,1)
        )
        res_layer = self.prediction_head(query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        # top_proposals_index, [2, 200], [2, 10, 200] [2, 10, 32400]
        res_layer["query_heatmap_score"] = top_scores.gather(
            index=top_rows[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap.features # sparse Tensor
        
        return res_layer
    
    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def forward(self, batch_dict):
        # feats = batch_dict['spatial_features_2d']
        feats = batch_dict['encoded_spconv_tensor']
        # spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)
        res = self.predict(feats)
        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):

        #spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(inputs)
        spatial_shape = self.forward_ret_dict['spatial_shape']
        spatial_indices = self.forward_ret_dict['spatial_indices']
        num_voxels = self.forward_ret_dict['num_voxels']
        assign_results = []
        heatmap_list = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            #import pdb;pdb.set_trace()
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result, heatmap = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict, num_voxels[batch_idx], spatial_indices[batch_idx], spatial_shape)
            assign_results.append(assign_result)
            heatmap_list.append(heatmap)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        #import pdb;pdb.set_trace()
        heatmap = torch.cat(heatmap_list, dim=2) # [1, 10 , 32618]
        #heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap
        

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, num_voxels, spatial_indices, spatial_shape):
        
        #import pdb;pdb.set_trace()
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]["pred_boxes"]
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        #feature_map_size = (self.grid_size[:2] // self.feature_map_stride) 
        #heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, num_voxels) # 替换为稀疏体素

        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                distance = self.distance(spatial_indices, center)
                inds = distance.argmin()
                # 执行两次高斯map的绘制
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[gt_labels_3d[idx]], distance, radius * self.gaussian_ratio) # 要为每一个类均生成一个热图
                centernet_utils.draw_gaussian_to_heatmap_voxels(heatmap[gt_labels_3d[idx]], self.distance(spatial_indices, spatial_indices[inds]), radius * self.gaussian_ratio)
                # centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou)), heatmap[None]
    
    def distance(self, voxel_indices, center):
        distances = ((voxel_indices - center.unsqueeze(0))**2).sum(-1)
        return distances

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0
        loss_heatmap = self.loss_heatmap(clip_sigmoid(pred_dicts["dense_heatmap"]), torch.squeeze(heatmap, dim=0).permute(1,0))
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all
        #print("loss_cls:     ",loss_dict["loss_cls"], "loss_bbox:     ",loss_dict["loss_bbox"], "loss_heatmap", loss_dict["loss_heatmap"])

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        code_size = 10
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_bboxes(self, preds_dicts):

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        one_hot = F.one_hot(
            self.query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=True,
        )
        for k in range(batch_size):
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

        return ret_dict 
