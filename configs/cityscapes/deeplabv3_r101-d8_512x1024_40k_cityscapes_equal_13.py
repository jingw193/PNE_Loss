_base_ = [
    '../deeplabv3_r50-d8_512x1024_40k_cityscapes.py'
]
model = dict(pretrained='open-mmlab://resnet101_v1c',
             backbone=dict(depth=101),
             decode_head=dict(type='ASPPHead',
                              use_con_loss=True,
                              loss_con_decode=dict(type='ReverseContrastiveLoss',
                                                   patch_size=100,
                                                   loss_ratio=True,
                                                   temp=1,
                                                   loss_weight=1.3,
                                                   cal_function='EQUAL',
                                                   cal_gate=[0, 199],
                                                   posweight=True)))
