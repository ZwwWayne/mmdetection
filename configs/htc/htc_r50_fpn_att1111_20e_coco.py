_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='1111',
                kv_stride=2),
            stages=(False, False, True, True),
            position='after_conv2')
    ]))
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
