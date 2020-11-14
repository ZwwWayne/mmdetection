_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(plugins=[
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 4),
            stages=(False, True, True, True),
            position='after_conv3')
    ]))

# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
