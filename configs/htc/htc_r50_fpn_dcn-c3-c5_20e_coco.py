_base_ = './htc_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
# learning policy
lr_config = dict(step=[16, 19])
total_epochs = 20
