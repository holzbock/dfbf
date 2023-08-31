#######################################
###### changed/added by holzbock ######
#######################################

# optimizer
# according to: https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html
# Important: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8*2 = 16).
# According to the linear scaling rule, you need to set the learning rate proportional to the batch size
# if you use different GPUs or images per GPU, e.g., lr=0.01 for 4 GPUs * 2 imgs/gpu and lr=0.08 for
# 16 GPUs * 4 imgs/gpu. Default lr: 0.02
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
