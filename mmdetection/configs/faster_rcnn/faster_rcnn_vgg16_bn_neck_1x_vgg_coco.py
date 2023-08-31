#######################################
###### changed/added by holzbock ######
#######################################

_base_ = [
    '../_base_/models/faster_rcnn_vgg16_bn_neck.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x_vgg.py', '../_base_/default_runtime.py'
]
