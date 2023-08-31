#######################################
###### changed/added by holzbock ######
#######################################

_base_ = [
    '../_base_/models/faster_rcnn_vgg16_bn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
