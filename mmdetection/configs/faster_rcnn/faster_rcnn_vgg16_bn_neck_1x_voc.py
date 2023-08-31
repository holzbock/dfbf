
#######################################
###### changed/added by holzbock ######
#######################################

_base_ = [
    '../_base_/models/faster_rcnn_vgg16_bn_neck.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
