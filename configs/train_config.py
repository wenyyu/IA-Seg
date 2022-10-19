import argparse
from easydict import EasyDict as edict
from network.filters import *
MODEL = 'RefineNet'  # PSPNet, DeepLab, RefineNet,deeplabv3plus
# RESTORE_FROM = './pretrained_models/pretrain_deeplab_150000.pth'
# RESTORE_FROM = './pretrained_models/pretrain_pspnet_150000.pth'
RESTORE_FROM = './pretrained_models/pretrain_refinenet_150000.pth'


BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 2

SET = 'train'
DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/cityscapes'
DATA_LIST_PATH = './dataset/lists/cityscapes_train.txt'

DATA_DIRECTORY_ACDC = '/data/vdd/liuwenyu/DANNet/ACDC'
DATA_LIST_PATH_ACDC_NIGHT = './dataset/lists/acdc_night_train.txt'

DATA_DIRECTORY_NIGHTCITY = '/data/vdd/liuwenyu/DANNet/nightcity'
DATA_LIST_PATH_NIGHTCITY = './dataset/lists/nightcity_train.txt'

INPUT_SIZE = '512'
DATA_DIRECTORY_TARGET = '/data/vdd/liuwenyu/DANNet/Dark_Zurich_train_anon/rgb_anon'
DATA_LIST_PATH_TARGET = './dataset/lists/zurich_dn_pair_train.csv'
INPUT_SIZE_TARGET = '960'

NUM_CLASSES = 19
IGNORE_LABEL = 255

LEARNING_RATE = 2.5e-4
POWER = 0.9
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4

NUM_STEPS = 50001
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = '/data/vdd/liuwenyu/DANNet/snapshots/'+MODEL +'/exp169'
STD = 0.05


def get_arguments():
    parser = argparse.ArgumentParser(description="IA-Seg")
    parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0',
                        help='if use gpu, use gpu device id')
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument('--DGF_FLAG', dest='DGF_FLAG', type=bool, default=True, help='whether use DGF')

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-acdc", type=str, default=DATA_DIRECTORY_ACDC,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-nightcity", type=str, default=DATA_DIRECTORY_NIGHTCITY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-list-acdc-night", type=str, default=DATA_LIST_PATH_ACDC_NIGHT,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-list-nightcity", type=str, default=DATA_LIST_PATH_NIGHTCITY,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=int, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with pimgolynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--std", type=float, default=STD)
    return parser.parse_args()

__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

###########################################################################
# Filter Parameters
###########################################################################
# cfg.filters = [
#     ExposureFilter, GammaFilter, ImprovedWhiteBalanceFilter,
#     SaturationPlusFilter, ToneFilter, ContrastFilter, WNBFilter, ColorFilter
# ]

# cfg.filters = [GammaFilter, DefogFilter]
# cfg.num_filter_parameters = 2
#
# cfg.gamma_begin_param = 0
# cfg.defog_begin_param = 1

# cfg.filters = [DefogFilter]
# cfg.num_filter_parameters = 1
#
# cfg.defog_begin_param = 0

# cfg.filters = [UsmFilter]
# cfg.num_filter_parameters = 1
#
# cfg.usm_begin_param = 0

# cfg.filters = [
#     DefogFilter, UsmFilter
# ]
# cfg.num_filter_parameters = 2
#
# cfg.defog_begin_param = 0
# cfg.usm_begin_param = 1

cfg.filters = [ExposureFilter, GammaFilter, ContrastFilter, UsmFilter]
# cfg.filters = []

cfg.num_filter_parameters = 4

cfg.exposure_begin_param = 0
cfg.gamma_begin_param = 1
cfg.contrast_begin_param = 2
cfg.usm_begin_param = 3


# cfg.filters = [ExposureFilter, GammaFilter, ToneFilter, ContrastFilter, UsmFilter]
# # cfg.filters = [GammaFilter]
# cfg.num_filter_parameters = 12
#
# cfg.exposure_begin_param = 0
# cfg.gamma_begin_param = 1
# cfg.tone_begin_param = 2
# cfg.contrast_begin_param = 10
# cfg.usm_begin_param = 11
# cfg.defog_begin_param = 0
#
# cfg.wb_begin_param = 1
# cfg.gamma_begin_param = 4
# cfg.tone_begin_param = 5
# cfg.contrast_begin_param = 13
# cfg.usm_begin_param = 14

# cfg.filters = [
#     DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
#     ToneFilter, ContrastFilter
# ]
# cfg.num_filter_parameters = 14
#
# cfg.defog_begin_param = 0
#
# cfg.wb_begin_param = 1
# cfg.gamma_begin_param = 4
# cfg.tone_begin_param = 5
# cfg.contrast_begin_param = 13

# cfg.filters = [
#     ImprovedWhiteBalanceFilter,  GammaFilter,
#     ToneFilter, ContrastFilter, UsmFilter
# ]
# cfg.num_filter_parameters = 14
#
# cfg.wb_begin_param = 0
# cfg.gamma_begin_param = 3
# cfg.tone_begin_param = 4
# cfg.contrast_begin_param = 12
# cfg.usm_begin_param = 13

# cfg.filters = [
#     ImprovedWhiteBalanceFilter,  GammaFilter,
#     ToneFilter, ContrastFilter
# ]
# cfg.num_filter_parameters = 13
#
# cfg.wb_begin_param = 0
# cfg.gamma_begin_param = 3
# cfg.tone_begin_param = 4
# cfg.contrast_begin_param = 12

# cfg.filters = [
#     DefogFilter, ExposureFilter, ImprovedWhiteBalanceFilter, SaturationPlusFilter,
#     GammaFilter, ToneFilter, ContrastFilter, UsmFilter
# ]
# cfg.num_filter_parameters = 17
#
# cfg.defog_begin_param = 0
# cfg.exposure_begin_param = 1
# cfg.wb_begin_param = 2
# cfg.saturation_begin_param = 5
# cfg.gamma_begin_param = 6
# cfg.tone_begin_param = 7
# cfg.contrast_begin_param = 15
# cfg.usm_begin_param = 16

# cfg.filters = [
#     ImprovedWhiteBalanceFilter,  GammaFilter, ExposureFilter,
#     ToneFilter, ContrastFilter, DefogFilter
# ]
# cfg.num_filter_parameters = 15
#
# cfg.wb_begin_param = 0
# cfg.gamma_begin_param = 3
# cfg.exposure_begin_param = 4
# cfg.tone_begin_param = 5
# cfg.contrast_begin_param = 13
# cfg.defog_begin_param = 14
# cfg.color_begin_param = 5
#
# cfg.wnb_begin_param = 4
# cfg.level_begin_param = 5
# cfg.vignet_begin_param = 5


# Gamma = 1/x ~ x
cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)
cfg.cont_range = (0.0, 1.0)