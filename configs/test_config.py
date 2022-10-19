import argparse

# validation set path

# For supervised nightcity + cityscapes
# DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/nightcity/images/val/'
# DATA_LIST_PATH = './dataset/lists/nightcity_val.txt'
# For supervised nightcity + cityscapes

# DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/ACDC/rgb_anon_trainvaltest/rgb_anon/'
# DATA_LIST_PATH = './dataset/lists/acdc_night_val.txt'

DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/cityscapes/leftImg8bit/val/'
DATA_LIST_PATH = './dataset/lists/cityscapes_val.txt'

# For unsupervised
# DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/Dark_Zurich_test_anon/rgb_anon/'
# DATA_LIST_PATH = './dataset/lists/zurich_test.txt'
# DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/Dark_Zurich_val_anon/rgb_anon/val/'
# DATA_LIST_PATH = './dataset/lists/zurich_val.txt'
# DATA_DIRECTORY = '/data/vdd/liuwenyu/DANNet/NighttimeDrivingTest/leftImg8bit/'
# DATA_LIST_PATH = './dataset/lists/nightdriving_test.txt'



# PSPNet, DeepLab, RefineNet,
IGNORE_LABEL = 255
NUM_CLASSES = 19
SET = 'test'
# PSPNet, DeepLab, RefineNet,
MODEL = 'PSPNet'
# supervised-nightcity, supervised-acdc, unsupervised
EXP_TYPE = 'supervised-nightcity'
# cityscapes_val, nightcity_val, acdc_night_val, zurich_test...
DATASET_VAL = 'nightcity_val'


RESTORE_FROM = './models/' + EXP_TYPE + '/' + MODEL + '/ianet.pth'
RESTORE_FROM_LIGHT = './models/' + EXP_TYPE + '/' + MODEL + '/ianet_cnnpp.pth'




SAVE_PATH = './output/'+ MODEL +'_' + EXP_TYPE +'_'+ DATASET_VAL

STD = 0.16


def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='2',
                        help='if use gpu, use gpu device id')
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument('--NIGHTCITY_FLAG', dest='NIGHTCITY_FLAG', type=bool, default=True,
                        help='whether nightcity  ')
    parser.add_argument('--CITYSCAPES_FLAG', dest='CITYSCAPES_FLAG', type=bool, default=False,
                        help='whether  cityscapes ')
    parser.add_argument('--DGF_FLAG', dest='DGF_FLAG', type=bool, default=True,
                        help='whether use DGF')
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-light", type=str, default=RESTORE_FROM_LIGHT,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--std", type=float, default=STD)
    return parser.parse_args()