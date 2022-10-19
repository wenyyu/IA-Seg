import json
import argparse
import numpy as np
from PIL import Image
from os.path import join


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'zurich_val.txt')
    label_path_list = join(devkit_dir, 'label_zurich.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


# def main(args):
#    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gt_dir', default='/data/vdd/liuwenyu/DANNet/Dark_Zurich_val_anon/', type=str, help='directory which stores CityScapes val gt images')
#     parser.add_argument('--pred_dir', default='./result/dannet_PSPNet', type=str, help='directory which stores CityScapes val pred images')
#     parser.add_argument('--devkit_dir', default='./dataset/lists', help='base directory of cityscapes')
#     args = parser.parse_args()
#     main(args)


def compute_mIoU_nightdriving(gt_dir, pred_dir, devkit_dir='', data_mode=''):
    """
    Compute IoU given the predicted colorized images and
    """

    with open('./dataset/lists/info.json', 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    hist = np.zeros((num_classes, num_classes))
    mapping = np.array(info['label2train'], dtype=np.int)
    pred_imgs_ = open(devkit_dir, 'r').read().splitlines()
    if data_mode == 'nightdriving':
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs_]
        gt_imgs = [join(gt_dir, x.split('/')[-1][:-15] + 'gtCoarse_labelIds.png') for x in pred_imgs_]
    elif data_mode == 'nightcity':
        pred_imgs = [join(pred_dir, x) for x in pred_imgs_]
        gt_imgs = [join(gt_dir, x[:-4] + '_labelIds.png') for x in pred_imgs_]
    elif data_mode == 'acdc':
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs_]
        gt_imgs = [join(gt_dir, x[:-12] + 'gt_labelIds.png') for x in pred_imgs_]
    elif data_mode == 'cityscapes':
        pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs_]
        gt_imgs = [join(gt_dir, x[:-15] + 'gtFine_labelIds.png') for x in pred_imgs_]



    for ind in range(len(gt_imgs)):
        # print(pred_imgs[ind])
        # print(gt_imgs[ind])
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs

def main(args):
   compute_mIoU_nightdriving(args.gt_dir, args.pred_dir, args.devkit_dir, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gt_dir', default='/data/vdd/liuwenyu/DANNet/NighttimeDrivingTest/gtCoarse_daytime_trainvaltest/test/night/', type=str, help='directory which stores CityScapes val gt images')
    # parser.add_argument('--pred_dir', default='/data/vdd/liuwenyu/DANNet/result/test/dannet_PSPNet/exp85nightdriving/', type=str, help='directory which stores CityScapes val pred images')
    # parser.add_argument('--devkit_dir', default='./dataset/lists/nightdriving_test.txt', help='base directory of cityscapes')

    parser.add_argument('--gt_dir', default='/data/vdd/liuwenyu/DANNet/cityscapes/gtFine/val/', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', default='./output/PSPNet_unsupervised_cityscapes_val/', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='./dataset/lists/cityscapes_val.txt', help='base directory of cityscapes')

    # parser.add_argument('--gt_dir', default='/data/vdd/liuwenyu/DANNet/nightcity/label/val/', type=str, help='directory which stores CityScapes val gt images')
    # parser.add_argument('--pred_dir', default='/data/vdd/liuwenyu/DANNet/result/test/dannet_PSPNet/exp106nightcity_val/', type=str, help='directory which stores CityScapes val pred images')
    # parser.add_argument('--devkit_dir', default='./dataset/lists/nightcity_val.txt', help='base directory of cityscapes')

    # parser.add_argument('--gt_dir', default='/data/vdd/liuwenyu/DANNet/ACDC/gt_trainval/gt/', type=str, help='directory which stores CityScapes val gt images')
    # parser.add_argument('--pred_dir', default='/data/vdd/liuwenyu/DANNet/result/test/dannet_RefineNet/exp134acdcval/', type=str, help='directory which stores CityScapes val pred images')
    # parser.add_argument('--devkit_dir', default='./dataset/lists/acdc_night_val.txt', help='base directory of cityscapes')

    parser.add_argument("--data", type=str, default='cityscapes',
                        help="Data Choice (nightcity/nightdriving/acdc/cityscapes).")
    args = parser.parse_args()
    main(args)
