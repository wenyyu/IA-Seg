import os

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from PIL import Image

from network import *
from network import dip
import numpy as np
import random

from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.acdc_night_dataset import acdcnightDataSet
from dataset.nightcity_dataset import nightcityDataSet


from configs.train_config import get_arguments


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def write_mes(msg, log_name=None, show=True, mode='a'):
    get_end = lambda line: '' if line.endswith('\n') else '\n'
    if show:
        if isinstance(msg, str):
            print(msg, end=get_end(msg))
        elif isinstance(msg, (list, tuple)):
            for line in msg:
                print(line, end=get_end(line))  # might be a different thing
        else:
            print(msg)

    if log_name is not None:
        with open(log_name, mode) as f:
            f.writelines(msg)

args = get_arguments()
if not os.path.exists(args.snapshot_dir):
    os.makedirs(args.snapshot_dir)
config_log = os.path.join(args.snapshot_dir, 'config.txt')

arg_dict = args.__dict__
msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
write_mes(msg, config_log, mode='w')
gpu_id = args.gpu_id
gpu_list = list()
gpu_ids = gpu_id.split(',')
for i in range(len(gpu_ids)):
    gpu_list.append('/gpu:%d' % int(i))

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda")

cudnn.enabled = True
cudnn.benchmark = True

def main():

    if args.model == 'PSPNet':
        model = PSPNet(num_classes=args.num_classes, dgf=args.DGF_FLAG)
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, dgf=args.DGF_FLAG)
    if args.model == 'RefineNet':
        model = RefineNet(num_classes=args.num_classes, imagenet=False, dgf=args.DGF_FLAG)
    saved_state_dict = torch.load(args.restore_from)

    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not i_parts[0] == 'fc':
            new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model.train()
    model.to(device)

    # lightnet = LightNet()
    # lightnet.train()
    # lightnet.to(device)
    CNNPP = dip.DIP()


    CNNPP.train()
    CNNPP.to(device)

    # model_D1 = FCDiscriminator(num_classes=args.num_classes)
    # model_D1 = FCDiscriminator(num_classes=19)
    #
    # model_D1.train()
    # model_D1.to(device)



    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(
        cityscapesDataSet(args, args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                          set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    nightcityloader = data.DataLoader(
        nightcityDataSet(args, args.data_dir_nightcity, args.data_list_nightcity, max_iters=args.num_steps * args.iter_size * args.batch_size,
                          set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    nightcityloader_iter = enumerate(nightcityloader)



    optimizer = optim.SGD(list(model.parameters())+list(CNNPP.parameters()),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()



    weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                           0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                           0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                           0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0



    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)



    interp = nn.Upsample(size=(args.input_size, args.input_size), mode='bilinear', align_corners=True)



    for i_iter in range(args.num_steps):
        loss_seg_value = 0
        loss_seg_value_nightcity = 0

        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)



        for sub_i in range(args.iter_size):



            # train with source
            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch

            images = images.to(device)

            labels = labels.long().to(device)

            enhanced_images_pre, ci_map, Pr, filter_parameters = CNNPP(images)
            enhanced_images = enhanced_images_pre


            for i_pre in range(enhanced_images_pre.shape[0]):
                enhanced_images[i_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    enhanced_images_pre[i_pre,...])

            if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
                pred_c = model(enhanced_images)
            else:
                _, pred_c = model(enhanced_images)
            pred_c = interp(pred_c)
            loss_seg = seg_loss(pred_c, labels)


            loss = loss_seg #+ loss_seg_dark_dynamic + loss_seg_mix #+ loss_seg_dark_dynamic #+ loss_enhance
            loss_s = loss / args.iter_size
            loss_s.backward(retain_graph=True)
            loss_seg_value += loss_seg.item() / args.iter_size

            # train with nightcity_night
            _, batch = nightcityloader_iter.__next__()

            images_nightcity, labels, _, _ = batch

            images_nightcity = images_nightcity.to(device)

            labels = labels.long().to(device)

            enhanced_images_nightcity_pre, ci_map, Pr, filter_parameters = CNNPP(images_nightcity)
            enhanced_images_nightcity = enhanced_images_nightcity_pre

            for i_pre in range(enhanced_images_nightcity_pre.shape[0]):
                enhanced_images_nightcity[i_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    enhanced_images_nightcity_pre[i_pre,...])

            if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
                pred_c_nightcity = model(enhanced_images_nightcity)
            else:
                _, pred_c_nightcity = model(enhanced_images_nightcity)
            pred_c_nightcity = interp(pred_c_nightcity)
            loss_seg_nightcity = seg_loss(pred_c_nightcity, labels)

            loss_s_nightcity = loss_seg_nightcity / args.iter_size
            loss_s_nightcity.backward()
            loss_seg_value_nightcity += loss_seg_nightcity.item() / args.iter_size

        optimizer.step()
        if i_iter % 100 == 0:# and i_iter != 0:
            print(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_seg_value_nightcity = {3:.3f},'.format(
                    i_iter, args.num_steps, loss_seg_value, loss_seg_value_nightcity))


        if i_iter < 30000:
            if i_iter % 25000 == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'ianet' + str(i_iter) + '.pth'))
                torch.save(CNNPP.state_dict(), os.path.join(args.snapshot_dir, 'ianet_cnnpp' + str(i_iter) + '.pth'))
        elif i_iter < 45000:
            if i_iter % 2500 == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'ianet' + str(i_iter) + '.pth'))
                torch.save(CNNPP.state_dict(), os.path.join(args.snapshot_dir, 'ianet_cnnpp' + str(i_iter) + '.pth'))
        else:
            if i_iter % 1000 == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'ianet' + str(i_iter) + '.pth'))
                torch.save(CNNPP.state_dict(),
                           os.path.join(args.snapshot_dir, 'ianet_cnnpp' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
