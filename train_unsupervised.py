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

from dataset.zurich_pair_dataset import zurich_pair_DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
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


    CNNPP = dip.DIP()


    CNNPP.train()
    CNNPP.to(device)

    # model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D1 = FCDiscriminator(num_classes=19)

    model_D1.train()
    model_D1.to(device)

    # model_D2 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=19)

    model_D2.train()
    model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    trainloader = data.DataLoader(
        cityscapesDataSet(args, args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                          set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(zurich_pair_DataSet(args, args.data_dir_target, args.data_list_target,
                                                       max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                       set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    targetloader_iter = enumerate(targetloader)
    optimizer = optim.SGD(list(model.parameters())+list(CNNPP.parameters()),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    weights = torch.log(torch.FloatTensor([0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272, 0.01227341,
                                           0.00207795, 0.0055127, 0.15928651, 0.01157818, 0.04018982, 0.01218957,
                                           0.00135122, 0.06994545, 0.00267456, 0.00235192, 0.00232904, 0.00098658,
                                           0.00413907])).cuda()
    weights = (torch.mean(weights) - weights) / torch.std(weights) * args.std + 1.0

    weights_dynamic = weights
    weights_dynamic[0:11] = weights_dynamic[0:11] * 0.8

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)

    static_loss = StaticLoss(num_classes=11, weight=weights[:11])

    interp = nn.Upsample(size=(args.input_size, args.input_size), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.input_size_target, args.input_size_target), mode='bilinear', align_corners=True)

    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):
        loss_seg_dark_value = 0
        loss_seg_value = 0

        loss_adv_target_value_D1 = 0
        loss_adv_target_value_D2 = 0

        loss_pseudo = 0
        loss_D_value1 = 0
        loss_D_value2 = 0

        num_ex = 0

        optimizer.zero_grad()
        adjust_learning_rate(args, optimizer, i_iter)
        optimizer_D1.zero_grad()
        adjust_learning_rate_D(args, optimizer_D1, i_iter)
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(args, optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):
            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with target
            _, batch = targetloader_iter.__next__()
            images_n, images_d, _, _ = batch
            images_d = images_d.to(device)



            enhanced_images_d_pre, ci_map_d, Pr, filter_parameters = CNNPP(images_d)
            enhanced_images_d = enhanced_images_d_pre

            for i_d_pre in range(enhanced_images_d_pre.shape[0]):
                enhanced_images_d[i_d_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    enhanced_images_d_pre[i_d_pre,...])


            if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
                pred_target_d = model(enhanced_images_d)
            else:
                _, pred_target_d = model(enhanced_images_d)
            pred_target_d = interp_target(pred_target_d)
            D_out_d = model_D1(F.softmax(pred_target_d, dim=1))
            # D_out_d = model_D1(enhanced_images_d)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(source_label).to(device)
            loss_adv_target_d = weightedMSE(D_out_d, D_label_d)
            loss = 0.01 * loss_adv_target_d
            # loss = 0.01 * loss_adv_target_d

            loss = loss / args.iter_size
            loss.backward(retain_graph=True)
            loss_adv_target_value_D1 += loss_adv_target_d.item() / args.iter_size


            images_n = images_n.to(device)
            # r = lightnet(images_n)
            # enhanced_images_n = images_n + r
            # loss_enhance = 10*loss_TV(r)+torch.mean(loss_SSIM(enhanced_images_n, images_n))\
            #                + torch.mean(loss_exp_z(enhanced_images_n, mean_light))

            enhanced_images_n_pre, ci_map_n, Pr, filter_parameters = CNNPP(images_n)
            enhanced_images_n = enhanced_images_n_pre

            for i_n_pre in range(enhanced_images_n_pre.shape[0]):
                enhanced_images_n[i_n_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    enhanced_images_n_pre[i_n_pre,...])
            # enhanced_images_n = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(enhanced_images_n_pre)

            # loss_enhance = torch.mean(loss_SSIM(enhanced_images_n, images_n))\
            #                + torch.mean(loss_exp_z(enhanced_images_n, mean_light))

            if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
                pred_target_n = model(enhanced_images_n)
            else:
                _, pred_target_n = model(enhanced_images_n)
            pred_target_n = interp_target(pred_target_n)
            
            psudo_prob = torch.zeros_like(pred_target_d)
            threshold = torch.ones_like(pred_target_d[:,:11,:,:])*0.2
            threshold[pred_target_d[:,:11,:,:]>0.4] = 0.8
            psudo_prob[:,:11,:,:] = threshold*pred_target_d[:,:11,:,:].detach() + (1-threshold)*pred_target_n[:,:11,:,:].detach()
            psudo_prob[:,11:,:,:] = pred_target_n[:,11:,:,:].detach()

            weights_prob = weights.expand(psudo_prob.size()[0], psudo_prob.size()[3], psudo_prob.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            psudo_prob = psudo_prob*weights_prob
            psudo_gt = torch.argmax(psudo_prob.detach(), dim=1)
            psudo_gt[psudo_gt >= 11] = 255

            D_out_n_19 = model_D2(F.softmax(pred_target_n, dim=1))
            # D_out_n_19 = model_D2(enhanced_images_n)

            D_label_n_19 = torch.FloatTensor(D_out_n_19.data.size()).fill_(source_label).to(device)
            loss_adv_target_n_19 = weightedMSE(D_out_n_19, D_label_n_19,)

            # loss_pseudo = static_loss(pred_target_n[:,:11,:,:], psudo_gt)
            try:
                loss_pseudo = static_loss(pred_target_n[:, :11, :, :], psudo_gt)
            except:
                num_ex += 1
                # print("pred_target_mix_max:{},pred_target_mix_min:{}".format(pred_target_n.max(),pred_target_n.min()))
                print("num_ex:{}".format(num_ex))
                print("psudo_gt_mix_max:{},psudo_gt_mix_min:{}".format(psudo_gt.max(),psudo_gt.min()))

            loss = 0.01 * loss_adv_target_n_19 + loss_pseudo #+ 0.01 * loss_enhance
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value_D2 += loss_adv_target_n_19.item() / args.iter_size

            # train with source
            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            lowlight_param = random.uniform(1.5, 3)

            images_dark = np.power(images, lowlight_param)
            images = images.to(device)
            images_dark = images_dark.to(device)

            labels = labels.long().to(device)
            # labels_dark = labels

            # # train with source
            # _, batch = trainloader_iter.__next__()
            # images, labels, _, _ = batch
            # images = images.to(device)
            # labels = labels.long().to(device)

            # mix_mask = torch.zeros_like(labels)
            #
            # mix_mask[labels == random.randint(11, 18)] = 1
            # images_mix = images
            # if mix_mask.max() > 0:
            #     mix_mask_3d = torch.zeros_like(images)
            #     for i_mix in range(3):
            #         mix_mask_3d[:, i_mix, :, :] = mix_mask
            #
            #     # mix_mask = interp(mix_mask)
            #     images_mix = mix_mask_3d * images + (1 - mix_mask_3d) * images_n
            #     enhanced_images_mix_pre, ci_map_mix, Pr_mix, filter_parameters_mix = CNNPP(images_mix)
            #     enhanced_images_mix = enhanced_images_mix_pre
            #     if args.CI_FLAG:
            #         # enhanced_images_n = torch.cat((enhanced_images_n, ci_map_n), -1)
            #         enhanced_images_mix = enhanced_images_mix + ci_map_mix
            #
            #     for i_mix_pre in range(enhanced_images_mix_pre.shape[0]):
            #         enhanced_images_mix[i_mix_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            #             enhanced_images_mix_pre[i_mix_pre,...])
            #
            #     if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
            #         pred_target_mix = model(enhanced_images_mix)
            #     else:
            #         _, pred_target_mix = model(enhanced_images_mix)
            #     weights_prob = weights.expand(pred_target_mix.size()[0], pred_target_mix.size()[3], pred_target_mix.size()[2], 19)
            #     weights_prob = weights_prob.transpose(1, 3)
            #     pred_label_mix = pred_target_mix * weights_prob
            #     pred_label_mix = interp(pred_label_mix)
            #     psudo_gt_mix = torch.argmax(pred_label_mix.detach(), dim=1)
            #
            #     label_mix = mix_mask * labels + (1 - mix_mask) * psudo_gt_mix
            #     pred_target_mix = interp(pred_target_mix)
            #     loss_seg_mix = seg_loss(pred_target_mix, label_mix)
            #     loss_seg_mix_value += loss_seg_mix.item() / args.iter_size

            # ##### get the enhanced image
            # enhancement = images_d.cpu().data[0].numpy().transpose(1, 2, 0)
            # # enhancement = enhancement*mean_std[1]+mean_std[0]
            # enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
            # # enhancement = enhancement[:, :, ::-1]*255  # change to BGR
            # enhancement = enhancement * 255  # change to BGR
            # enhancement_d = Image.fromarray(enhancement.astype(np.uint8))
            # enhancement_d.save('./mix_d.png' )
            #
            # enhancement = images_mix.cpu().data[0].numpy().transpose(1, 2, 0)
            # # enhancement = enhancement*mean_std[1]+mean_std[0]
            # enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
            # # enhancement = enhancement[:, :, ::-1]*255  # change to BGR
            # enhancement = enhancement * 255  # change to BGR
            # enhancement = Image.fromarray(enhancement.astype(np.uint8))
            # enhancement.save('./mix.png' )

            # psudo_prob_mix = torch.zeros_like(pred_target_d)
            # threshold_mix = torch.ones_like(pred_target_d[:,11:,:,:])*0.2
            # threshold_mix[pred_target_d[:,11:,:,:]>0.4] = 0.8
            # psudo_prob_mix[:,11:,:,:] = threshold_mix*pred_target_d[:,11:,:,:].detach() + (1-threshold_mix)*pred_target_mix[:,11:,:,:].detach()
            # psudo_prob_mix[:,:11,:,:] = pred_target_mix[:,:11,:,:].detach()
            #
            # weights_prob_mix = weights.expand(psudo_prob.size()[0], psudo_prob.size()[3], psudo_prob.size()[2], 19)
            # weights_prob_mix = weights_prob_mix.transpose(1, 3)
            # psudo_prob_mix = psudo_prob_mix*weights_prob_mix
            # psudo_gt_mix = torch.argmax(psudo_prob_mix.detach(), dim=1)
            # psudo_gt_mix[psudo_gt_mix < 11] = 255
            # print("pred_target_mix_max:{},pred_target_mix_min:{}".format(pred_target_mix.max(), pred_target_mix.min()))
            # print("psudo_gt_mix_max:{},psudo_gt_mix_min:{}".format(psudo_gt_mix.max(), psudo_gt_mix.min()))
            # enhancement.save('./mix_imgs/mix_%s.png' % (str(i_iter)))


            # try:
            #     loss_pseudo_mix = dynamic_loss(pred_target_mix[:, 11:, :, :], psudo_gt_mix)
            # except:
            #     print("pred_target_mix_max:{},pred_target_mix_min:{}".format(pred_target_mix.max(),pred_target_mix.min()))
            #     print("psudo_gt_mix_max:{},psudo_gt_mix_min:{}".format(psudo_gt_mix.max(),psudo_gt_mix.min()))
            #
            #     ##### get the enhanced image
            #     enhancement = images_d.cpu().data[0].numpy().transpose(1, 2, 0)
            #     # enhancement = enhancement*mean_std[1]+mean_std[0]
            #     enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
            #     # enhancement = enhancement[:, :, ::-1]*255  # change to BGR
            #     enhancement = enhancement * 255  # change to BGR
            #     enhancement_d = Image.fromarray(enhancement.astype(np.uint8))
            #
            #     enhancement = images_mix.cpu().data[0].numpy().transpose(1, 2, 0)
            #     # enhancement = enhancement*mean_std[1]+mean_std[0]
            #     enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
            #     # enhancement = enhancement[:, :, ::-1]*255  # change to BGR
            #     enhancement = enhancement * 255  # change to BGR
            #     enhancement = Image.fromarray(enhancement.astype(np.uint8))
            #
            #
            #     enhancement.save('./mix_imgs/mix_%s.png'%(str(i_iter)) )
            #     enhancement_d.save('./mix_imgs/mix_d%s.png'%(str(i_iter)) )
            #
            #     output = interp(psudo_prob_mix).cpu().data[0].numpy()
            #     output = output.transpose(1, 2, 0)
            #     output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            #     output_col = colorize_mask(output)
            #     output_col.save('./mix_imgs/mix_gt%s.png'%(str(i_iter)) )
            #
            #     output = interp(pred_target_mix).cpu().data[0].numpy()
            #     output = output.transpose(1, 2, 0)
            #     output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            #     output_col = colorize_mask(output)
            #     output_col.save('./mix_imgs/mix_pre_n%s.png'%(str(i_iter)) )
            #






            # # train with source
            # _, batch = trainloader_iter.__next__()
            #
            # images, labels, _, _ = batch
            # # lowlight_param = random.uniform(1.5, 3)
            #
            # # images_dark = np.power(images, lowlight_param)
            # images = images.to(device)
            # # images_dark = images_dark.to(device)
            # # labels_dark = labels
            #
            # labels = labels.long().to(device)


            # r = lightnet(images)
            # enhanced_images = images + r
            # loss_enhance = 10 * loss_TV(r) + torch.mean(loss_SSIM(enhanced_images, images)) \
            #                + torch.mean(loss_exp_z(enhanced_images, mean_light))

            enhanced_images_pre, ci_map, Pr, filter_parameters = CNNPP(images)
            enhanced_images = enhanced_images_pre


            for i_pre in range(enhanced_images_pre.shape[0]):
                enhanced_images[i_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                    enhanced_images_pre[i_pre,...])
            # enhanced_images = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(enhanced_images_pre)

            # loss_enhance = torch.mean(loss_SSIM(enhanced_images, images)) \
            #                + torch.mean(loss_exp_z(enhanced_images, mean_light))
            # if args.CI_FLAG:
            #     # enhanced_images = torch.cat((enhanced_images, ci_map_n), -1)
            #     enhanced_images = enhanced_images + ci_map_n
            #
            # else:
            #     enhanced_images = enhanced_images
            if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
                pred_c = model(enhanced_images)
            else:
                _, pred_c = model(enhanced_images)
            pred_c = interp(pred_c)
            loss_seg = seg_loss(pred_c, labels)


            # enhanced_images_pre_dark, ci_map_n_dark, Pr_dark, filter_parameters_dark = CNNPP(images_dark)
            # enhanced_images_dark = enhanced_images_pre_dark
            # if args.CI_FLAG:
            #     # enhanced_images = torch.cat((enhanced_images, ci_map_n), -1)
            #     enhanced_images_dark = enhanced_images_dark + ci_map_n_dark
            # for i_pre in range(enhanced_images_pre_dark.shape[0]):
            #     enhanced_images_dark[i_pre,...] = standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            #         enhanced_images_pre_dark[i_pre,...])
            #
            # if args.model == 'RefineNet' or args.model.startswith('deeplabv3'):
            #     pred_c_dark = model(enhanced_images_dark)
            # else:
            #     _, pred_c_dark = model(enhanced_images_dark)
            # pred_c_dark = interp(pred_c_dark)
            # loss_seg_dark_dynamic = seg_dynamic_loss(pred_c_dark, labels)

            # labels_dark = labels.detach()
            # loss_dynamic_dark = dynamic_loss(pred_c_dark[:,11:,:,:], labels_dark)




            loss = loss_seg #+ loss_seg_dark_dynamic + loss_seg_mix #+ loss_seg_dark_dynamic #+ loss_enhance
            loss_s = loss / args.iter_size
            loss_s.backward()
            loss_seg_value += loss_seg.item() / args.iter_size
            # loss_seg_dark_value += loss_seg_dark_dynamic.item() / args.iter_size

            # train D
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred_c = pred_c.detach()
            enhanced_images = enhanced_images.detach()
            D_out1 = model_D1(F.softmax(pred_c, dim=1))
            # D_out1 = model_D1(enhanced_images)

            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            pred_c = pred_c.detach()
            enhanced_images = enhanced_images.detach()
            D_out2 = model_D2(F.softmax(pred_c, dim=1))
            # D_out2 = model_D2(enhanced_images)

            D_label2 = torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / args.iter_size /2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target_d = pred_target_d.detach()
            enhanced_images_d = enhanced_images_d.detach()
            D_out1 = model_D1(F.softmax(pred_target_d, dim=1))
            # D_out1 = model_D1(enhanced_images_d)
            D_label1 = torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            pred_target_n = pred_target_n.detach()
            enhanced_images_n = enhanced_images_n.detach()
            D_out2 = model_D2(F.softmax(pred_target_n, dim=1))
            # D_out2 = model_D2(enhanced_images_n)

            D_label2 = torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / args.iter_size / 2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        if i_iter % 100 == 0:# and i_iter != 0:
            print(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_D1 = {3:.3f}, loss_D1 = {4:.3f}, loss_D2 = {5:.3f}, loss_pseudo = {6:.3f}, '
                'loss_seg_dark = {7:.3f}, loss_adv_D2 = {8:.3f}'.format(
                    i_iter, args.num_steps, loss_seg_value,
                    loss_adv_target_value_D1, loss_D_value1, loss_D_value2, loss_pseudo, loss_seg_dark_value, loss_adv_target_value_D2))
            # ##### get the enhanced image
            # enhancement = images_n.cpu().data[0].numpy().transpose(1, 2, 0)
            # # enhancement = enhancement*mean_std[1]+mean_std[0]
            # enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
            # # enhancement = enhancement[:, :, ::-1]*255  # change to BGR
            # enhancement = enhancement * 255  # change to BGR
            # enhancement_n = Image.fromarray(enhancement.astype(np.uint8))
            # enhancement_n.save('/data/vdd/liuwenyu/DANNet/mix_imgs/mix_%s_n.png' % (str(i_iter)))
            #
            # enhancement = images_mix.cpu().data[0].numpy().transpose(1, 2, 0)
            # # enhancement = enhancement*mean_std[1]+mean_std[0]
            # enhancement = (enhancement - enhancement.min()) / (enhancement.max() - enhancement.min())
            # # enhancement = enhancement[:, :, ::-1]*255  # change to BGR
            # enhancement = enhancement * 255  # change to BGR
            # enhancement = Image.fromarray(enhancement.astype(np.uint8))
            # enhancement.save('/data/vdd/liuwenyu/DANNet/mix_imgs/mix_%s.png' % (str(i_iter)))

        if i_iter < 40000:
            if i_iter % 10000 == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'ianet' + str(i_iter) + '.pth'))
                torch.save(CNNPP.state_dict(), os.path.join(args.snapshot_dir, 'ianet_cnnpp' + str(i_iter) + '.pth'))
                # torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d1_' + str(i_iter) + '.pth'))
                # torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))
        elif i_iter < 45000:
            if i_iter % 2500 == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'ianet' + str(i_iter) + '.pth'))
                torch.save(CNNPP.state_dict(), os.path.join(args.snapshot_dir, 'ianet_cnnpp' + str(i_iter) + '.pth'))
        else:
            if i_iter % 1000 == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'ianet' + str(i_iter) + '.pth'))
                torch.save(CNNPP.state_dict(), os.path.join(args.snapshot_dir, 'ianet_cnnpp' + str(i_iter) + '.pth'))
                # torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d1_' + str(i_iter) + '.pth'))
                # torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))


if __name__ == '__main__':
    main()
