import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from utils.util import *
from utils.util import AveragePrecisionMeter
from torch.cuda.amp import GradScaler, autocast
import torchnet as tnt
from torch.optim import lr_scheduler
from src.data_loader.coco_fsl import CocoDatasetAugmentation
from src.data_loader.nus_fsl import NUSWIDEClassification_fsl
from src.data_loader.voc_fsl import Voc2007Classification_fsl
from randaugment import RandAugment
# from exp import *
import os
# use_cos = True

import torch.nn.functional as F

class MultiLabelEngine():
    def __init__(self, args):
        # hyper-parameters
        self.evaluation = args['evaluation']
        self.thre = args['threshold']
        self.max_epoch = args['max_epoch']
        self.epis = args['epis']
        self.data = args['data']
        self.image_size = args['image_size']
        self.workers = args['workers']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.threshold = args['threshold']
        self.resume = args['resume']
        self.ratio = args['lamda']
        self.alpha = args['alpha']
        print('-----------test epis: '+str(self.epis))

        # measure mAP
        print("thre:", self.thre)
        self.regular_ap_meter = AveragePrecisionMeter(threshold=self.threshold, difficult_examples=False)
        self.regular_loss_meter = tnt.meter.AverageValueMeter()

        

    def meter_reset(self):
        self.regular_ap_meter.reset()
        self.regular_loss_meter.reset()

    # def meter_print_val(self):
    #     print("starting metric r......")
    #     regular_ap = 100 * self.regular_ap_meter.value()
    #     regular_map = regular_ap.mean()

    #     print('=================================================>>>>>>> Experimental Results')
    #     print('regular mAP score: {map:.3f}'.format(map=regular_map))


    #     OP, OR, OF1, CP, CR, CF1 = self.regular_ap_meter.overall()
    #     OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.regular_ap_meter.overall_topk(3)
    #     print('CP: {CP:.4f}\t'
    #           'CR: {CR:.4f}\t'
    #           'CF1: {CF1:.4f}'
    #           'OP: {OP:.4f}\t'
    #           'OR: {OR:.4f}\t'
    #           'OF1: {OF1:.4f}\t'.format(CP=CP, CR=CR,
    #                                   CF1=CF1, OP=OP, OR=OR, OF1=OF1))
    #     print('OP_3: {OP:.4f}\t'
    #           'OR_3: {OR:.4f}\t'
    #           'OF1_3: {OF1:.4f}\t'
    #           'CP_3: {CP:.4f}\t'
    #           'CR_3: {CR:.4f}\t'
    #           'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
    #                                     CR=CR_k, CF1=CF1_k))
        

    #     return regular_map


    def meter_print(self,model_name):
        regular_loss = self.regular_loss_meter.value()[0]
        regular_ap = 100 * self.regular_ap_meter.value()
        regular_map = regular_ap.mean()
        reg_meters = self.regular_ap_meter.overall()
        OP, OR, OF1, CP, CR, CF1 = reg_meters
        reg_topk = self.regular_ap_meter.overall_topk(3)
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = reg_topk
        print('=================================================>>>>>>> Experimental Results on regular {}'.format(model_name))
        print('mAP score: {map:.3f}\t loss: {loss:.3f}'.format(map=regular_map, loss=regular_loss))
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR,
                                      CF1=CF1))
        print('OP_3: {OP:.4f}\t'
              'OR_3: {OR:.4f}\t'
              'OF1_3: {OF1:.4f}\t'
              'CP_3: {CP:.4f}\t'
              'CR_3: {CR:.4f}\t'
              'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k,
                                        CR=CR_k, CF1=CF1_k))

        return regular_map, reg_meters, reg_topk
    

    def learning(self, model, clip_model, criterion, optimizer, filtered_dict, model_name,shot='5',dataset = 'coco', inp_seman=''):
        regular_maps = []
        reg_meter_ls = []
        # filtered_dict['fc.3.weight'] = filtered_dict.pop('fc.2.weight')
        # filtered_dict['fc.3.bias'] = filtered_dict.pop('fc.2.bias')
        try:
            for epi in self.epis:
                print("=> loading checkpoint '{}'".format(self.resume))
                # model.load_state_dict(filtered_dict)
                highest_regular_map = 0
                highest_reg_meters = None
                if dataset == 'coco':
                    train_dataset = CocoDatasetAugmentation(root_dir=self.data,
                                            used_ind_path = './idx/'+shot+'shotRun'+epi+'UsedIndices.pkl', 
                                            class_ind_dict_path = './idx/'+shot+'shotRun'+epi+'ClassIdxDict.pkl',
                                            set_name='train2014',
                                            transform=transforms.Compose([
                                                transforms.Resize((self.image_size, self.image_size)),
                                                RandAugment(),
                                                transforms.ToTensor(),
                                            ]),
                                            inp_name=inp_seman
                                            )

                    val_dataset = CocoDatasetAugmentation(root_dir=self.data,
                                            used_ind_path = './idx/'+shot+'shotRun'+epi+'UsedIndices.pkl', 
                                            class_ind_dict_path = './idx/'+shot+'shotRun'+epi+'ClassIdxDict.pkl',
                                            set_name='val2014',
                                            transform=transforms.Compose([
                                            transforms.Resize((self.image_size, self.image_size)),
                                            transforms.ToTensor(),
                                            ]),
                                            inp_name=inp_seman
                                            )
                elif dataset == 'nus':
                    train_dataset = NUSWIDEClassification_fsl(self.data, 'idx/nus/classification_Train_novel_{}shot_{}.csv'.format(shot,epi),
                                                word_emb_file='data/nuswide/nuswide_glove_word2vec.pkl',
                                                transform=transforms.Compose([
                                                    transforms.Resize((self.image_size, self.image_size)),
                                                    RandAugment(),
                                                    transforms.ToTensor(),
                                                ]))
                    val_dataset = NUSWIDEClassification_fsl(self.data, 'idx/nus/classification_Test_novel.csv',
                                                word_emb_file='data/nuswide/nuswide_glove_word2vec.pkl',
                                                transform=transforms.Compose([
                                                    transforms.Resize((self.image_size, self.image_size)),
                                                    transforms.ToTensor()
                                                ]))
                else:
                    train_dataset = Voc2007Classification_fsl(self.data, 'idx/voc/classification_Train_novel_{}shot_{}.csv'.format(shot,epi),
                                                word_emb_file='data/voc/voc_glove_word2vec.pkl',
                                                transform=transforms.Compose([
                                                    transforms.Resize((self.image_size, self.image_size)),
                                                    RandAugment(),
                                                    transforms.ToTensor(),
                                                ]))
                    val_dataset = Voc2007Classification_fsl(self.data, 'idx/voc/classification_Test_novel.csv',
                                                word_emb_file='data/voc/voc_glove_word2vec.pkl',
                                                transform=transforms.Compose([
                                                    transforms.Resize((self.image_size, self.image_size)),
                                                    transforms.ToTensor()
                                                ]))
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True,
                    num_workers=self.workers, pin_memory=False)

                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.workers, pin_memory=False)

                steps_per_epoch = len(train_loader)
                scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, epochs=self.max_epoch,
                                                    pct_start=0.1)
                if self.evaluation:
                    model.eval()
                    self.meter_reset()
                    self.validate(model, val_loader)
                    _ = self.meter_print(model_name)
                else:
                    scaler = GradScaler()
                    for epoch in range(self.max_epoch):
                        # train step
                        model.train()
                        self.meter_reset()
                        self.train(model, clip_model, train_loader, criterion, optimizer, scheduler, scaler, epoch, epi)
                        # evaluate step
                        if (epoch%10==9 and  (epoch >= 25 and epoch < 50))or (epoch%10==9 and  (epoch >= 50 and epoch <=80)):
                            _= self.meter_print(model_name)
                            model.eval()
                            self.meter_reset()
                            try:
                                self.validate(model, clip_model, val_loader, criterion, epi)
                                regular_map, reg_meters, reg_topk = self.meter_print(model_name)
                                if regular_map > highest_regular_map:
                                    highest_regular_map = regular_map
                                    highest_reg_meters = [reg_meters, reg_topk]
                                print('======================>>>>>>> Highest Experimental Results <<<<<<<======================')
                                print('Highest regular {} model mAP for epi {}, epoch {}: {}, alpha {}'.format(model_name, epi, epoch, highest_regular_map, self.alpha))
                                
                            except KeyboardInterrupt:
                                print(str(regular_maps))
                                print(str(reg_meter_ls))
                            model.train()
                        elif epoch >80:
                            break
                        else:
                            pass
                    regular_maps.append(highest_regular_map)
                    reg_meter_ls.append(highest_reg_meters)
        except Exception as e:
            print("Error happened!") 
            print(e)                
        return regular_maps, reg_meter_ls

    def train(self, model, clip_model, train_loader, criterion, optimizer, scheduler, scaler, epoch, epi):
        regular_model = model
        train_loader = tqdm(train_loader, desc='Train Epi {} Epoch {}'.format(epi, epoch))
        for i, (inputData, target, semantic) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            semantic = semantic[0].cuda().float()
            with autocast():  # mixed precision
                _, _,cls_output, _,dist_feat = regular_model(inputData)
                # cls_output, _,dist_feat = regular_model(inputData)
                cls_output = cls_output.float()

            cls_loss = criterion[0](cls_output, target)
            # cls_loss = ranking_lossT(cls_output, target)

            # with torch.no_grad():
            # _, tea_dist_feat = clip_model.encode_image(inputData)
            # tea_dist_feat = tea_dist_feat.float()

            # dist_loss = F.l1_loss(dist_feat.float(), tea_dist_feat)
            # dist_loss = F.smooth_l1_loss(dist_feat.float(), tea_dist_feat)
            regular_loss = cls_loss

            # if epoch%10 < 5 :
            #     self.set_layer(regular_model.attention,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.attention,freeze = False)

            # elif epoch%10 >= 5 and epoch%10 < 9:
            #     self.set_layer(regular_model.label_emb,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.label_emb,freeze = False)

            # else:
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()

            
            # if epoch%10 < 5 :
            #     self.set_layer(regular_model.label_emb,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.label_emb,freeze = False)

            # elif epoch%10 >= 5 and epoch%10 < 9:
            #     self.set_layer(regular_model.attention,freeze = True)
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            #     self.set_layer(regular_model.attention,freeze = False)

            # else:
            #     regular_model.zero_grad()
            #     scaler.scale(regular_loss).backward()
            #     # loss.backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     # optimizer.step()
            #     scheduler.step()
            regular_model.zero_grad()
            scaler.scale(regular_loss).backward()
            # loss.backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            scheduler.step()

            # store information
            train_loader.set_postfix(cls=cls_loss.item())
            self.regular_ap_meter.add(cls_output.data, target)
            self.regular_loss_meter.add(regular_loss.item())

    def validate(self, model, clip_model, val_loader, criterion, epi):
        regular_model = model
        val_loader = tqdm(val_loader, desc='Test Epi {}'.format(epi))
        for i, (inputData, target, semantic) in enumerate(val_loader):
            # compute output
            target = target.cuda()
            semantic = semantic[0].cuda().float()
            # regular model
            with torch.no_grad():
                with autocast():
                    # regular model
                    _, _,regular_cls_output, _,regular_dist_feat = regular_model(inputData.cuda())
                    # regular_cls_output, _,regular_dist_feat = regular_model(inputData.cuda())
                    regular_cls_output = regular_cls_output.float()

                    regular_overall_loss = criterion[0](regular_cls_output, target)
                    # regular_overall_loss = ranking_lossT(regular_cls_output, target)
                    # _, regular_tea_dist_feat = clip_model.encode_image(inputData.cuda())

                    # regular_dist_loss = F.l1_loss(regular_dist_feat.float(), regular_tea_dist_feat.float())
                    # regular_dist_loss = F.smooth_l1_loss(regular_dist_feat.float(), regular_tea_dist_feat.float())

                    regular_loss = regular_overall_loss

            val_loader.set_postfix(cls=regular_overall_loss.item())
            self.regular_ap_meter.add(regular_cls_output.data, target)
            self.regular_loss_meter.add(regular_loss.item())

