import os
import time
import torch
import torchvision

import numpy as np
import pandas as pd 
import torch.nn as nn
import torch.optim as optim
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm
from torchvision import models
from collections import defaultdict

from options import Options
from lib.data.dataloader import load_data
from lib.models.TWO_STREAM import TWO_STREAM, GLP
from lib.utils import seed_everything, AverageMeter, accuracy, to_python_float
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

def load_two_stram(opt):

    #load pretrain imagenet model
    # print("=> using pre-trained model '{}'".format(opt.model))
    pre_model = models.__dict__[opt.model](pretrained=False)

    if opt.model == 'resnet18': 
        is_inception = False
        pre_model= nn.Sequential(*list(pre_model.children())[:-2])

    elif opt.model == 'inception_v3': 
        is_inception = True
        num_ftrs = pre_model.AuxLogits.fc.in_features
        pre_model.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        torch.nn.init.xavier_uniform_(pre_model.AuxLogits.fc.weight)
        pre_model.fc = nn.Identity()

    pre_model = pre_model.cuda(1)
    pre_model.eval()

    #load glp model
    glp_model = GLP(opt)
    glp_model = glp_model.cuda(1)
    glp_model.eval()

    model = TWO_STREAM(pre_model,glp_model)
    model.load_state_dict(torch.load(opt.two_stream_path))
    model=model.cuda(1)
    model.eval()

    return model

def test_on_attacks(opt,data):
    
    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")


    results = defaultdict(list)
    two_stream_model = load_two_stram(opt)
    
    #load pretrain imagenet model
    print("=> using pre-trained model '{}'".format(opt.model))
    model_imgnet = models.__dict__[opt.model](pretrained=False)

    if opt.model == 'resnet18': 
        num_ftrs = model_imgnet.fc.in_features
        model_imgnet.fc = nn.Linear(num_ftrs, opt.num_classes)

    elif opt.model == 'inception_v3': 
        is_inception = True
        num_ftrs = model_imgnet.AuxLogits.fc.in_features
        model_imgnet.AuxLogits.fc = nn.Linear(num_ftrs, opt.num_classes)
        num_ftrs = model_imgnet.fc.in_features
        model_imgnet.fc = nn.Linear(num_ftrs, opt.num_classes)

    model_imgnet.load_state_dict(torch.load(opt.pretrained_path))
    model_imgnet = model_imgnet.cuda(1)
    model_imgnet.eval()

    end = time.time()
    

    epsilons = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.5]

    for eps in epsilons:

        fgsm_top1_avg = AverageMeter()
        fgsm_top1_org_avg = AverageMeter()
        pgd_top1_avg = AverageMeter()
        pgd_top1_org_avg = AverageMeter()

        fgsm_top5_avg = AverageMeter()
        fgsm_top5_org_avg = AverageMeter()
        pgd_top5_avg = AverageMeter()
        pgd_top5_org_avg = AverageMeter()

        fgsm_top10_avg = AverageMeter()
        fgsm_top10_org_avg = AverageMeter()
        pgd_top10_avg = AverageMeter()
        pgd_top10_org_avg = AverageMeter()

        fgsm_fscore_avg = AverageMeter()
        fgsm_fscore_org_avg = AverageMeter()
        pgd_fscore_avg = AverageMeter()
        pgd_fscore_org_avg = AverageMeter()

        fgsm_precisions_avg = AverageMeter()
        fgsm_precisions_org_avg = AverageMeter()
        pgd_precisions_avg = AverageMeter()
        pgd_precisions_org_avg = AverageMeter()

        fgsm_recalls_avg = AverageMeter()
        fgsm_recalls_org_avg = AverageMeter()
        pgd_recalls_avg = AverageMeter()
        pgd_recalls_org_avg = AverageMeter()
     
        for i in range(opt.repeat_on_attacks):
            
            batch_time = AverageMeter()
            inf_time = AverageMeter()
            inf_time_org = AverageMeter()

            top1 = AverageMeter()
            top5 = AverageMeter()
            top10 = AverageMeter()
            fscore = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()

            top1_org = AverageMeter()
            top5_org = AverageMeter()
            top10_org = AverageMeter()
            fscore_org = AverageMeter()
            precisions_org = AverageMeter()
            recalls_org = AverageMeter()
          
            pgd_top1 = AverageMeter()
            pgd_top5 = AverageMeter()
            pgd_top10 = AverageMeter()
            pgd_fscore = AverageMeter()
            pgd_precisions = AverageMeter()
            pgd_recalls = AverageMeter()

            pgd_top1_org = AverageMeter()
            pgd_top5_org = AverageMeter()
            pgd_top10_org = AverageMeter()
            pgd_fscore_org = AverageMeter()
            pgd_precisions_org = AverageMeter()
            pgd_recalls_org = AverageMeter()
       

            for images, labels in tqdm(data.test):
                added_res=None
                images = images.cuda(1)
                labels = labels.squeeze(-1).long().cuda(1)
                test_loader_l = len(data.test.dataset)

                x_fgm_img = fast_gradient_method(model_imgnet, images, eps, np.inf)
                x_fgm = fast_gradient_method(two_stream_model, images, eps, np.inf)
                
                x_pgd_img = projected_gradient_descent(model_imgnet, images, eps, eps/3, 5, np.inf)
                x_pgd = projected_gradient_descent(two_stream_model, images, eps, eps/3, 5, np.inf)



                # compute output
                with torch.no_grad():
     
                    tic = time.time()
                    y_fgm_img = model_imgnet(x_fgm_img)
                    toc = time.time()
                    y_fgm = two_stream_model(x_fgm)

                    inf_time.update(time.time()-toc )
                    inf_time_org.update(toc - tic)

                    y_pd_img = model_imgnet(x_pgd_img)
                    y_pd = two_stream_model(x_pgd)

     
                prec1, prec5, prec10, f1score, precision, recall = accuracy(y_fgm.data, labels, topk=(1, 5, 10))
                prec1_org, prec5_org, prec10_org, f1score_org, precision_org, recall_org = accuracy(y_fgm_img.data, labels, topk=(1, 5, 10))
                
                pgd_prec1, pgd_prec5, pgd_prec10, pgd_f1score, pgd_precision, pgd_recall = accuracy(y_pd.data, labels, topk=(1, 5, 10))
                pgd_prec1_org, pgd_prec5_org, pgd_prec10_org, pgd_f1score_org, pgd_precision_org, pgd_recall_org = accuracy(y_pd_img.data, labels, topk=(1, 5, 10))


                top1.update(to_python_float(prec1), images.size(0))
                top5.update(to_python_float(prec5), images.size(0))
                top10.update(to_python_float(prec10), images.size(0))
                fscore.update(f1score, images.size(0))
                precisions.update(precision, images.size(0))
                recalls.update(recall, images.size(0))
                
                top1_org.update(to_python_float(prec1_org), images.size(0))
                top5_org.update(to_python_float(prec5_org), images.size(0))
                top10_org.update(to_python_float(prec10_org), images.size(0))
                fscore_org.update(f1score_org, images.size(0))
                precisions_org.update(precision_org, images.size(0))
                recalls_org.update(recall_org, images.size(0))

                pgd_top1.update(to_python_float(pgd_prec1), images.size(0))
                pgd_top5.update(to_python_float(pgd_prec5), images.size(0))
                pgd_top10.update(to_python_float(pgd_prec10), images.size(0))
                pgd_fscore.update(pgd_f1score, images.size(0))
                pgd_precisions.update(pgd_precision, images.size(0))
                pgd_recalls.update(pgd_recall, images.size(0))
                
                pgd_top1_org.update(to_python_float(pgd_prec1_org), images.size(0))
                pgd_top5_org.update(to_python_float(pgd_prec5_org), images.size(0))
                pgd_top10_org.update(to_python_float(pgd_prec10_org), images.size(0))
                pgd_fscore_org.update(pgd_f1score_org, images.size(0))
                pgd_precisions_org.update(pgd_precision_org, images.size(0))
                pgd_recalls_org.update(pgd_recall_org, images.size(0))
                


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
           
            results['esp_'+str(eps)+'_rep_'+ str(i)]=[top1_org.avg,top1.avg,pgd_top1_org.avg,pgd_top1.avg,
                          top5_org.avg,top5.avg,pgd_top5_org.avg,pgd_top5.avg,
                          top10_org.avg,top10.avg,pgd_top10_org.avg,pgd_top10.avg,
                          fscore_org.avg,fscore.avg,pgd_fscore_org.avg,pgd_fscore.avg,
                          precisions_org.avg,precisions.avg,pgd_precisions_org.avg,pgd_precisions.avg,
                          recalls_org.avg,recalls.avg,pgd_recalls_org.avg,pgd_recalls.avg  ]
        
          
            
            print(f'test acc on FSGM TWO_STREAM model eps {eps}:')
            print(' * Top@1 {top1.avg:.3f} Top@5 {top5.avg:.3f} Top@10 {top10.avg:.3f} \
                F1-Score {fscore.avg:.3f} \
                    precision {precisions.avg:.3f} recall {recalls.avg:.3f} \n'
                .format(top1=top1, top5=top5, top10=top10, fscore=fscore, precisions=precisions , recalls=recalls))

            print(f'test acc on FSGM pre-trained on imagenet eps {eps}:')
            print(' * Top@1 {top1_org.avg:.3f} Top@5 {top5_org.avg:.3f} Top@10 {top10_org.avg:.3f} \
                    F1-Score {fscore_org.avg:.3f} \
                        precision {precisions_org.avg:.3f} recall {recalls_org.avg:.3f} \n '
                    .format(top1_org=top1_org, top5_org=top5_org, top10_org=top10_org, fscore_org=fscore_org, \
                        precisions_org=precisions_org , recalls_org=recalls_org))

            print(f'test acc on PGD TWO_STREAM model eps {eps}:')
            print(' * Top@1 {pgd_top1.avg:.3f} Top@5 {pgd_top5.avg:.3f} Top@10 {pgd_top10.avg:.3f} \
                F1-Score {pgd_fscore.avg:.3f} \
                    precision {pgd_precisions.avg:.3f} recall {pgd_recalls.avg:.3f} \n'
                .format(pgd_top1=pgd_top1, pgd_top5=pgd_top5, pgd_top10=pgd_top10, pgd_fscore=pgd_fscore, pgd_precisions=pgd_precisions , pgd_recalls=recalls))

            print(f'test acc on PGD pre-trained on imagenet eps {eps}:')
            print(' * Top@1 {pgd_top1_org.avg:.3f} Top@5 {pgd_top5_org.avg:.3f} Top@10 {pgd_top10_org.avg:.3f} \
                    F1-Score {pgd_fscore_org.avg:.3f} \
                        precision {pgd_precisions_org.avg:.3f} recall {pgd_recalls_org.avg:.3f} \n '
                    .format(pgd_top1_org=pgd_top1_org, pgd_top5_org=pgd_top5_org, pgd_top10_org=pgd_top10_org, pgd_fscore_org=pgd_fscore_org, \
                        pgd_precisions_org=pgd_precisions_org , pgd_recalls_org=pgd_recalls_org))
        
            fgsm_top1_avg.update(top1.avg)
            fgsm_top1_org_avg.update(top1_org.avg)
            pgd_top1_avg.update(pgd_top1.avg)
            pgd_top1_org_avg.update(pgd_top1_org.avg)


            fgsm_top5_avg.update(top5.avg)
            fgsm_top5_org_avg.update(top5_org.avg)
            pgd_top5_avg.update(pgd_top5.avg)
            pgd_top5_org_avg.update(pgd_top5_org.avg)


            fgsm_top10_avg.update(top10.avg)
            fgsm_top10_org_avg.update(top10_org.avg)
            pgd_top10_avg.update(pgd_top10.avg)
            pgd_top10_org_avg.update(pgd_top10_org.avg)


            fgsm_fscore_avg.update(fscore.avg)
            fgsm_fscore_org_avg.update(fscore_org.avg)
            pgd_fscore_avg.update(pgd_fscore.avg)
            pgd_fscore_org_avg.update(pgd_fscore_org.avg)


            fgsm_precisions_avg.update(precisions.avg)
            fgsm_precisions_org_avg.update(precisions_org.avg)
            pgd_precisions_avg.update(pgd_precisions.avg)
            pgd_precisions_org_avg.update(pgd_precisions_org.avg)


            fgsm_recalls_avg.update(recalls.avg)
            fgsm_recalls_org_avg.update(recalls_org.avg)
            pgd_recalls_avg.update(pgd_recalls.avg)
            pgd_recalls_org_avg.update(pgd_recalls_org.avg)
            

        print(' * FGSAM_Top@1 {fgsm_top1_avg.avg:.3f} FGSM_TOP1_org {fgsm_top1_org_avg.avg:.3f} PGD_TOP1 {pgd_top1_avg.avg:.3f} \
        PGD_TOP1_org {pgd_top1_org_avg.avg:.3f} \n'
        .format(fgsm_top1_avg=fgsm_top1_avg, fgsm_top1_org_avg=fgsm_top1_org_avg, pgd_top1_avg=pgd_top1_avg, pgd_top1_org_avg=pgd_top1_org_avg))

        results['esp_'+str(eps)+'_avg']=[fgsm_top1_org_avg.avg,fgsm_top1_avg.avg,pgd_top1_org_avg.avg,pgd_top1_avg.avg,
                  fgsm_top5_org_avg.avg,fgsm_top5_avg.avg,pgd_top5_org_avg.avg,pgd_top5_avg.avg,
                  fgsm_top10_org_avg.avg,fgsm_top10_avg.avg,pgd_top10_org_avg.avg,pgd_top10_avg.avg,
                  fgsm_fscore_org_avg.avg,fgsm_fscore_avg.avg,pgd_fscore_org_avg.avg,pgd_fscore_avg.avg,
                  fgsm_precisions_org_avg.avg,fgsm_precisions_avg.avg,pgd_precisions_org_avg.avg,pgd_precisions_avg.avg,
                  fgsm_recalls_org_avg.avg,fgsm_recalls_avg.avg,pgd_recalls_org_avg.avg,pgd_recalls_avg.avg  ]
     
            
        # print(f'Average inference time: imagenet model {inf_time_org.sum/len(data.test.dataset)} , two_stream {inf_time.sum/len(data.test.dataset)}')
       

    df = pd.DataFrame.from_dict(results, orient='index',
                                    columns =['FGSM_TOP1_'+opt.model, 'FGSM_TOP1_TWO_STREAM',
                                           'PGD_TOP1_'+opt.model, 'PGD_TOP1_TWO_STREAM',

                                           'FGSM_TOP5_'+opt.model, 'FGSM_TOP5_TWO_STREAM',
                                           'PGD_TOP5_'+opt.model, 'PGD_TOP5_TWO_STREAM',

                                            'FGSM_TOP10_'+opt.model, 'FGSM_TOP10_TWO_STREAM',
                                           'PGD_TOP10_'+opt.model, 'PGD_TOP10_TWO_STREAM',

                                            'FGSM_fscore_'+opt.model, 'FGSM_fscore_TWO_STREAM',
                                           'PGD_fscore_'+opt.model, 'PGD_fscore_TWO_STREAM',

                                            'FGSM_precisions_'+opt.model, 'FGSM_precisions_TWO_STREAM',
                                           'PGD_precisions_'+opt.model, 'PGD_precisions_TWO_STREAM',

                                            'FGSM_recalls_'+opt.model, 'FGSM_recalls_TWO_STREAM',
                                           'PGD_recalls_'+opt.model, 'PGD_recalls_TWO_STREAM' 
                                           ])
    df.to_csv( opt.save_csv)
    print('the results are saved to', opt.save_csv)


def main():
    opt = Options().parse()

    '''SEED Everything'''
    seed_everything(SEED=opt.seed)


    data = load_data(opt)
    test_on_attacks(opt,data)



if __name__ == "__main__" :
    main()