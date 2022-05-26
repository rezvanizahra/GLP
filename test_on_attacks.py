import os
import time
import torch

import numpy as np
import pandas as pd 

from tqdm import tqdm
from collections import defaultdict

from options import Options
from lib.models import load_model
from lib.data.dataloader import load_data
from lib.models.inception_v3 import inception_v3
from lib.models.models import GLP, GAS
from lib.utils import seed_everything, AverageMeter, accuracy, to_python_float
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


def test_on_attacks(opt,model, data):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = defaultdict(list)

    model.eval()

    end = time.time()

    epsilons = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.5]

    for eps in epsilons:

        top1_avg = AverageMeter()
        top5_avg = AverageMeter()
        top10_avg = AverageMeter()
     
        for i in range(opt.repeat_on_attacks):
            
            batch_time = AverageMeter()
            inf_time = AverageMeter()
          
            top1 = AverageMeter()
            top5 = AverageMeter()
            top10 = AverageMeter()
      
            correct = 0
            total = 0

            for images, labels in tqdm(data.test):
                added_res=None
                images = images.to(device)
                labels = labels.squeeze(-1).long().to(device)
                test_loader_l = len(data.test.dataset)
                
                if opt.attack_method == 'FGSM':
                    x_attack= fast_gradient_method(model, images, eps, np.inf)
                
                elif opt.attack_method == 'PGD':
                    x_attack = projected_gradient_descent(model, images, eps, eps/3, 5, np.inf)
                else:
                    raise ValueError('The attack is not supported')


                # compute output
                with torch.no_grad():
     
                    tic = time.time()
                    y_attack = model(x_attack)
                    inf_time.update(time.time()-tic )

             
                prec1, prec5, prec10, _, _, _ = accuracy(y_attack.data, labels, topk=(1, 5, 10))
             

                top1.update(to_python_float(prec1), images.size(0))
                top5.update(to_python_float(prec5), images.size(0))
                top10.update(to_python_float(prec10), images.size(0))
              
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
           
            results['esp_'+str(eps)+'_rep_'+ str(i)]=[top1.avg,top5.avg,top10.avg]
                         
        
            print(f'test acc on {opt.attack_method} on {opt.model}  for eps {eps}:')
            print(' * Top@1 {top1.avg:.3f} Top@5 {top5.avg:.3f} Top@10 {top10.avg:.3f} \n'
                .format(top1=top1, top5=top5, top10=top10))

            top1_avg.update(top1.avg)
            top5_avg.update(top5.avg)
            top10_avg.update(top10.avg)
      
            
        
        results['esp_'+str(eps)+'_avg']=[top1_avg.avg,top5_avg.avg,top10_avg.avg ]
     
        print(f'Average inference time: {inf_time.sum/len(data.test.dataset)}')
       

    df = pd.DataFrame.from_dict(results, orient='index',
                                    columns =['TOP1_'+opt.model,'TOP5_'+opt.model,'TOP10_'+opt.model])
    df.to_csv( opt.save_csv)
    print('the results are saved to', opt.save_csv)


def main():
    opt = Options().parse()

    '''SEED Everything'''
    seed_everything(SEED=opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      
    data = load_data(opt)
    model = load_model(opt)
    model.load_state_dict(torch.load(opt.checkpoint_path, map_location='cpu'))
    model = model.to(device)

    test_on_attacks(opt,model, data)



if __name__ == "__main__" :
    main()