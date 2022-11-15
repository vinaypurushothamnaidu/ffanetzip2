import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
#from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,log_dir
from data_utils import *
print('log_dir :',log_dir)
print('model_name:',model_name)

models_={
	'ffa':FFA(gps=opt.gps,blocks=opt.blocks),
}
loaders_={
	'its_train':train_loader,
	'its_test':test_loader
}
start_time=time.time()
T=opt.steps	
def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]
	if opt.resume and os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		losses=ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step=ckp['step']
		max_ssim=ckp['max_ssim']
		max_psnr=ckp['max_psnr']
		psnrs=ckp['psnrs']
		ssims=ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=opt.lr
		if not opt.no_lr_sche:
			lr=lr_schedule_cosdecay(step,T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr  
		x,y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device)
		out=net(x)
		loss=criterion[0](out,y)
		if opt.perloss:
			loss2=criterion[1](out,y)
			loss=loss+0.04*loss2
		
		loss.backward()
		
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time()-start_time)/60 :.1f}',end='',flush=True)

		if step % opt.eval_step ==0 :
			with torch.no_grad():
				ssim_eval,psnr_eval=test(net,loader_test, max_psnr,max_ssim,step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim and psnr_eval > max_psnr :
				max_ssim=max(max_ssim,ssim_eval)
				max_psnr=max(max_psnr,psnr_eval)
				# torch.save({
				# 			'step':step,
				# 			'max_psnr':max_psnr,
				# 			'max_ssim':max_ssim,
				# 			'ssims':ssims,
				# 			'psnrs':psnrs,
				# 			'losses':losses,
				# 			'model':net.state_dict()
				# },opt.model_dir)
				# print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
				torch.save({
							'step':step,
							'max_psnr':max_psnr,
							'max_ssim':max_ssim,
							'ssims':ssims,
							'psnrs':psnrs,
							'losses':losses,
							'model':net.state_dict()
				},"/kaggle/working/ffanetzip2/ffamodelkaggle.pk")
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')


def test(net,loader_test,max_psnr,max_ssim,step):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims) ,np.mean(psnrs)


if __name__ == "__main__":
	loader_train=loaders_[opt.trainset]
	loader_test=loaders_[opt.testset]
	net=models_[opt.net]
	net=net.to(opt.device)
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
	

