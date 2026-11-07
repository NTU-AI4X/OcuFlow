#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, random
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import checkpointing
import config
import core as base
import validation
from model import CleanResidualPMF
from models.renderer_objective import renderer_loss, sample_from_batch
from training_utils import backward_renderer_losses, clip_or_measure_gradients

ARCH='clean_adaptive_maskfree_v5_lens_coverage'

PHASE_PMF='pmf'
PHASE_GATE='gate_warmup'
PHASE_JOINT='joint'

RUNTIME_ONLY_KEYS={
    'output','resume','init_weights','device','data_root','grid_cache','num_workers',
    'log_every','save_every','vis_every','vis_split','vis_names','vis_count',
}


def component_config(args):
    semantic={key:value for key,value in vars(args).items() if key not in RUNTIME_ONLY_KEYS}
    return {'implementation_revision':ARCH,**semantic}


def build_ema(model,args,device):
    ema=CleanResidualPMF(args,enable_training_v=False).to(device)
    ema.load_state_dict(base._inference_subset(model.state_dict()),strict=True)
    ema.eval().requires_grad_(False)
    return ema

@torch.no_grad()
def sync_ema(ema,model):
    ema.load_state_dict(base._inference_subset(model.state_dict()),strict=True)

@torch.no_grad()
def update_ema(ema,model,decay):
    source=base._inference_subset(model.state_dict()); target=ema.state_dict()
    for k,v in target.items():
        s=source[k].to(device=v.device,dtype=v.dtype)
        if torch.is_floating_point(v): v.mul_(decay).add_(s,alpha=1-decay)
        else: v.copy_(s)


def phase_for_step(step,args):
    if int(step)<int(args.pmf_only_steps): return PHASE_PMF
    if int(step)<int(args.pmf_only_steps)+int(args.gate_warmup_steps): return PHASE_GATE
    return PHASE_JOINT


def phase_start(phase,args):
    if phase==PHASE_PMF: return 0
    if phase==PHASE_GATE: return int(args.pmf_only_steps)
    return int(args.pmf_only_steps)+int(args.gate_warmup_steps)


def joint_renderer_weight(step,args):
    start=phase_start(PHASE_JOINT,args)
    ramp=max(1,int(args.renderer_ramp_steps))
    progress=min(1.0,max(0.0,float(int(step)-start+1)/float(ramp)))
    return float(args.lambda_renderer)*progress


def model_parameter_groups(model,args):
    groups=[
        {'name':'backbone','params':list(model.backbone.parameters()),'lr':float(args.lr)},
        {'name':'conditioner','params':list(model.conditioner.parameters()),'lr':float(args.lr)},
        {'name':'renderer','params':list(model.learned_frequency_gate.parameters()),'lr':0.0},
    ]
    covered={id(p) for group in groups for p in group['params']}
    missing=[name for name,p in model.named_parameters() if id(p) not in covered]
    if missing: raise RuntimeError(f'unassigned trainable parameter groups: {missing}')
    return groups


def configure_phase(model,opt,phase,args):
    if phase not in (PHASE_PMF,PHASE_GATE,PHASE_JOINT):
        raise ValueError(f'unknown training phase: {phase}')
    train_backbone=phase in (PHASE_PMF,PHASE_JOINT)
    train_conditioner=phase in (PHASE_PMF,PHASE_JOINT)
    train_renderer=phase in (PHASE_GATE,PHASE_JOINT)
    model.backbone.requires_grad_(train_backbone)
    model.conditioner.requires_grad_(train_conditioner)
    model.learned_frequency_gate.requires_grad_(train_renderer)
    if phase==PHASE_GATE:
        model.eval(); model.learned_frequency_gate.train()
    else:
        model.train(); model.learned_frequency_gate.train(train_renderer)
    lrs={
        'backbone':float(args.lr if phase==PHASE_PMF else args.joint_backbone_lr if phase==PHASE_JOINT else 0.0),
        'conditioner':float(args.lr if phase==PHASE_PMF else args.joint_conditioner_lr if phase==PHASE_JOINT else 0.0),
        'renderer':float(args.renderer_lr if train_renderer else 0.0),
    }
    for group in opt.param_groups: group['lr']=lrs[group['name']]
    return lrs


def make_loader(dataset,args,seed,*,pin_memory):
    worker_generator=torch.Generator().manual_seed(int(seed)+1_000_003)
    kwargs=dict(
        num_workers=args.num_workers,pin_memory=bool(pin_memory),
        persistent_workers=args.num_workers>0,collate_fn=base.collate_batch,
        generator=worker_generator,
    )
    if args.sampling=='balanced':
        sampler=base.BalancedDomainBatchSampler(
            dataset,args.batch_size,args.real_batch_fraction,
            args.synthetic_prefix,args.real_prefix,seed=seed,
        )
        loader=DataLoader(dataset,batch_sampler=sampler,**kwargs)
        detail=(f'{sampler.synthetic_per_batch}syn+{sampler.real_per_batch}real')
    else:
        sampler=base.ResumableRandomBatchSampler(
            dataset,args.batch_size,seed=seed,drop_last=True
        )
        loader=DataLoader(dataset,batch_sampler=sampler,**kwargs)
        detail='natural'
    return loader,detail,sampler,worker_generator


def initial_data_progress():
    return {
        'pmf':{'epoch':0,'batch_cursor':0},
        'native':{'epoch':0,'batch_cursor':0},
    }


def derived_legacy_progress(step,args,samplers):
    """Best-effort data position for old, explicitly non-strict checkpoints."""
    pmf_updates=min(int(step),int(args.pmf_only_steps))*int(args.grad_accum_steps)
    native_updates=max(0,int(step)-int(args.pmf_only_steps))*int(args.grad_accum_steps)
    progress={}
    for key,consumed in (('pmf',pmf_updates),('native',native_updates)):
        epoch,cursor=divmod(consumed,int(samplers[key].num_batches))
        progress[key]={'epoch':int(epoch),'batch_cursor':int(cursor)}
    return progress


def initial_early_stopping_state():
    return {
        'best_score':None,'best_step':0,'bad_validations':0,
        'last_validation_step':0,'stopped':False,
    }


def early_stop_start_step(args):
    if int(args.early_stop_start_step)>=0:
        return int(args.early_stop_start_step)
    return phase_start(PHASE_JOINT,args)+int(args.renderer_ramp_steps)


def update_early_stopping(state,score,step,args):
    score=float(score)
    if math.isnan(score) or score==float('-inf'):
        raise FloatingPointError(
            f'nonfinite validation metric {args.early_stop_metric} at step={step}: {score}'
        )
    previous=state.get('best_score')
    improved=previous is None or score>float(previous)+float(args.early_stop_min_delta)
    state['last_validation_step']=int(step)
    if improved:
        state.update(best_score=score,best_step=int(step),bad_validations=0)
    else:
        state['bad_validations']=int(state.get('bad_validations',0))+1
    should_stop=(
        int(args.early_stop_patience)>0
        and int(state['bad_validations'])>=int(args.early_stop_patience)
    )
    state['stopped']=bool(should_stop)
    return improved,should_stop


def should_validate(step,args):
    if int(args.val_every)<=0:
        return False
    boundaries={
        int(args.pmf_only_steps),phase_start(PHASE_JOINT,args),int(args.num_steps),
    }
    return int(step)%int(args.val_every)==0 or int(step) in boundaries


def snapshot_data_state(
    progress,samplers,worker_generators,validation_names,early_stopping
):
    return {
        'version':2,
        'validation_names':list(validation_names),
        'early_stopping':dict(early_stopping),
        'loaders':{
            key:{
                **{name:int(value) for name,value in progress[key].items()},
                'kind':type(samplers[key]).__name__,
                'seed':int(samplers[key].seed),
                'num_batches':int(samplers[key].num_batches),
                'worker_generator_state':worker_generators[key].get_state(),
            }
            for key in ('pmf','native')
        },
    }


def restore_data_state(saved,samplers,worker_generators,validation_names):
    if int(saved.get('version',0))!=2 or not isinstance(saved.get('loaders'),dict):
        raise ValueError('unsupported or incomplete strict data_state')
    saved_names=list(saved.get('validation_names',()))
    if saved_names!=list(validation_names):
        raise ValueError(
            'validation holdout mismatch across strict resume: '
            f'checkpoint={saved_names} current={list(validation_names)}'
        )
    progress={}
    for key in ('pmf','native'):
        state=saved['loaders'].get(key)
        if not isinstance(state,dict): raise ValueError(f'data_state missing loader {key!r}')
        expected=(type(samplers[key]).__name__,int(samplers[key].seed),int(samplers[key].num_batches))
        actual=(state.get('kind'),int(state.get('seed',-1)),int(state.get('num_batches',-1)))
        if actual!=expected:
            raise ValueError(f'{key} sampler mismatch: checkpoint={actual} current={expected}')
        epoch=int(state['epoch']); cursor=int(state['batch_cursor'])
        samplers[key].set_position(epoch,cursor)
        worker_generators[key].set_state(state['worker_generator_state'].cpu())
        progress[key]={'epoch':epoch,'batch_cursor':cursor}
    early_stopping=saved.get('early_stopping')
    if not isinstance(early_stopping,dict):
        raise ValueError('data_state missing early_stopping state')
    expected=set(initial_early_stopping_state())
    if set(early_stopping)!=expected:
        raise ValueError('data_state has incomplete early_stopping state')
    return progress,dict(early_stopping)


def next_training_batch(key,loaders,samplers,progress,iterators):
    position=progress[key]; sampler=samplers[key]
    if key not in iterators:
        sampler.set_position(position['epoch'],position['batch_cursor'])
        iterators[key]=iter(loaders[key])
    try:
        batch=next(iterators[key])
    except StopIteration:
        sampler.set_position(position['epoch'],position['batch_cursor'])
        iterators[key]=iter(loaders[key])
        batch=next(iterators[key])
    consumed_position={
        'sampler_epoch':int(position['epoch'])+1,
        'sampler_batch':int(position['batch_cursor'])+1,
    }
    position['batch_cursor']+=1
    if position['batch_cursor']>=sampler.num_batches:
        position['epoch']+=1; position['batch_cursor']=0
    return batch,consumed_position


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def move_training_batch(batch,device,*,stream_native):
    if not stream_native: return base.move_batch(batch,device)
    # Keep ragged native tensors on CPU and stream one sample at a time in
    # renderer_backward. Fixed token tensors still move as the full batch of six.
    native_keys={
        'native_clean','native_glare','native_prior_u8','native_lens_u8',
    }
    output={}
    for key,value in batch.items():
        if key in native_keys:
            output[key]=value
        elif torch.is_tensor(value):
            output[key]=value.to(device,non_blocking=True)
        else:
            output[key]=value
    return output


def renderer_backward(model,x0,batch,args,*,backward_scale,propagate_to_pmf,device):
    """Backprop renderer sample-by-sample and optionally return dL/dx0.

    Joint training uses detached x0 proxy leaves for the native renderer. This frees
    each ragged native-resolution graph immediately, then injects the accumulated
    dL/dx0 into the PMF graph in one backward call. It is exactly the chain rule but
    avoids retaining six full native renderer graphs at once.
    """
    batch_size=int(x0.shape[0]); totals: Dict[str,float]={}; x0_grads=[]
    for index in range(batch_size):
        proxy=x0[index].detach()
        if propagate_to_pmf: proxy=proxy.requires_grad_(True)
        sample=base.move_sample(sample_from_batch(batch,index),device)
        native_clean=batch['native_clean'][index].to(device,non_blocking=True)
        with base.autocast_ctx(device,args.precision):
            pred,_,_,lens,_,_,_,_=model.decode(proxy,sample)
            losses_i=renderer_loss(
                pred,native_clean,lens,model.learned_frequency_gate,args
            )
        proxy_gradient=backward_renderer_losses(
            losses_i,proxy,scale=float(backward_scale)/float(batch_size),
            propagate_to_pmf=propagate_to_pmf,
        )
        if propagate_to_pmf:
            if proxy_gradient is None: raise RuntimeError('renderer produced no data gradient for x0')
            x0_grads.append(proxy_gradient.to(dtype=x0.dtype))
        for key,value in losses_i.metrics.items():
            totals[key]=totals.get(key,0.0)+float(value.float().detach().cpu())/batch_size
    gradient=torch.stack(x0_grads,dim=0) if propagate_to_pmf else None
    return totals,gradient


def save_ckpt(
    path,model,ema,opt,step,epoch,args,progress,samplers,worker_generators,
    validation_names,early_stopping,
):
    checkpointing.save_training_checkpoint(
        path,architecture=ARCH,
        phase=phase_for_step(min(int(step),int(args.num_steps)-1),args),
        model=model,ema=ema,optimizer=opt,step=step,epoch=epoch,args=args,
        component_config=component_config(args),
        data_state=snapshot_data_state(
            progress,samplers,worker_generators,validation_names,early_stopping
        ),
    )


def panel_only(sample,model,out_dir,step,epoch,seed):
    was=model.training; model.eval()
    with torch.inference_mode():
        pred,_,x0,_,lens,_,_,_,_=model.predict(sample,seed=seed)
    model.train(was)
    glare=sample['native_glare'].float(); clean=sample['native_clean'].float(); box=sample['crop_box']
    x0_native=base.render_continuous_field_batch(
        x0.unsqueeze(0),[sample['leaves_cpu']],[int(sample['native_side'])],model.canvas_size,model.halo_ratio
    )[0].clamp(-1,1)
    glare_o,pred_o,clean_o,x0_o=[base.crop_original(x,box) for x in (glare,pred.clamp(-1,1),clean,x0_native)]
    grid=base.grid_overlay(glare,sample['leaves'],model.canvas_size)
    grid_t=base.crop_original(base.pil_to_tensor(grid),box)
    prior=base.crop_original(sample['native_prior_u8'].float()/127.5-1.0,box).expand(3,-1,-1)
    lens_o=base.crop_original(lens,box).expand(3,-1,-1)*2-1
    tiles=[
        base.labeled_tile(base.tensor_to_pil(glare_o),'Glare'),
        base.labeled_tile(base.tensor_to_pil(pred_o),'Prediction'),
        base.labeled_tile(base.tensor_to_pil(clean_o),'Clean'),
        base.labeled_tile(base.tensor_to_pil(x0_o),'Raw PoU x0'),
        base.labeled_tile(base.tensor_to_pil(prior),'Eye geometry'),
        base.labeled_tile(base.tensor_to_pil(grid_t),'Adaptive grid'),
    ]
    panel=Image.new('RGB',(3*320,2*320),'white')
    for i,tile in enumerate(tiles): panel.paste(tile,((i%3)*320,(i//3)*320))
    root=Path(out_dir)/'visualizations'/str(sample['name']); root.mkdir(parents=True,exist_ok=True)
    path=root/f'epoch_{int(epoch):04d}_step_{int(step):08d}.png'; panel.save(path)
    print(f'[vis] epoch={epoch} step={step} sample={sample["name"]} -> {path}',flush=True)


def validate_ema(ema,dataset,device,out,step,args,early_stopping):
    summary=validation.evaluate_model(
        ema,dataset,device,seed=args.seed+90_000,
        mixed_mask_weight=args.val_mixed_mask_weight,
        ssim_win_size=args.val_ssim_win_size,
    )
    completed_phase=phase_for_step(max(0,int(step)-1),args)
    score=float(summary['macro'][args.early_stop_metric])
    eligible=(
        completed_phase==PHASE_JOINT and int(step)>=early_stop_start_step(args)
    )
    improved=False; should_stop=False
    if eligible:
        improved,should_stop=update_early_stopping(
            early_stopping,score,step,args
        )

    flat=validation.flatten_summary(summary)
    fields=[
        'step','phase','early_stop_eligible','early_stop_metric','score',
        'improved','bad_validations','best_score','best_step',*flat.keys(),
    ]
    log_path=Path(out)/'validation.csv'
    if not log_path.exists():
        with log_path.open('w',newline='',encoding='utf-8') as file:
            csv.DictWriter(file,fieldnames=fields).writeheader()
    row={
        'step':int(step),'phase':completed_phase,
        'early_stop_eligible':int(eligible),'early_stop_metric':args.early_stop_metric,
        'score':score,'improved':int(improved),
        'bad_validations':int(early_stopping['bad_validations']),
        'best_score':early_stopping['best_score'],
        'best_step':int(early_stopping['best_step']),**flat,
    }
    with log_path.open('a',newline='',encoding='utf-8') as file:
        csv.DictWriter(file,fieldnames=fields).writerow(row)
    payload={
        'step':int(step),'phase':completed_phase,
        'early_stop_eligible':eligible,'early_stop_metric':args.early_stop_metric,
        'score':score,'summary':summary,'early_stopping':dict(early_stopping),
    }
    latest=Path(out)/'validation_latest.json'
    temporary=latest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(latest)
    syn=summary['synthetic']; real=summary['real']; macro=summary['macro']
    print(
        f'[validation] step={step} phase={completed_phase} '
        f'macro {args.early_stop_metric}={score:.4f} '
        f'Mixed-PSNR syn/real={syn["PSNR_mixed"]:.4f}/{real["PSNR_mixed"]:.4f} '
        f'Mixed-SSIM syn/real={syn["SSIM_mixed"]:.4f}/{real["SSIM_mixed"]:.4f} '
        f'eligible={eligible} improved={improved} '
        f'bad={early_stopping["bad_validations"]}/{args.early_stop_patience}',flush=True,
    )
    return improved,should_stop


def parse_args():
    p=argparse.ArgumentParser(description='Clean mask-free adaptive PMF trainer',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_common(p)
    for action in p._actions:
        if action.dest == 'output':
            action.required = False
    p.add_argument('--output_dir',dest='output',help='alias of --output')
    init=p.add_mutually_exclusive_group()
    init.add_argument('--resume',default='')
    init.add_argument('--init_weights',default='',help='official pMF or compatible local checkpoint; format is detected automatically')
    p.add_argument('--num_steps',type=int,default=60000)
    p.add_argument('--batch_size',type=int,default=6)
    p.add_argument('--grad_accum_steps',type=int,default=1)
    p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026,
                   help='global experiment seed; strict resume restores the saved RNG states')
    p.add_argument('--sampling',choices=('balanced','natural'),default='balanced',
                   help='balanced uses a fixed synthetic/real ratio; natural follows dataset frequency')
    p.add_argument('--real_batch_fraction',type=float,default=.5)
    p.add_argument('--synthetic_prefix',default='syn_')
    p.add_argument('--real_prefix',default='pub_')
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--joint_backbone_lr',type=float,default=2e-6,
                   help='backbone learning rate during joint fine-tuning')
    p.add_argument('--joint_conditioner_lr',type=float,default=1e-5,
                   help='AdaptiveConditioner learning rate during joint fine-tuning')
    p.add_argument('--renderer_lr',type=float,default=1e-4,
                   help='learned renderer gate learning rate in warm-up and joint phases')
    p.add_argument('--weight_decay',type=float,default=1e-4)
    p.add_argument('--grad_clip',type=float,default=5.0)
    p.add_argument('--ema_decay',type=float,default=0.999)
    p.add_argument('--log_every',type=int,default=20)
    p.add_argument('--save_every',type=int,default=1000)
    p.add_argument('--vis_every',type=int,default=500,
                   help='optimizer steps between training visualizations; 0 disables')
    p.add_argument('--vis_split',default='train')
    p.add_argument('--vis_names',nargs='+',default=[])
    p.add_argument('--vis_count',type=int,default=2,
                   help='number of deterministic samples used when --vis_names is omitted')
    monitor=p.add_argument_group('validation and early stopping')
    monitor.add_argument('--val_every',type=int,default=1000,
                         help='optimizer steps between fixed-holdout validation; 0 disables')
    monitor.add_argument('--val_synthetic_count',type=int,default=16)
    monitor.add_argument('--val_real_count',type=int,default=16)
    monitor.add_argument('--val_mixed_mask_weight',type=float,default=.5)
    monitor.add_argument('--val_ssim_win_size',type=int,default=7)
    monitor.add_argument('--early_stop_metric',choices=validation.FAST_METRICS,
                         default='PSNR_mixed')
    monitor.add_argument('--early_stop_patience',type=int,default=5,
                         help='joint-phase validation checks without improvement; 0 disables')
    monitor.add_argument('--early_stop_min_delta',type=float,default=.02,
                         help='minimum absolute metric increase counted as improvement')
    monitor.add_argument('--early_stop_start_step',type=int,default=-1,
                         help='-1 starts after the joint renderer ramp')
    p.add_argument('--conditioner_bottleneck',type=int,default=256)
    p.add_argument('--condition_gate_init',type=float,default=.1)
    p.add_argument('--context_gate_init',type=float,default=.05)
    schedule=p.add_argument_group('three-stage optimization schedule')
    schedule.add_argument('--pmf_only_steps',type=int,default=45000,
                          help='initial optimizer steps with PMF/conditioner trainable and renderer frozen')
    schedule.add_argument('--gate_warmup_steps',type=int,default=5000,
                          help='following optimizer steps with PMF frozen and only renderer gate trainable')
    schedule.add_argument('--lambda_renderer',type=float,default=1.0,
                          help='full renderer-objective weight; joint phase ramps from zero to this value')
    schedule.add_argument('--renderer_ramp_steps',type=int,default=2000,
                          help='joint-phase steps used to linearly ramp renderer loss to full weight')
    schedule.add_argument('--charbonnier_eps',type=float,default=1e-3)
    schedule.add_argument('--renderer_lowpass_kernel',type=int,default=15)
    schedule.add_argument('--lambda_renderer_low',type=float,default=.5)
    schedule.add_argument('--lambda_renderer_gradient',type=float,default=.1)
    schedule.add_argument('--lambda_gate_prior',type=float,default=.01)
    schedule.add_argument('--lambda_gate_tv',type=float,default=.01)
    switches=p.add_argument_group('independent component switches')
    switches.add_argument('--disable_geometry_token',action='store_true')
    switches.add_argument('--disable_semantic_conditioning',action='store_true')
    switches.add_argument('--disable_coarse_to_fine_context',action='store_true')
    switches.add_argument('--disable_geometry_guided_source',action='store_true')
    switches.add_argument('--uniform_grid',action='store_true')
    args=p.parse_args()
    if not args.output: p.error('one of --output or --output_dir is required')
    try:
        config.validate_training_args(args)
    except (ValueError,FileNotFoundError) as error:
        p.error(str(error))
    if args.val_every>0 and validation.structural_similarity is None:
        p.error(
            'validation requires scikit-image: '
            f'{validation.STRUCTURAL_SIMILARITY_ERROR}'
        )
    return args


def main():
    args=parse_args(); device=base.resolve_device(args,train_mode=True); base.validate_precision_for_device(args,device); base.configure_cuda_speed(device)
    out=Path(args.output)
    if not args.resume and any((out/name).exists() for name in ('train.csv','checkpoint_latest.pt')):
        raise FileExistsError(
            f'{out} already contains training state; use --resume with its checkpoint or choose a new --output'
        )
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    cfg=config.make_grid_config(args); uniform_grid=args.uniform_grid
    validation_split=(
        validation.select_holdout_names(
            args.data_root,seed=args.seed,
            synthetic_count=args.val_synthetic_count,real_count=args.val_real_count,
            synthetic_prefix=args.synthetic_prefix,real_prefix=args.real_prefix,
        )
        if args.val_every>0 else {'synthetic':[],'real':[]}
    )
    validation_names=[
        *validation_split['synthetic'],*validation_split['real'],
    ]
    pmf_dataset=base.ReglassDataset(
        args.data_root,'train',args.grid_cache,cfg,args.canonical_size,
        augment=True,identity_prob=0.0,exclude_names=validation_names,
        need_native=False,uniform_grid=uniform_grid,
    )
    native_dataset=base.ReglassDataset(
        args.data_root,'train',args.grid_cache,cfg,args.canonical_size,
        augment=False,identity_prob=0.0,exclude_names=validation_names,
        need_native=True,uniform_grid=uniform_grid,
    )
    validation_dataset=(
        base.ReglassDataset(
            args.data_root,'train',args.grid_cache,cfg,args.canonical_size,
            augment=False,identity_prob=0.0,sample_names=validation_names,
            need_native=True,need_mask=True,uniform_grid=uniform_grid,
        )
        if validation_names else None
    )
    pmf_loader,pmf_sampling,pmf_sampler,pmf_worker_generator=make_loader(
        pmf_dataset,args,args.seed,pin_memory=device.type=='cuda'
    )
    native_loader,native_sampling,native_sampler,native_worker_generator=make_loader(
        native_dataset,args,args.seed+10000,pin_memory=False
    )
    loaders={'pmf':pmf_loader,'native':native_loader}
    samplers={'pmf':pmf_sampler,'native':native_sampler}
    worker_generators={'pmf':pmf_worker_generator,'native':native_worker_generator}
    progress=initial_data_progress()
    updates_per_epoch={
        key:max(1,math.ceil(samplers[key].num_batches/args.grad_accum_steps))
        for key in loaders
    }
    print(
        f'[domain-sampling] mode={args.sampling} pmf={pmf_sampling} native={native_sampling} '
        f'batch={args.batch_size} pmf_batches={len(pmf_loader)} native_batches={len(native_loader)}',
        flush=True,
    )
    print(
        f'[validation] fixed_train_holdout={len(validation_dataset) if validation_dataset else 0} '
        f'({len(validation_split["synthetic"])} synthetic + '
        f'{len(validation_split["real"])} real) every={args.val_every} '
        f'metric={args.early_stop_metric} patience={args.early_stop_patience} '
        f'start_step={early_stop_start_step(args)}',flush=True,
    )
    model=CleanResidualPMF(args,enable_training_v=True).to(device)
    print(
        '[component-config] '
        f'coarse_to_fine={model.use_coarse_to_fine_context} '
        f'low_frequency_anchor={model.lambda_x0>0} lambda_x0={model.lambda_x0} '
        f'lambda_eye_gradient={model.lambda_eye_gradient} lambda_preserve={model.lambda_preserve} '
        f'geometry_source={model.use_geometry_guided_source} '
        f'adaptive_grid={model.use_adaptive_grid} semantic_conditioning={model.use_semantic_conditioning} '
        f'learned_renderer={model.use_learned_frequency_renderer} sampling={args.sampling} '
        f'schedule={args.pmf_only_steps}pmf+{args.gate_warmup_steps}gate+'
        f'{args.num_steps-args.pmf_only_steps-args.gate_warmup_steps}joint',
        flush=True,
    )
    ema=build_ema(model,args,device)
    opt=torch.optim.AdamW(
        model_parameter_groups(model,args),betas=(.9,.95),weight_decay=args.weight_decay
    )
    step=0
    early_stopping=initial_early_stopping_state()
    if args.resume:
        step,saved_data_state,strict=checkpointing.load_training_checkpoint(
            args.resume,model=model,ema=ema,optimizer=opt,device=device,
            component_config=component_config(args),
        )
        if strict and saved_data_state is not None:
            progress,early_stopping=restore_data_state(
                saved_data_state,samplers,worker_generators,validation_names
            )
        else:
            progress=derived_legacy_progress(step,args,samplers)
        print(
            f'[resume] step={step} strict_trajectory={strict} data_progress={progress} '
            f'<- {args.resume}',flush=True
        )
    elif args.init_weights:
        weight_format=base.load_initial_weights(args.init_weights,model); sync_ema(ema,model)
        print(f'[init_weights] format={weight_format} optimizer=fresh start_step=0 EMA=synchronized',flush=True)
    else:
        sync_ema(ema,model); print('[init] local initialization',flush=True)
    active_phase=phase_for_step(step,args)
    active_lrs=configure_phase(model,opt,active_phase,args)
    print(f'[phase] step={step} phase={active_phase} lrs={active_lrs}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    (out/'validation_split.json').write_text(
        json.dumps(
            {'source_split':'train','seed':args.seed,**validation_split},
            indent=2,ensure_ascii=False,
        ),encoding='utf-8'
    )
    vis=[]
    if args.vis_every>0:
        if args.vis_split=='train' and not args.vis_names:
            vds=native_dataset
        else:
            vds=base.ReglassDataset(
                args.data_root,args.vis_split,args.grid_cache,cfg,args.canonical_size,
                augment=False,sample_names=args.vis_names,need_native=True,uniform_grid=uniform_grid,
            )
        vis_indices=(
            [vds.index_for_name(n) for n in args.vis_names]
            if args.vis_names else list(range(min(args.vis_count,len(vds))))
        )
        vis=[vds[index] for index in vis_indices]
        print(f'[visualization] every={args.vis_every} samples={[str(v["name"]) for v in vis]}',flush=True)
    log=out/'train.csv'
    if not log.exists():
        with log.open('w',newline='') as f:
            csv.writer(f).writerow([
                'phase','phase_step','epoch','batch_in_epoch','step','loss','pmf','renderer','renderer_weight',
                'u','v','x0','eye_grad','preserve','renderer_reconstruction','renderer_low',
                'renderer_gradient','gate_prior','gate_tv','raw_u','raw_v','gate_c','gate_gl',
                'support','preserve_outside','grad','lr_backbone','lr_conditioner','lr_renderer',
            ])
    iterators={}
    if early_stopping['stopped']:
        print(
            f'[early-stop] checkpoint already stopped at step={step}; '
            'training loop will not resume',flush=True,
        )
    while step<args.num_steps and not early_stopping['stopped']:
        phase=phase_for_step(step,args)
        if phase!=active_phase:
            active_phase=phase; active_lrs=configure_phase(model,opt,phase,args)
            print(f'[phase] step={step} phase={phase} lrs={active_lrs}',flush=True)
        loader_key='pmf' if phase==PHASE_PMF else 'native'
        opt.zero_grad(set_to_none=True); acc={}
        for micro_step in range(args.grad_accum_steps):
            batch,data_position=next_training_batch(
                loader_key,loaders,samplers,progress,iterators
            )
            batch=move_training_batch(batch,device,stream_native=loader_key=='native')
            renderer_metrics={}; renderer_weight=0.0
            if phase==PHASE_PMF:
                with base.autocast_ctx(device,args.precision):
                    pmf_loss,met=model.forward_loss_batch(batch)
                scaled=pmf_loss/args.grad_accum_steps
                if not torch.isfinite(scaled): raise FloatingPointError(f'nonfinite PMF loss at step={step}')
                scaled.backward(); total_value=float(pmf_loss.detach())
            elif phase==PHASE_GATE:
                with torch.no_grad(),base.autocast_ctx(device,args.precision):
                    x0=model.one_step_x0_batch(
                        batch['glare_patches'],batch['leaves'],batch['region_probs'],
                        seed=args.seed+step*args.grad_accum_steps+micro_step,
                    )
                renderer_weight=float(args.lambda_renderer)
                renderer_metrics,_=renderer_backward(
                    model,x0,batch,args,
                    backward_scale=renderer_weight/args.grad_accum_steps,
                    propagate_to_pmf=False,device=device,
                )
                pmf_loss=x0.new_zeros(()); met={}
                total_value=renderer_weight*renderer_metrics['renderer']
            else:
                with base.autocast_ctx(device,args.precision):
                    pmf_loss,met,x0=model.forward_loss_batch(batch,return_x0=True)
                if not torch.isfinite(pmf_loss):
                    raise FloatingPointError(f'nonfinite PMF loss at joint step={step}')
                renderer_weight=joint_renderer_weight(step,args)
                renderer_metrics,x0_gradient=renderer_backward(
                    model,x0,batch,args,
                    backward_scale=renderer_weight/args.grad_accum_steps,
                    propagate_to_pmf=True,device=device,
                )
                assert x0_gradient is not None
                if not torch.isfinite(x0_gradient).all():
                    raise FloatingPointError(f'nonfinite renderer dL/dx0 at joint step={step}')
                torch.autograd.backward(
                    [pmf_loss/args.grad_accum_steps,x0],
                    grad_tensors=[None,x0_gradient],
                )
                total_value=float(pmf_loss.detach())+renderer_weight*renderer_metrics['renderer']
            conditioner_metrics=model.conditioner.gate_metrics()
            values={
                **{k:float(v.float().detach().cpu()) for k,v in met.items()},
                **renderer_metrics,
                'loss':total_value,'pmf':float(pmf_loss.detach()),
                'renderer':renderer_metrics.get('renderer',0.0),'renderer_weight':renderer_weight,
                'condition_gate_mean':float(conditioner_metrics['condition_gate_mean'].detach()),
                'context_gate_mean':float(conditioner_metrics['context_gate_mean'].detach()),
            }
            for key,value in values.items():
                acc[key]=acc.get(key,0.0)+float(value)/args.grad_accum_steps
        parameters=trainable_parameters(model)
        grad=clip_or_measure_gradients(parameters,args.grad_clip)
        opt.step(); update_ema(ema,model,args.ema_decay); step+=1
        completed=step-phase_start(phase,args)
        epoch=(completed-1)//updates_per_epoch[loader_key]+1
        phase_step=completed
        def metric(name): return float(acc.get(name,0.0))
        with log.open('a',newline='') as f:
            csv.writer(f).writerow([
                phase,phase_step,data_position['sampler_epoch'],
                data_position['sampler_batch'],step,metric('loss'),metric('pmf'),
                metric('renderer'),metric('renderer_weight'),metric('u'),metric('v'),metric('x0'),
                metric('eye_gradient'),metric('preserve'),metric('renderer_reconstruction'),
                metric('renderer_low'),metric('renderer_gradient'),metric('gate_prior'),metric('gate_tv'),
                metric('raw_u'),metric('raw_v'),metric('condition_gate_mean'),metric('context_gate_mean'),
                metric('geometry_support_mean'),metric('preserve_outside_mean'),float(grad),
                active_lrs['backbone'],active_lrs['conditioner'],active_lrs['renderer'],
            ])
        if step==1 or step%args.log_every==0:
            print(
                f"phase={phase} phase_step={phase_step} phase_epoch={epoch} "
                f"sampler={data_position['sampler_epoch']}:"
                f"{data_position['sampler_batch']}/{samplers[loader_key].num_batches} "
                f"step={step}/{args.num_steps} "
                f"loss={metric('loss'):.5f} pmf={metric('pmf'):.5f} "
                f"renderer={metric('renderer'):.5f}@{metric('renderer_weight'):.3f} "
                f"u/v={metric('u'):.5f}/{metric('v'):.5f} x0={metric('x0'):.5f} "
                f"gate_reg={metric('gate_prior'):.5f}/{metric('gate_tv'):.5f} "
                f"grad={float(grad):.3f} lrs={active_lrs}",flush=True,
            )
        stop_now=False
        if should_validate(step,args):
            assert validation_dataset is not None
            improved,stop_now=validate_ema(
                ema,validation_dataset,device,out,step,args,early_stopping
            )
            if improved:
                save_ckpt(
                    out/'checkpoint_best.pt',model,ema,opt,step,epoch,args,
                    progress,samplers,worker_generators,
                    validation_names,early_stopping,
                )
                print(
                    f'[best] {args.early_stop_metric}='
                    f'{early_stopping["best_score"]:.4f} step={step} '
                    f'-> {out/"checkpoint_best.pt"}',flush=True,
                )
        if (args.save_every>0 and step%args.save_every==0) or stop_now:
            save_ckpt(
                out/'checkpoint_latest.pt',model,ema,opt,step,epoch,args,
                progress,samplers,worker_generators,
                validation_names,early_stopping,
            )
        if stop_now:
            print(
                f'[early-stop] step={step} best_step={early_stopping["best_step"]} '
                f'best_{args.early_stop_metric}={early_stopping["best_score"]:.4f} '
                f'patience={args.early_stop_patience}',flush=True,
            )
            break
        if vis and step%args.vis_every==0:
            for sample in vis: panel_only(base.move_sample(sample,device),ema,out,step,epoch,args.vis_seed)
    final_phase=phase_for_step(max(0,step-1),args)
    final_loader_key='pmf' if final_phase==PHASE_PMF else 'native'
    final_completed=max(1,step-phase_start(final_phase,args))
    final_epoch=(final_completed-1)//updates_per_epoch[final_loader_key]+1
    save_ckpt(
        out/'checkpoint_latest.pt',model,ema,opt,step,final_epoch,args,
        progress,samplers,worker_generators,
        validation_names,early_stopping,
    )
    status='early-stopped' if early_stopping['stopped'] else 'completed'
    print(f'[PASS] {status} epoch={final_epoch} step={step}',flush=True)

if __name__=='__main__': main()
