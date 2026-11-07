from __future__ import annotations

"""Current mask-free adaptive PMF and its independent component switches.

Training and inference never consume a reflection mask.  Eye geometry / adaptive
leaf scale are the only spatial conditions.  Clean images are paired training
targets only.
"""

from typing import Dict, Mapping, Optional
import torch

import core as base
from models.adaptive_conditioning import AdaptiveConditioner
from models.learned_frequency_renderer import LearnedFrequencyGate


def load_inference_state(model:torch.nn.Module,state:Mapping[str,torch.Tensor]):
    """Load only checkpoints created by this cleaned implementation."""
    return model.load_state_dict(state,strict=True)


class CleanResidualPMF(base.MaskFreeResidualPMF):
    def __init__(self,args,*,enable_training_v:bool=True):
        self.use_geometry_token=not bool(getattr(args,'disable_geometry_token',False))
        self.use_semantic_conditioning=not bool(getattr(args,'disable_semantic_conditioning',False))
        self.use_coarse_to_fine_context=not bool(getattr(args,'disable_coarse_to_fine_context',False))
        self.use_geometry_guided_source=not bool(getattr(args,'disable_geometry_guided_source',False))
        self.use_adaptive_grid=not bool(getattr(args,'uniform_grid',False))
        self.use_frequency_renderer=True
        self.use_learned_frequency_renderer=(
            bool(getattr(args,'learned_frequency_renderer',True))
        )
        super().__init__(args,enable_training_v=enable_training_v)
        self.lambda_eye_gradient=(
            0.0 if not self.use_semantic_conditioning
            else float(getattr(args,'lambda_eye_gradient',0.005))
        )
        self.eye_gradient_periocular_weight=float(
            getattr(args,'eye_gradient_periocular_weight',0.05)
        )
        self.eye_gradient_eye_weight=float(getattr(args,'eye_gradient_eye_weight',1.0))
        self.eye_gradient_iris_weight=float(getattr(args,'eye_gradient_iris_weight',2.0))
        self.lambda_preserve=float(getattr(args,'lambda_preserve',0.01))
        self.conditioner=AdaptiveConditioner(
            canonical_size=args.canonical_size,
            hidden_size=self.backbone.hidden_size,
            num_heads=self.backbone.num_heads,
            bottleneck=int(getattr(args,'conditioner_bottleneck',128)),
            canvas_size=args.canvas_size,
            base_patch=args.base_patch_size,
            condition_gate_init=float(getattr(args,'condition_gate_init',0.1)),
            context_gate_init=float(getattr(args,'context_gate_init',0.05)),
            use_geometry_token=self.use_geometry_token,
            use_coarse_to_fine_context=self.use_coarse_to_fine_context,
        )
        self.learned_frequency_gate=LearnedFrequencyGate(
            width=int(getattr(args,'renderer_gate_width',32)),
            map_size=int(getattr(args,'renderer_gate_map_size',256)),
            residual_scale=float(getattr(args,'renderer_gate_residual_scale',1.0)),
        )

    def _backbone_region_probs(self,region_probs):
        return region_probs if self.use_semantic_conditioning else None

    def _semantic_region_probs(self,region_probs):
        if self.use_semantic_conditioning:
            return region_probs
        neutral=torch.zeros_like(region_probs)
        neutral[...,0]=1
        return neutral

    def _condition_tokens(self,state,glare,leaves,region_probs):
        b,t=state.shape[:2]
        region_probs=self._semantic_region_probs(region_probs)
        raw=self.backbone._canonical_patch_embed(state.reshape(b*t,*state.shape[2:])).reshape(b,t,self.backbone.hidden_size)
        return self.conditioner.forward_from_state_tokens(raw,glare,region_probs,leaves).conditioned_state

    def _network_batch(self,state,t,r,leaves,region_probs,*,need_v:bool,
                       glare_patches:Optional[torch.Tensor]=None):
        if glare_patches is None:
            raise ValueError('adaptive PMF requires glare_patches for conditioning')
        batch,tokens=state.shape[:2]
        conditioned=self._condition_tokens(state,glare_patches,leaves,region_probs)
        flat_state=state.reshape(batch*tokens,*state.shape[2:])
        flat_leaves=leaves.reshape(batch*tokens,3)
        model_region=self._semantic_region_probs(region_probs)
        flat_region=model_region.reshape(batch*tokens,model_region.shape[-1])
        flat_cond=conditioned.reshape(batch*tokens,self.backbone.hidden_size)
        offsets=self._offsets(batch,tokens,state.device)
        labels=torch.zeros(batch,device=state.device,dtype=torch.long)
        ux0,vx0=self.backbone.forward_adaptive(
            flat_state,offsets,flat_leaves,t.reshape(batch),(t-r).reshape(batch),labels,
            self.canvas_size,need_v=need_v,padded_image_tokens=tokens,
            region_probs=self._backbone_region_probs(flat_region),image_tokens_override=flat_cond,
        )
        ux0=ux0.reshape(batch,tokens,*state.shape[2:])
        u=(state-ux0)/t.clamp_min(self.t_eps)
        if not need_v: return u,None
        if vx0 is None: raise RuntimeError('V head requested but missing')
        vx0=vx0.reshape(batch,tokens,*state.shape[2:])
        return u,(state-vx0)/t.clamp_min(self.t_eps)

    def _source_support(self,leaves,region_probs):
        if not self.use_geometry_guided_source:
            return torch.ones_like(region_probs[...,0])
        model_region=self._semantic_region_probs(region_probs)
        return self.conditioner.geometry_support(leaves,model_region)

    def _geometry_source(self,glare,leaves,region_probs,generator=None):
        support=self._source_support(leaves,region_probs)[...,None,None,None].to(glare.dtype)
        noise=torch.randn(glare.shape,device=glare.device,dtype=glare.dtype,generator=generator)
        return (glare+support*noise*self.noise_scale).clamp(-1,1)

    def _preservation_support(self,leaves,region_probs):
        """Inference-time geometry support, independent of the source-noise switch."""
        semantic=self._semantic_region_probs(region_probs)
        return self.conditioner.geometry_support(leaves,semantic)

    def _meanflow_loss_batch(self,clean,glare,leaves,region_probs):
        if not self.enable_training_v: raise RuntimeError('training requires V head')
        batch=clean.shape[0]
        t,r=self._sample_time(batch,clean.device,clean.dtype)
        source=self._geometry_source(glare,leaves,region_probs)
        state=(1-t)*clean+t*source
        v_target=((state-clean)/t.clamp_min(self.t_eps)).detach()
        def fn(state_arg,t_arg,r_arg):
            u,v=self._network_batch(state_arg,t_arg,r_arg,leaves,region_probs,need_v=True,
                                    glare_patches=glare)
            if v is None: raise RuntimeError('V missing')
            return u,v
        original=self.backbone.gradient_checkpointing
        self.backbone.set_gradient_checkpointing(False)
        try:
            u,du_dt,v=torch.func.jvp(fn,(state,t,r),(v_target,torch.ones_like(t),torch.zeros_like(r)),has_aux=True)
        finally:
            self.backbone.set_gradient_checkpointing(original)
        capital_v=u+(t-r)*du_dt.detach()
        eu=capital_v-v_target; ev=v-v_target
        token_u=eu.float().square().flatten(2).sum(2)*self.canonical_pixel_loss_scale
        token_v=ev.float().square().flatten(2).sum(2)*self.canonical_pixel_loss_scale
        per_u=token_u.sum(1); per_v=token_v.sum(1)
        au=self._adaptive_normalize(per_u); av=self._adaptive_normalize(per_v)
        lu=au.mean(); lv=av.mean()
        return lu,lv,{
            'u':lu.detach(),'v':lv.detach(),'adaptive_u':au.mean().detach(),'adaptive_v':av.mean().detach(),
            'sample_u_raw':per_u.mean().detach(),'sample_v_raw':per_v.mean().detach(),
            'raw_u':eu.float().square().mean().detach(),'raw_v':ev.float().square().mean().detach(),
            'meanflow_t':t.mean().detach(),'meanflow_interval':(t-r).mean().detach(),
        }

    def one_step_x0_batch(self,glare,leaves,region_probs,*,seed:Optional[int]=None):
        batch,tokens=glare.shape[:2]
        generator=None
        if seed is not None:
            generator=torch.Generator(device=glare.device); generator.manual_seed(int(seed))
        source=self._geometry_source(glare,leaves,region_probs,generator)
        cond=self._condition_tokens(source,glare,leaves,region_probs)
        fs=source.reshape(batch*tokens,*source.shape[2:]); fl=leaves.reshape(batch*tokens,3)
        fr=region_probs.reshape(batch*tokens,region_probs.shape[-1]); fc=cond.reshape(batch*tokens,self.backbone.hidden_size)
        offsets=self._offsets(batch,tokens,glare.device); labels=torch.zeros(batch,device=glare.device,dtype=torch.long)
        ones=torch.ones(batch,device=glare.device,dtype=glare.dtype)
        x0,_=self.backbone.forward_adaptive(fs,offsets,fl,ones,ones,labels,self.canvas_size,need_v=False,
            padded_image_tokens=tokens,region_probs=self._backbone_region_probs(fr),image_tokens_override=fc)
        return x0.reshape(batch,tokens,*glare.shape[2:])

    def one_step_x0(self,glare,leaves,region_probs,*,seed:Optional[int]=None):
        return self.one_step_x0_batch(
            glare.unsqueeze(0),leaves.unsqueeze(0),region_probs.unsqueeze(0),
            seed=seed,
        )[0]

    def _eye_gradient_loss(self,x0,clean,region_probs):
        pred=x0.float(); target=clean.float()
        dx=(pred[...,1:]-pred[...,:-1])-(target[...,1:]-target[...,:-1])
        dy=(pred[...,1:,:]-pred[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])
        token_error=0.5*(dx.abs().flatten(2).mean(2)+dy.abs().flatten(2).mean(2))
        weights=(
            self.eye_gradient_periocular_weight*region_probs[...,1].float()
            +self.eye_gradient_eye_weight*region_probs[...,2].float()
            +self.eye_gradient_iris_weight*region_probs[...,3].float()
        )
        denominator=weights.sum(1)
        per_image=(token_error*weights).sum(1)/denominator.clamp_min(1e-6)
        return torch.where(denominator>0,per_image,torch.zeros_like(per_image)).mean()

    @staticmethod
    def _preserve_loss(x0,glare,support,leaves):
        token_change=(x0.float()-glare.float()).abs().flatten(2).mean(2)
        outside=(1.0-support.float()).clamp(0.0,1.0)
        # Adaptive tokens cover different native areas. A p32 leaf represents 64x
        # the area of a p4 leaf and must therefore contribute 64x as much to an
        # outside-region preservation average.
        area=leaves[...,2].float().square()
        weight=outside*area
        denominator=weight.sum(1)
        per_image=(token_change*weight).sum(1)/denominator.clamp_min(1e-6)
        return torch.where(denominator>0,per_image,torch.zeros_like(per_image)).mean()

    def forward_loss_batch(self,batch_data:Dict[str,object],*,return_x0:bool=False):
        clean=batch_data['clean_patches']; glare=batch_data['glare_patches']; leaves=batch_data['leaves']
        supervision_region=batch_data['region_probs']
        region=self._condition_batch(supervision_region,training_dropout=self.training)
        loss_u,loss_v,metrics=self._meanflow_loss_batch(clean,glare,leaves,region)
        zero=loss_u.detach().new_zeros(())
        need_x0=(return_x0 or self.lambda_x0>0 or self.lambda_eye_gradient>0 or self.lambda_preserve>0)
        x0=None
        if need_x0:
            x0=self.one_step_x0_batch(glare,leaves,region)
        if self.lambda_x0<=0:
            x0_loss=mid=coarse=zero
        else:
            assert x0 is not None
            token_mid=self._low_frequency_token_error(x0,clean,self.x0_mid_size)
            token_coarse=self._low_frequency_token_error(x0,clean,self.x0_coarse_size)
            mid=base.token_vote_loss(token_mid,self.target_tokens)
            coarse=base.token_vote_loss(token_coarse,self.target_tokens)
            x0_loss=mid+self.x0_coarse_weight*coarse
        if x0 is None:
            eye_gradient=preserve=x0_raw=delta_abs=zero
            preserve_support=self._preservation_support(leaves,supervision_region)
        else:
            x0_raw=(x0.float()-clean.float()).square().mean().detach()
            delta_abs=(x0.float()-glare.float()).abs().mean().detach()
            eye_gradient=(
                self._eye_gradient_loss(x0,clean,supervision_region)
                if self.lambda_eye_gradient>0 else zero
            )
            preserve_support=self._preservation_support(leaves,supervision_region)
            preserve=(
                self._preserve_loss(x0,glare,preserve_support,leaves)
                if self.lambda_preserve>0 else zero
            )
        loss=(
            self.lambda_u*loss_u+self.lambda_v*loss_v+self.lambda_x0*x0_loss
            +self.lambda_eye_gradient*eye_gradient
            +self.lambda_preserve*preserve
        )
        metrics.update({'loss':loss.detach(),'x0':x0_loss.detach(),'x0_low_mid':mid.detach(),'x0_low_coarse':coarse.detach(),
                        'eye_gradient':eye_gradient.detach(),'preserve':preserve.detach(),
                        'x0_raw':x0_raw,'delta_abs':delta_abs})
        metrics.update({k:v.detach() for k,v in self.conditioner.gate_metrics().items()})
        source_support=self._source_support(leaves,region)
        metrics['geometry_support_mean']=source_support.mean().detach()
        metrics['preserve_outside_mean']=(1.0-preserve_support).mean().detach()
        if return_x0:
            assert x0 is not None
            return loss,metrics,x0
        return loss,metrics

    @torch.inference_mode()
    def predict(self,sample:Dict[str,object],seed:int=0):
        region=sample['region_probs']
        x0=self.one_step_x0(sample['glare_patches'],sample['leaves'],region,seed=seed)
        pred,delta,alpha,lens_native,reconstruction_need,correction_magnitude,native_hf_weight,highlight_suppression=self.decode(x0,sample)
        return pred,delta,x0,alpha,lens_native,reconstruction_need,correction_magnitude,native_hf_weight,highlight_suppression

    def decode(self,x0:torch.Tensor,sample:Dict[str,object]):
        if self.use_frequency_renderer:
            return super().decode(x0,sample)
        # Explicit renderer bypass: keep adaptive PMF and linear PoU rendering,
        # but bypass all low/high splitting, native-HF injection and semantic
        # frequency gates in render_hybrid_clean_residual().
        delta_native=base.render_continuous_field_batch(
            (x0-sample['glare_patches']).unsqueeze(0),
            [sample['leaves_cpu']],[int(sample['native_side'])],
            self.canvas_size,self.halo_ratio,
        )[0].float()
        native_glare=sample['native_glare'].float()
        lens_native=self._renderer_lens_native(sample,torch.float32)
        rendered=(native_glare+delta_native).clamp(-1,1)
        pred=((1.0-lens_native)*native_glare+lens_native*rendered).clamp(-1,1)
        delta=pred-native_glare
        zero=torch.zeros_like(lens_native)
        return pred,delta,zero,lens_native,zero,zero,zero,zero
