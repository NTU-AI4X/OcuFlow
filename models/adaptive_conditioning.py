from __future__ import annotations

"""Mask-free geometry-aware glare -> state conditioning.

Region probabilities remain a semantic prior for the backbone and geometry-
guided source distribution. This conditioner deliberately has no Structure
Token projection, Structure self-attention, or Structure-to-glare attention.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedder import BottleneckPatchEmbedder


class CanonicalPatchTokenEmbedder(nn.Module):
    def __init__(self, canonical_size: int, hidden_size: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.p = int(canonical_size)
        self.d = int(hidden_size)
        self.embedder = BottleneckPatchEmbedder(
            input_size=self.p,
            initial_patch_size=self.p,
            pca_channels=int(bottleneck),
            in_channels=3,
            hidden_size=self.d,
            bias=True,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"patches must be [B,T,3,P,P], got {tuple(patches.shape)}")
        b,t,c,h,w = patches.shape
        if c != 3 or h != self.p or w != self.p:
            raise ValueError(f"expected [B,T,3,{self.p},{self.p}], got {tuple(patches.shape)}")
        x = self.embedder(patches.reshape(b*t,c,h,w))
        if x.shape[1] != 1:
            raise RuntimeError("canonical patch embedder must emit one token per adaptive leaf")
        return x[:,0].reshape(b,t,self.d)


class GeometryEmbedding(nn.Module):
    def __init__(self, hidden_size: int, canvas_size: int = 256, base_patch: int = 16) -> None:
        super().__init__()
        self.canvas_size = float(canvas_size)
        self.base_patch = float(base_patch)
        inner = max(32, int(hidden_size)//4)
        self.net = nn.Sequential(nn.Linear(3,inner), nn.SiLU(), nn.Linear(inner,int(hidden_size)))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, leaves: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        v = leaves.float()
        x,y,s = v.unbind(-1)
        f = torch.stack([
            (y + .5*s)/self.canvas_size*2-1,
            (x + .5*s)/self.canvas_size*2-1,
            torch.log2(s.clamp_min(1)/self.base_patch),
        ],-1)
        return self.net(f).to(dtype=dtype)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        d=int(hidden_size); h=int(num_heads)
        if d % h: raise ValueError("hidden_size must divide num_heads")
        self.h=h; self.dh=d//h; self.d=d
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,d,bias=False)
        self.v=nn.Linear(d,d,bias=False); self.o=nn.Linear(d,d,bias=False)
        # The conditioner is added to a pretrained PMF backbone. Zero output
        # projection makes that residual branch an exact no-op at step 0 while
        # preserving gradients into the projection so conditioning can grow safely.
        nn.init.zeros_(self.o.weight)
        self.qn=nn.LayerNorm(self.dh, elementwise_affine=False)
        self.kn=nn.LayerNorm(self.dh, elementwise_affine=False)

    def forward(self, q_content: torch.Tensor, kv_content: torch.Tensor,
                q_geom: Optional[torch.Tensor]=None, kv_geom: Optional[torch.Tensor]=None,
                key_valid: Optional[torch.Tensor]=None) -> torch.Tensor:
        b,nq,d=q_content.shape; nk=kv_content.shape[1]
        qin=q_content if q_geom is None else q_content+q_geom.to(q_content.dtype)
        kin=kv_content if kv_geom is None else kv_content+kv_geom.to(kv_content.dtype)
        q=self.qn(self.q(qin).reshape(b,nq,self.h,self.dh))
        k=self.kn(self.k(kin).reshape(b,nk,self.h,self.dh))
        v=self.v(kv_content).reshape(b,nk,self.h,self.dh)
        logits=torch.einsum('bqhd,bkhd->bhqk',q/math.sqrt(self.dh),k)
        any_valid=None
        if key_valid is not None:
            valid=key_valid.bool(); any_valid=valid.any(-1)
            safe=valid.clone(); safe[:,0] |= ~any_valid
            logits=logits.masked_fill(~safe[:,None,None,:],-1e4)
        w=F.softmax(logits.float(),dim=-1).to(v.dtype)
        if any_valid is not None: w=w*any_valid[:,None,None,None].to(w.dtype)
        out=torch.einsum('bhqk,bkhd->bqhd',w,v).reshape(b,nq,d)
        return self.o(out)


@dataclass
class ConditioningOutput:
    conditioned_state: torch.Tensor
    state_tokens: torch.Tensor
    glare_tokens: torch.Tensor


class AdaptiveConditioner(nn.Module):
    def __init__(self, canonical_size:int, hidden_size:int, num_heads:int,
                 bottleneck:int=128, canvas_size:int=256, base_patch:int=16,
                 condition_gate_init:float=.1, context_gate_init:float=.05,
                 use_geometry_token:bool=True,
                 use_coarse_to_fine_context:bool=True) -> None:
        super().__init__(); d=int(hidden_size)
        self.use_geometry_token=bool(use_geometry_token)
        self.use_coarse_to_fine_context=bool(use_coarse_to_fine_context)
        self.glare_embed=CanonicalPatchTokenEmbedder(canonical_size,d,bottleneck)
        self.geom=GeometryEmbedding(d,canvas_size,base_patch) if self.use_geometry_token else None
        self.sg_q=nn.LayerNorm(d); self.sg_k=nn.LayerNorm(d); self.sg=CrossAttention(d,num_heads)
        if self.use_coarse_to_fine_context:
            self.gl_q=nn.LayerNorm(d); self.gl_k=nn.LayerNorm(d); self.gl=CrossAttention(d,num_heads)
        else:
            self.gl_q=self.gl_k=self.gl=None
        self.g_sg=nn.Parameter(torch.full((d,),float(condition_gate_init)))
        self.g_gl=(
            nn.Parameter(torch.full((d,),float(context_gate_init)))
            if self.use_coarse_to_fine_context else None
        )

    @staticmethod
    def masks(leaves: torch.Tensor, region_probs: torch.Tensor):
        sizes=leaves[...,2]
        fine=sizes<=8; coarse=sizes>=16
        semantic=(1.0-region_probs[...,0])>1e-4
        # Fine leaves are also the geometry-derived spectacle-band proxy.
        active=semantic | fine
        return active,fine,coarse

    @staticmethod
    def geometry_support(leaves: torch.Tensor, region_probs: torch.Tensor) -> torch.Tensor:
        sizes=leaves[...,2]
        fine=(sizes<=8).to(region_probs.dtype)
        semantic=(1.0-region_probs[...,0]).clamp(0,1)
        # Full semantic eye/periocular support; 0.75 support for fine lens-band leaves.
        return torch.maximum(semantic, 0.75*fine).clamp(0,1)

    def gate_metrics(self):
        zero=self.g_sg.new_zeros(())
        condition=self.g_sg.abs().mean()
        context=self.g_gl.abs().mean() if self.g_gl is not None else zero
        return {'condition_gate_mean':condition,'context_gate_mean':context}

    def _attention_without_coarse_to_fine(self,attention,q,kv,q_geom,kv_geom,key_valid,fine,coarse):
        delta=attention(q,kv,q_geom,kv_geom,key_valid)
        if not self.use_coarse_to_fine_context:
            fine_delta=attention(q,kv,q_geom,kv_geom,key_valid & ~coarse)
            delta=torch.where(fine[...,None],fine_delta,delta)
        return delta

    def forward_from_state_tokens(self,state_tokens,glare_patches,region_probs,leaves):
        b,t,d=state_tokens.shape
        # Disabling geometry tokens removes the learned (x,y,scale) route. The grid
        # stays fixed, so token count, crops, PMF RoPE and compute remain comparable.
        g=self.geom(leaves,state_tokens.dtype) if self.geom is not None else None
        active,fine,coarse=self.masks(leaves,region_probs)
        glare=self.glare_embed(glare_patches)
        delta=self._attention_without_coarse_to_fine(
            self.sg,self.sg_q(state_tokens),self.sg_k(glare),g,g,active,fine,coarse
        )
        state2=torch.where(active[...,None],state_tokens+self.g_sg.view(1,1,-1)*delta,state_tokens)
        if self.use_coarse_to_fine_context:
            assert self.gl is not None and self.gl_q is not None and self.gl_k is not None
            assert self.g_gl is not None
            delta=self.gl(self.gl_q(state2),self.gl_k(state2),g,g,coarse)
            conditioned=torch.where(fine[...,None],state2+self.g_gl.view(1,1,-1)*delta,state2)
        else:
            # Strict switch: fine tokens receive no coarse-token context. The
            # glare -> state conditioning before this point is unchanged.
            conditioned=state2
        return ConditioningOutput(conditioned,state_tokens,glare)
