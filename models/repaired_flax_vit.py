# coding=utf-8
# Copyright 2021 The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import flax.linen as nn
import jax
import jax.numpy as jnp

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from transformers.modeling_flax_utils import (
    ACT2FN,
)
from transformers.models.vit.modeling_flax_vit import FlaxViTSelfAttention,FlaxViTSelfOutput, FlaxViTOutput, FlaxViTPooler, FlaxViTEmbeddings
from transformers.models.vit.configuration_vit import ViTConfig

class RepairedFlaxViTAttention(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32
    tracker : bool = False
    repaired : bool = False
    
    def setup(self):
        self.attention = FlaxViTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxViTSelfOutput(self.config, dtype=self.dtype)
        self.batchnorm = nn.BatchNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        
        if self.tracker is True:
            self.batchnorm(attn_output, use_running_average=False)
        elif self.repaired is True:
            attn_output = self.batchnorm(attn_output, use_running_average=deterministic)
        
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs

class RepairedFlaxViTIntermediate(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    tracker : bool = False
    repaired : bool = False
    
    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.batchnorm = nn.BatchNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states, deterministic : bool = True):
        hidden_states = self.dense(hidden_states)
        if self.tracker is True:
            self.batchnorm(hidden_states, use_running_average=False)
        elif self.repaired is True:
            hidden_states = self.batchnorm(hidden_states, use_running_average=deterministic)
        hidden_states = self.activation(hidden_states)
        return hidden_states



class RepairedFlaxViTLayer(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    tracker : bool = False
    repaired : bool = False
    
    def setup(self):
        self.attention = RepairedFlaxViTAttention(self.config, dtype=self.dtype, tracker=self.tracker, repaired=self.repaired)
        self.batchnorm_attention = nn.BatchNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.intermediate = RepairedFlaxViTIntermediate(self.config, dtype=self.dtype, tracker=self.tracker, repaired=self.repaired)
        self.output = FlaxViTOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.batchnorm_after = nn.BatchNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # self.batchnorm = nn.BatchNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        # first residual connection
        attention_output = attention_output + hidden_states
        
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(attention_output)
        
        # repair attention output after layernorm
        if self.tracker:
            self.batchnorm_attention(layer_output, use_running_average=False)
        elif self.repaired:
            layer_output = self.batchnorm_attention(layer_output, use_running_average=deterministic)
        
        hidden_states = self.intermediate(layer_output, deterministic=deterministic)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)
        
        # repair layer output
        if self.tracker:
            self.batchnorm_after(hidden_states, use_running_average=False)
        elif self.repaired:
            hidden_states = self.batchnorm_after(hidden_states, use_running_average=deterministic)

        # if self.tracker:
        #     self.batchnorm(hidden_states, use_running_average=False)
        # elif self.repaired:
        #     hidden_states = self.batchnorm(hidden_states, use_running_average=deterministic)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class RepairedFlaxViTLayerCollection(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    tracker : bool = False
    repaired : bool = False

    def setup(self):
        self.layers = [
            RepairedFlaxViTLayer(self.config, name=str(i), dtype=self.dtype, tracker=self.tracker , repaired=self.repaired) for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class RepairedFlaxViTEncoder(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    tracker : bool = False
    repaired : bool = False
    def setup(self):
        self.layer = RepairedFlaxViTLayerCollection(self.config, dtype=self.dtype, tracker=self.tracker , repaired = self.repaired)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class RepairedFlaxViTModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    tracker : bool = False
    repaired : bool = False
    
    def setup(self):
        assert not (self.tracker and self.repaired)
        self.embeddings = FlaxViTEmbeddings(self.config, dtype=self.dtype)
        self.embeddings_batchnorm = nn.BatchNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.encoder = RepairedFlaxViTEncoder(self.config, dtype=self.dtype, tracker=self.tracker, repaired=self.repaired)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxViTPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(pixel_values, deterministic=deterministic)
        all_hidden_states = ()
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.tracker:
            self.embeddings_batchnorm(hidden_states, use_running_average=False)
        elif self.repaired:
            hidden_states = self.embeddings_batchnorm(hidden_states, use_running_average=deterministic)
        
        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
