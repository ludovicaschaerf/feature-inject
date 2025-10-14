from diffusers import FluxPipeline, PixArtAlphaPipeline, StableDiffusion3Pipeline, AutoPipelineForText2Image
from typing import List, Dict, Callable, Union
import torch

def retrieve(io):
    if isinstance(io, tuple):
       return io[0]
    elif isinstance(io, torch.Tensor):
        return io
    else:
        raise ValueError("Input/Output must be a tensor, or 1-element tuple")


class HookedDiffusionAbstractPipeline:
    parent_cls = None
    pipe = None
    def __init__(self, pipe: parent_cls, use_hooked_scheduler: bool = False):
        self.__dict__['pipe'] = pipe
        self.use_hooked_scheduler = use_hooked_scheduler

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(cls.parent_cls.from_pretrained(*args, **kwargs))


    def run_with_hooks(self, 
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]], 
        **kwargs
    ):
        hooks = []
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]

        try:
            output = self.pipe(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
            if self.use_hooked_scheduler:
                self.pipe.scheduler.pre_hooks = []
                self.pipe.scheduler.post_hooks = []
        
        return output

    def run_with_cache(self, 
        *args,
        positions_to_cache: List[str],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs
    ):
        cache_input, cache_output = dict() if save_input else None, dict() if save_output else None
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output) for position in positions_to_cache
        ]
        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict['input'] = cache_input
        
        if save_output:
            for position, block in cache_output.items():
                try:
                    cache_output[position] = torch.stack(block, dim=1)
                except Exception as e:
                    print(e)
            cache_dict['output'] = cache_output
        return output, cache_dict

    def run_with_hooks_and_cache(self,
        *args,
        position_hook_dict: Dict[str, Union[Callable, List[Callable]]],
        positions_to_cache: List[str] = [],
        save_input: bool = False,
        save_output: bool = True,
        **kwargs
    ):
        cache_input, cache_output = dict() if save_input else None, dict() if save_output else None
        hooks = [
            self._register_cache_hook(position, cache_input, cache_output) for position in positions_to_cache
        ]
        
        for position, hook in position_hook_dict.items():
            if isinstance(hook, list):
                for h in hook:
                    hooks.append(self._register_general_hook(position, h))
            else:
                hooks.append(self._register_general_hook(position, hook))

        hooks = [hook for hook in hooks if hook is not None]
        output = self.pipe(*args, **kwargs)
        for hook in hooks:
            hook.remove()
        if self.use_hooked_scheduler:
            self.pipe.scheduler.pre_hooks = []
            self.pipe.scheduler.post_hooks = []

        cache_dict = {}
        if save_input:
            for position, block in cache_input.items():
                cache_input[position] = torch.stack(block, dim=1)
            cache_dict['input'] = cache_input

        if save_output:
            for position, block in cache_output.items():
                try:
                    cache_output[position] = torch.stack(block, dim=1)
                except Exception as e:
                    print(e)
            cache_dict['output'] = cache_output
        
        return output, cache_dict

    
    def _locate_block(self, position: str):
        block = self.pipe
        for step in position.split('.'):
            if step.isdigit():
                step = int(step)
                block = block[step]
            else:
                block = getattr(block, step)
        return block
    

    def _register_cache_hook(self, position: str, cache_input: Dict, cache_output: Dict):

        if position.endswith('$self_attention') or position.endswith('$cross_attention'):
            return self._register_cache_attention_hook(position, cache_output)

        if position == 'noise':
            def hook(model_output, timestep, sample, generator):
                if position not in cache_output:
                    cache_output[position] = []
                cache_output[position].append(sample)
            
            if self.use_hooked_scheduler:
                self.pipe.scheduler.post_hooks.append(hook)
            else:
                raise ValueError('Cannot cache noise without using hooked scheduler')
            return

        block = self._locate_block(position)

        def hook(module, input, kwargs, output):
            if cache_input is not None:
                if position not in cache_input:
                    cache_input[position] = []
                cache_input[position].append(retrieve(input))
            
            if cache_output is not None:
                if position not in cache_output:
                    cache_output[position] = []
                cache_output[position].append(retrieve(output))

        return block.register_forward_hook(hook, with_kwargs=True)

    def _register_cache_attention_hook(self, position, cache):
        attn_block = self._locate_block(position.split('$')[0])
        if position.endswith('$self_attention'):
            attn_block = attn_block.attn1
        elif position.endswith('$cross_attention'):
            attn_block = attn_block.attn2
        else:
            raise ValueError('Wrong attention type')

        def hook(module, args, kwargs, output):
            hidden_states = args[0]
            encoder_hidden_states = kwargs['encoder_hidden_states']
            attention_mask = kwargs['attention_mask']
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn_block.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = attn_block.to_q(hidden_states)


            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn_block.norm_cross is not None:
                encoder_hidden_states = attn_block.norm_cross(encoder_hidden_states)

            key = attn_block.to_k(encoder_hidden_states)
            value = attn_block.to_v(encoder_hidden_states)

            query = attn_block.head_to_batch_dim(query)
            key = attn_block.head_to_batch_dim(key)
            value = attn_block.head_to_batch_dim(value)

            attention_probs = attn_block.get_attention_scores(query, key, attention_mask)
            attention_probs = attention_probs.view(
                batch_size, 
                attention_probs.shape[0] // batch_size,
                attention_probs.shape[1],
                attention_probs.shape[2]
            )
            if position not in cache:
                cache[position] = []
            cache[position].append(attention_probs)
        
        return attn_block.register_forward_hook(hook, with_kwargs=True) 

    def _register_general_hook(self, position, hook):
        if position == 'scheduler_pre':
            if not self.use_hooked_scheduler:
                raise ValueError('Cannot register hooks on scheduler without using hooked scheduler')
            self.pipe.scheduler.pre_hooks.append(hook)
            return
        elif position == 'scheduler_post':
            if not self.use_hooked_scheduler:
                raise ValueError('Cannot register hooks on scheduler without using hooked scheduler')
            self.pipe.scheduler.post_hooks.append(hook)
            return

        block = self._locate_block(position)
        return block.register_forward_hook(hook)

    def to(self, *args, **kwargs):
        self.pipe = self.pipe.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self.pipe, name)

    def __setattr__(self, name, value):
        return setattr(self.pipe, name, value)

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


class HookedStableDiffusionPipeline(HookedDiffusionAbstractPipeline):
    parent_cls = None  # Default class attribute

    
    @classmethod
    def from_pretrained(cls, model_type, pretrained_model_name, *args, **kwargs):
        if model_type == 'flux':
            parent_cls = FluxPipeline
        elif model_type == 'sd3':
            parent_cls = StableDiffusion3Pipeline
        elif model_type == 'pixart':
            parent_cls = PixArtAlphaPipeline
        elif model_type == 'auto':
            parent_cls = AutoPipelineForText2Image
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        parent_pipeline_instance = parent_cls.from_pretrained(pretrained_model_name, *args, **kwargs)
        return cls(parent_pipeline_instance)
    
    
class ModifiedStableDiffusionPipeline(HookedDiffusionAbstractPipeline):
    # Modified pipeline that caches skip connections from the UNet down blocks.
    parent_cls = None  # Default class attribute

   
    @classmethod
    def from_pretrained(cls, model_type, pretrained_model_name, *args, **kwargs):
        if model_type == 'flux':
            parent_cls = FluxPipeline
        elif model_type == 'sd3':
            parent_cls = StableDiffusion3Pipeline
        elif model_type == 'pixart':
            parent_cls = PixArtAlphaPipeline
        elif model_type == 'auto':
            parent_cls = AutoPipelineForText2Image
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        parent_pipeline_instance = parent_cls.from_pretrained(pretrained_model_name, *args, **kwargs)
        return cls(parent_pipeline_instance)
    
    
    def __call__(self, *args, **kwargs):
        # Decide which UNet down blocks to cache.
        blocks_to_save = []
        
        # check if unet in self:
        if hasattr(self, 'unet'):
            for name, _ in self.unet.named_modules():
                if 'mid_block' in name:
                    if len(name.split('.')) == 3 and ('resnets' in name or 'attentions' in name):
                        blocks_to_save.append('unet.' + name)
                elif len(name.split('.')) == 4 and ('resnets' in name or 'attentions' in name):
                    blocks_to_save.append('unet.' + name)
                elif len(name.split('.')) == 3 and ('resnets' in name or 'attentions' in name):
                    blocks_to_save.append('unet.' + name)
        elif hasattr(self, 'transformer'):
            for name, _ in self.transformer.named_modules():
                    if 'single' not in name and 'transformer' in name and len(name.split('.')) == 2: # and '18' in name
                        blocks_to_save.append('transformer.' + name)

        print('Saved', len(blocks_to_save), 'blocks for injection.')
        
        final_output, cache_dict = self.run_with_cache(
            *args,
            positions_to_cache=blocks_to_save,
            save_input=False,
            save_output=True,
            **kwargs
        )
        
        # The cache dictionary is organized as: {'output': {position: tensor_list, ...}}.
        # Here we simply extract the cached skip activations.
        skip_connections = cache_dict.get('output', {})
        return final_output, skip_connections
    