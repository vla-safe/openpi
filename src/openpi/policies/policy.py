from collections.abc import Sequence
import logging
import os
import pathlib
from typing import Any, TypeAlias
import pickle
import imageio

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(
            model.sample_actions, 
            static_argnames=("temperature", "n_action_samples")
        )
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    def _infer_maybe_multi(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng =jax.random.split(self._rng)
        
        # Now the sample_actions function also returns auxiliary information that we need to handle.
        sample_action_outputs = self._sample_actions(
            sample_rng, 
            _model.Observation.from_dict(inputs), 
            **self._sample_kwargs
        )
        
        if isinstance(sample_action_outputs, tuple):
            if "Pi0.sample_actions" in self._sample_actions.__repr__():
                output_tokens, aux_outputs = sample_action_outputs
                output_tokens_truncated = output_tokens
            elif "Pi0FAST.sample_actions" in self._sample_actions.__repr__():
                output_tokens, aux_outputs = sample_action_outputs
                # process the output and cut off the unused positions
                step = aux_outputs["decode_step"].max()
                output_tokens_truncated = output_tokens[:, :step]
                aux_outputs['encoded'] = aux_outputs['encoded'][:, :step]
                aux_outputs['logits'] = aux_outputs['logits'][:, :step]
                aux_outputs['pre_logits'] = aux_outputs['pre_logits'][:, :step]
            else:
                raise ValueError(f"Unknown model type: {self._sample_actions.__repr__()}")
        else:
            output_tokens = sample_action_outputs
            output_tokens_truncated = output_tokens
            aux_outputs = {}
        
        # Apply the output transforms on each sampled actions from the model
        batch_size = output_tokens.shape[0]
        outputs = {
            "state": inputs["state"].repeat(batch_size, 0), # (batch_size, n_tokens)
            "actions": output_tokens, # (batch_size, 8)
        }
        
        outputs_transformed = []
        for i in range(batch_size):
            # apply output transforms. returns a dict with only "action" key
            outputs_transformed.append(
                self._output_transform(
                    jax.tree.map(lambda x: np.asarray(x[i, ...]), outputs)
                )
            )
        outputs_transformed = {
            k: jnp.stack([v[k] for v in outputs_transformed], axis=0)
            for k in outputs_transformed[0].keys()
        }
        
        # Contruct the output dict again. Now include both raw actions and (normalized) actions
        outputs = {
            "state": inputs["state"].repeat(batch_size, 0),
            "raw_actions": output_tokens_truncated,
            "actions": outputs_transformed["actions"],
        }
        outputs.update(aux_outputs)
        
        return outputs
        
    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        outputs = self._infer_maybe_multi(obs)
        
        # Get the first sample along the batch dimension
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(
        self, 
        policy: _base_policy.BasePolicy, 
        record_dir: str,
        save_images: bool = False,
    ):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0
        self._save_images = save_images

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results_multi = self._policy._infer_maybe_multi(obs)

        # when there are multiple sampled actions, only keep large features for the first one
        results = {}
        for k in results_multi:
            if k in ['actions', "decode_step", "raw_actions"]:
                results[k] = np.asarray(results_multi[k])
            else:
                results[k] = np.asarray(results_multi[k][0])
        
        # Parse the input for where and how to save policy records
        save_name = f"step_{self._record_step}"
        if "run/run_note" in obs: save_name = save_name + f"--{obs['run/run_note']}"
        if "run/task_id" in obs: save_name = save_name + f"--task_{obs['run/task_id']}"
        if "run/episode_idx" in obs: save_name = save_name + f"--ep_{obs['run/episode_idx']}"
        if "run/timestep" in obs: save_name = save_name + f"--t_{obs['run/timestep']}"
        
        record_dir = self._record_dir
        if "run/save_folder" in obs:
            record_dir = os.path.join(obs["run/save_folder"], "policy_records")
            record_dir = pathlib.Path(record_dir)
            record_dir.mkdir(parents=True, exist_ok=True)

        output_prefix = str(record_dir / save_name)
        
        # Store the input to meta
        meta_to_save = {}
        for k, v in obs.items():
            if "image" in k:
                if self._save_images:
                    k = k.replace("/", "_")
                    save_path = output_prefix + f"--{k}.jpg"
                    imageio.imsave(save_path, v)
            else:
                meta_to_save[k] = v
                
        # Save the output to meta
        meta_to_save.update(results)

        # Handle the pi0-fast intermediate outputs
        if "logits" in meta_to_save:
            logits, start_index, end_index = self._process_pi0fast_logits(meta_to_save["logits"])
            meta_to_save["logits"] = logits
            meta_to_save["action_start_index_in_vocab"] = start_index
            meta_to_save["action_end_index_in_vocab"] = end_index
            
        # if multiple actions are sampled, remove the intermediate outputs
        if results_multi["actions"].shape[0] > 1:
            if "encoded" in meta_to_save: del meta_to_save['encoded']
            if "logits" in meta_to_save: del meta_to_save['logits']
            if "pre_logits" in meta_to_save: del meta_to_save['pre_logits']
            
        # Save the meta data
        save_path = output_prefix + "--meta.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(meta_to_save, f)
        self._record_step += 1
        
        # Return only the results that are needed for control the robot
        # Only use the first sample along the batch dimension
        trimmed_results = {
            "state": np.asarray(results_multi["state"][0]),
            "actions": np.asarray(results_multi["actions"][0]),
        }

        return trimmed_results


    def _process_pi0fast_logits(self, logits):
        '''
        About the meaning of different token indices in vocabulary
        
        In the normal case:
        The first three output tokens should be [4022, 235292, 235248], decoded to "Actions: ". 
        The last two output token should be [235371, 1], decoded to "|" and "<eos>".
        The rest of the predicted tokens will be decoded into actual action tokens (looks like <loc0594>). 

        The underlying tokenizer has two stages
        - self._output_transform.transforms[0].tokenizer._paligemma_tokenizer 
        converts the predicted token id to the corresponding token string. Vocab size is
        self._paligemma_tokenizer.vocab_size() (text_vocab_size)
        - self._output_transform.transforms[0].tokenizer._fast_tokenizer
        converts the action tokens to the action trajectories. Vocab size is 
        self._output_transform.transforms[0].tokenizer._fast_tokenizer.vocab_size (action_vocab_size)
        
        
        According to FASTTokenizer._act_tokens_to_paligemma_tokens() function, the action tokens are 
        from text_vocab_size - 1 - self._fast_skip_tokens - action_vocab_size + 1,
            corresponding to action_vocab_size - 1, inclusive
        to text_vocab_size - 1 - self._fast_skip_tokens 
            corresponding to 0, inclusive
            
        So the actual slicing index should be 
        [
            text_vocab_size - self._fast_skip_tokens - action_vocab_size:
            text_vocab_size - self._fast_skip_tokens
        ]
        '''
        
        text_tokenizer = self._policy._output_transform.transforms[0].tokenizer._paligemma_tokenizer
        action_tokenizer = self._policy._output_transform.transforms[0].tokenizer._fast_tokenizer
        text_vocab_size = text_tokenizer.vocab_size()
        action_vocab_size = action_tokenizer.vocab_size
        fast_skip_tokens = self._policy._output_transform.transforms[0].tokenizer._fast_skip_tokens
        start_index = text_vocab_size - fast_skip_tokens - action_vocab_size
        end_index = text_vocab_size - fast_skip_tokens
        logits = logits[:, start_index:end_index]
        
        return logits, start_index, end_index