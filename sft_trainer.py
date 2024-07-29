import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Literal
from transformers import DataCollatorForSeq2Seq,PreTrainedModel,BatchEncoding, Trainer
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler
from contextlib import nullcontext
from collections import defaultdict
from torch.nn import functional as F

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)

from transformers import Trainer
from torch.utils.data import DataLoader
class Collator(DataCollatorForSeq2Seq):
       def __init__(self, *args, max_steps,  **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_step = 0
        self.max_step = max_steps

       def __call__(self, batch):
        # 计算当前步数的阈值
        thresh_hold = self.cur_step / self.max_step
        p = np.random.random()

        # 根据阈值选择 'simple' 或 'hard' 数据
        if p > thresh_hold:
            for i, sample in enumerate(batch):
                # batch[i] = sample['simple']
                # print(batch[i])
                batch[i]={
              'input_ids':sample['input_ids'][0],
              'attention_mask':sample['attention_mask'][0],
              'labels':sample['labels'][0] 
              }
                
        else:
            for i, sample in enumerate(batch):
                # batch[i] = sample['hard']
                 batch[i]={
              'input_ids':sample['input_ids'][1],
              'attention_mask':sample['attention_mask'][1],
              'labels':sample['labels'][1] 
              }
        # for i, sample in enumerate(batch):
        #     print(f"Sample {i}: input_ids length = {len(sample['input_ids'])}, labels length = {len(sample['labels'])}")         
        self.cur_step += 1
        return super().__call__(batch)
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(self, train_dataloader,
        model: Union["PreTrainedModel", torch.nn.Module],
        finetuning_args: "FinetuningArguments",
        disable_dropout: bool = True,
        **kwargs) -> None:
        super().__init__(model,**kwargs)

        self.finetuning_args = finetuning_args
        self.reference_free = False
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self._peft_has_been_casted_to_bf16 = False
        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

      
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
        self.train_dataloader=train_dataloader
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
    def get_train_dataloader(self):
        return self.train_dataloader
    
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
  

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
