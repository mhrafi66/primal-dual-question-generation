from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

MODEL_NAME = "facebook/bart-base"


def build_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, model_max_length=512, use_fast=True)


class PrimalDualQuestionGenerator(nn.Module):
    """Stable restored version of the course-project primal-dual model.

    It keeps the original project idea—QG + dual QA + uncommon-word distillation—
    while fixing numerical instability and the gold-question leakage discovered in
    the original submission.
    """

    def __init__(self, model_name: str = MODEL_NAME, with_teacher: bool = True):
        super().__init__()
        self.model_name = model_name
        self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        self.embedding_dim = self.bart.config.d_model
        self.vocab_size = self.bart.config.vocab_size
        self.pad_token_id = self.bart.config.pad_token_id
        self.bos_token_id = self.bart.config.bos_token_id
        self.eos_token_id = self.bart.config.eos_token_id
        self.mask_token_id = getattr(self.bart.config, "mask_token_id", self.vocab_size - 1)

        self.task_embedding = nn.Embedding(3, self.embedding_dim)
        self.segment_embedding = nn.Embedding(3, self.embedding_dim)

        self.start_index_ff_qa = nn.Linear(self.embedding_dim, 1)
        self.end_index_ff_qa = nn.Linear(self.embedding_dim * 2, 1)

        # Distillation teacher: frozen pretrained BART encoder + frozen pretrained LM head.
        # The original course implementation used BART-family components here rather than
        # the T5 teacher used by Wang et al.; this repository documents that deviation.
        self.teacher_encoder = None
        self.teacher_uw_head = None
        if with_teacher:
            teacher = BartForConditionalGeneration.from_pretrained(model_name)
            self.teacher_encoder = teacher.model.encoder
            self.teacher_uw_head = copy.deepcopy(teacher.lm_head)
            for p in self.teacher_encoder.parameters():
                p.requires_grad = False
            for p in self.teacher_uw_head.parameters():
                p.requires_grad = False
            self.teacher_encoder.eval()
            self.teacher_uw_head.eval()

        # Student projection starts from the same pretrained vocabulary projection.
        self.student_uw_head = copy.deepcopy(self.bart.lm_head)

    def _embed(self, input_ids: torch.Tensor, task_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        word = self.bart.model.shared(input_ids)
        task = self.task_embedding(task_ids)
        segment = self.segment_embedding(segment_ids)
        return word + (task + segment) / math.sqrt(self.embedding_dim)

    def encode_qg(
        self,
        input_ids: torch.Tensor,
        task_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ):
        attention_mask = input_ids.ne(self.pad_token_id)
        embedded = self._embed(input_ids, task_ids, segment_ids)
        encoded = self.bart.model.encoder(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return encoded.last_hidden_state, attention_mask

    @torch.no_grad()
    def generate_questions(
        self,
        input_ids: torch.Tensor,
        task_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        num_beams: int = 4,
        max_new_tokens: int = 48,
    ) -> torch.Tensor:
        encoded, attention_mask = self.encode_qg(input_ids, task_ids, segment_ids)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoded)
        return self.bart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    def _distillation_mask(self, input_ids: torch.Tensor, ratio: float = 0.10) -> torch.Tensor:
        valid = input_ids.ne(self.pad_token_id)
        valid &= input_ids.ne(self.bos_token_id)
        valid &= input_ids.ne(self.eos_token_id)
        mask = torch.zeros_like(valid)
        for row in range(input_ids.size(0)):
            candidates = torch.nonzero(valid[row], as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                continue
            k = max(1, int(round(float(candidates.numel()) * ratio)))
            perm = torch.randperm(candidates.numel(), device=input_ids.device)[:k]
            mask[row, candidates[perm]] = True
        return mask

    def forward(
        self,
        *,
        qg_input_ids: torch.Tensor,
        qg_task_ids: torch.Tensor,
        qg_segment_ids: torch.Tensor,
        question_input_ids: torch.Tensor,
        qa_input_ids: Optional[torch.Tensor] = None,
        qa_task_ids: Optional[torch.Tensor] = None,
        qa_segment_ids: Optional[torch.Tensor] = None,
        answer_start_positions: Optional[torch.Tensor] = None,
        uw_input_ids: Optional[torch.Tensor] = None,
        uw_task_ids: Optional[torch.Tensor] = None,
        uw_segment_ids: Optional[torch.Tensor] = None,
        compute_qa: bool = True,
        compute_kd: bool = True,
        distill_temperature: float = 2.0,
    ):
        # ---------------- QG ----------------
        qg_encoded, qg_attention_mask = self.encode_qg(qg_input_ids, qg_task_ids, qg_segment_ids)
        decoder_input_ids = question_input_ids[:, :-1]
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)
        decoded = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=qg_encoded,
            encoder_attention_mask=qg_attention_mask,
            return_dict=True,
        ).last_hidden_state
        qg_logits = self.bart.lm_head(decoded) + self.bart.final_logits_bias

        start_logits = None
        end_logits = None
        kd_loss = None

        # ---------------- dual QA ----------------
        if compute_qa:
            assert qa_input_ids is not None and qa_task_ids is not None and qa_segment_ids is not None
            qa_context_mask = qa_input_ids.ne(self.pad_token_id)
            qa_context_embedded = self._embed(qa_input_ids, qa_task_ids, qa_segment_ids)

            generated_task = torch.ones_like(decoder_input_ids)
            generated_segment = torch.full_like(decoder_input_ids, 2)
            generated_question_repr = decoded + (
                self.task_embedding(generated_task) + self.segment_embedding(generated_segment)
            ) / math.sqrt(self.embedding_dim)

            qa_embedded = torch.cat([qa_context_embedded, generated_question_repr], dim=1)
            qa_attention_mask = torch.cat([qa_context_mask, decoder_attention_mask], dim=1)
            qa_encoded = self.bart.model.encoder(
                inputs_embeds=qa_embedded,
                attention_mask=qa_attention_mask,
                return_dict=True,
            ).last_hidden_state

            passage_candidate_mask = qa_context_mask.clone()
            passage_candidate_mask &= qa_input_ids.ne(self.bos_token_id)
            passage_candidate_mask &= qa_input_ids.ne(self.eos_token_id)
            candidate_mask = torch.cat(
                [passage_candidate_mask, torch.zeros_like(decoder_attention_mask)], dim=1
            )
            finite_floor = torch.finfo(qa_encoded.dtype).min

            start_logits = self.start_index_ff_qa(qa_encoded).squeeze(-1)
            start_logits = start_logits.masked_fill(~candidate_mask, finite_floor)
            predicted_start = start_logits.argmax(dim=-1)

            # Teacher-force the gold start position when available. This avoids the
            # historical infinity bug where a wrong predicted start masked the gold
            # end position before cross-entropy was computed.
            start_for_end = predicted_start
            if answer_start_positions is not None:
                gold = answer_start_positions.view(-1)
                valid_gold = (gold >= 0) & (gold < qa_input_ids.size(1))
                start_for_end = torch.where(valid_gold, gold, predicted_start)

            gather_idx = start_for_end.view(-1, 1, 1).expand(-1, 1, qa_encoded.size(-1))
            start_state = qa_encoded.gather(1, gather_idx).expand(-1, qa_encoded.size(1), -1)
            end_features = torch.cat([qa_encoded, start_state], dim=-1)
            end_logits = self.end_index_ff_qa(end_features).squeeze(-1)
            # During training we mask only non-passage locations. Ordering constraints
            # are applied at prediction time, not by setting the gold end logit to -inf.
            end_logits = end_logits.masked_fill(~candidate_mask, finite_floor)

        # ---------------- uncommon-word distillation ----------------
        if compute_kd:
            assert uw_input_ids is not None and uw_task_ids is not None and uw_segment_ids is not None
            attention_mask = uw_input_ids.ne(self.pad_token_id)
            distill_mask = self._distillation_mask(uw_input_ids)
            masked_ids = uw_input_ids.clone()
            if self.mask_token_id is not None:
                masked_ids[distill_mask] = self.mask_token_id

            student_embedded = self._embed(masked_ids, uw_task_ids, uw_segment_ids)
            student_hidden = self.bart.model.encoder(
                inputs_embeds=student_embedded,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

            if self.teacher_encoder is None or self.teacher_uw_head is None:
                raise RuntimeError("compute_kd=True requires with_teacher=True")
            with torch.no_grad():
                teacher_hidden = self.teacher_encoder(
                    input_ids=masked_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).last_hidden_state

            if distill_mask.any():
                student_logits = self.student_uw_head(student_hidden[distill_mask])
                with torch.no_grad():
                    teacher_logits = self.teacher_uw_head(teacher_hidden[distill_mask])

                # Stable temperature-smoothed version of the paper's modified CE.
                # Paper: -sum(ŷ_en * log ŷ_pre), with T=2.
                student_probs = F.softmax(student_logits / distill_temperature, dim=-1)
                teacher_log_probs = F.log_softmax(teacher_logits / distill_temperature, dim=-1)
                kd_loss = -(student_probs * teacher_log_probs).sum(dim=-1).mean()
            else:
                kd_loss = qg_logits.new_zeros(())

        return {
            "qg_logits": qg_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "kd_loss": kd_loss,
        }
