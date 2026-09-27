"""Teacher → student knowledge distillation against catastrophic forgetting.

The previous champion (teacher) is frozen; its outputs are computed under
``torch.no_grad()`` and are therefore detached.  The student is pulled towards the
teacher's

* tempered event probabilities (binary KL divergence, scaled by T²),
* regression means and aleatoric log-variances (MSE in normalised space),
* optionally its latent geometry (cosine distance of embeddings) when the student was
  cloned from the teacher, so embeddings stay comparable across model versions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nardis_neural.data.datasets import Batch
from nardis_neural.models.main import ModelOutput, NardisNeuralNetwork

Tensor = torch.Tensor


def binary_kl(teacher_logits: Tensor, student_logits: Tensor, temperature: float) -> Tensor:
    t = torch.sigmoid(teacher_logits / temperature)
    log_s = F.logsigmoid(student_logits / temperature)
    log_1ms = F.logsigmoid(-student_logits / temperature)
    t_c = t.clamp(1e-6, 1 - 1e-6)
    kl = t_c * (torch.log(t_c) - log_s) + (1 - t_c) * (torch.log(1 - t_c) - log_1ms)
    return kl.mean() * temperature**2


class DistillationLoss:
    def __init__(
        self,
        teacher: NardisNeuralNetwork,
        temperature: float = 2.0,
        weight: float = 0.5,
        regression_weight: float = 1.0,
        embedding_weight: float = 0.1,
    ) -> None:
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.temperature = temperature
        self.weight = weight
        self.regression_weight = regression_weight
        self.embedding_weight = embedding_weight

    def to(self, device: torch.device) -> DistillationLoss:
        self.teacher.to(device)
        return self

    @torch.no_grad()
    def teacher_output(self, batch: Batch) -> ModelOutput:
        self.teacher.eval()
        out: ModelOutput = self.teacher(batch)
        return out

    def __call__(self, student: ModelOutput, batch: Batch) -> dict[str, Tensor]:
        teacher = self.teacher_output(batch)
        cls = torch.stack(
            [
                binary_kl(teacher.logits[k].detach(), student.logits[k], self.temperature)
                for k in student.logits
            ]
        ).mean()
        reg_terms = []
        for k in student.means:
            reg_terms.append(F.mse_loss(student.means[k], teacher.means[k].detach()))
            reg_terms.append(0.5 * F.mse_loss(student.logvars[k], teacher.logvars[k].detach()))
        reg = torch.stack(reg_terms).mean()
        emb = 1.0 - F.cosine_similarity(student.embedding, teacher.embedding.detach(), dim=-1).mean()
        return {
            "distill_cls": self.weight * cls,
            "distill_reg": self.weight * self.regression_weight * reg,
            "distill_emb": self.weight * self.embedding_weight * emb,
        }
