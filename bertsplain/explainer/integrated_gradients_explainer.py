from bertsplain.explainer import BaseExplainer

import torch
from torch.autograd import grad
from copy import deepcopy


class IntegratedGradientsExplainer(BaseExplainer):
    def __init__(self, model_checkpoint: str, dataset: str, top_k=5, approx_steps_n=50):
        super().__init__(model_checkpoint, dataset, top_k)
        self.approx_steps_n = approx_steps_n
        self.word_embeddings = next(self.model.children()).embeddings.word_embeddings.weight
        self.static_embeddings = deepcopy(self.word_embeddings.data)
        self.static_embeddings.to_device(self.device)
        self.baseline = torch.zeros(self.static_embeddings.shape,device=self.device)

    def explain_sentence(self, sent):
        input = self.encode_sentence(sent)

        gradients = []
        for step in range(self.approx_steps_n + 1):
            alpha = step / self.approx_steps_n
            self.word_embeddings.data = self.baseline + alpha * (
                self.static_embeddings - self.baseline
            )
            output = self.model(**input, output_hidden_states=True)
            output_value = output.logits.max()
            hidden_state = output.hidden_states[0]
            gradient = grad(output_value, hidden_state, grad_outputs=torch.ones_like(output_value))[
                0
            ]
            gradients.append(gradient)

        label = output.logits.argmax(dim=1).tolist()  # actual prediction at the final step
        grads = torch.stack(gradients, dim=0)
        grads = (grads[:-1] + grads[1:]) / 2  # trapezoidal rule
        avg_grads = grads.mean(dim=0)
        integraded_grads = hidden_state * avg_grads
        attributions = integraded_grads.abs().sum(dim=-1).squeeze(0)
        topk_attributions = (attributions / torch.norm(attributions)).topk(
            self.top_k if self.top_k < len(attributions) else len(attributions)
        )
        return {
            self.tokenizer.decode(input["input_ids"][:, idx]): attribution
            for idx, attribution in zip(
                topk_attributions.indices.tolist(), topk_attributions.values.tolist()
            )
        }

    
