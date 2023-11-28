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
        self.static_embeddings.to(self.device)
        self.baseline = torch.zeros(self.static_embeddings.shape, device=self.device)

    def _path_integral_approximation(self, input):
        """
        Calculates the integrated gradients using the path integral approximation method.

        Args:
            input: The input to the model.

        Returns:
            integrated_grads: The integrated gradients calculated on a straight line
            between baseline (zero embedding) and the input embedding.
        """

        gradients = []
        # calculate gradients for each step on a path between baseline and input
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

        grads = torch.stack(gradients, dim=0)
        grads = (grads[:-1] + grads[1:]) / 2  # trapezoidal rule
        grads = grads.mean(dim=0)  # average over approx_steps_n

        integrated_grads = hidden_state * grads
        return integrated_grads

    def explain_sentence(self, sent):
        """
        Explains the given sentence by computing the attributions of each token using integrated gradients.

        Args:
            sent (str): The input sentence to be explained.

        Returns:
            dict: A dictionary containing the token attributions of the top-k tokens in the sentence.
                  The keys are the tokens and the values are the corresponding attributions.
        """

        input = self.encode_sentence(sent)
        integrated_grads = self._path_integral_approximation(input)

        attributions = (
            integrated_grads.abs().sum(dim=-1).squeeze(0)
        )  # abs and sum over embedding dimension
        attributions = attributions / torch.norm(attributions)  # normalize
        topk_attributions = attributions.topk(
            self.top_k if self.top_k < len(attributions) else len(attributions)
        )
        return {
            self.tokenizer.decode(input["input_ids"][:, idx]): attribution
            for idx, attribution in zip(
                topk_attributions.indices.tolist(), topk_attributions.values.tolist()
            )
        }
