from bertsplain.explainer.base_explainer import BaseExplainer
from lime.lime_text import LimeTextExplainer
from typing import Dict
import torch.nn.functional as F


class LimeExplainer(BaseExplainer):
    def __init__(self, model_checkpoint: str, dataset: str, top_k=5):
        super().__init__(model_checkpoint, dataset, top_k)
        self.text_explainer = LimeTextExplainer(class_names=self.class_names)

    def explain_sentence(self, sent) -> Dict[str, float]:
        def predictor(texts):
            outputs = self.model(**self.tokenizer(texts, return_tensors="pt", padding=True))
            tensor_logits = outputs[0]
            probas = F.softmax(tensor_logits, dim=1).detach().numpy()
            return probas

        exp = self.text_explainer.explain_instance(sent, predictor, num_features=5, num_samples=100)
        return {k: v for k, v in exp.as_list()}
