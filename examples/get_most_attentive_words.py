from bertsplain.explainer.lime_explainer import LimeExplainer
from bertsplain.explainer.integrated_gradients_explainer import IntegratedGradientsExplainer

model_checkpoint = "mediabiasgroup/babe-v3-roberta-fully-trained"
dataset = "mediabiasgroup/BABE-v3"
ann = IntegratedGradientsExplainer(model_checkpoint, dataset)
ann.explain_dataset(class_=1)
ann.get_wordcloud(silhouette_path="data/silh1.jpg")

