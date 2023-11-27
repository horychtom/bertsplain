from bertsplain.explainer.lime_explainer import LimeExplainer

model_checkpoint = "mediabiasgroup/babe-v3-roberta-fully-trained"
dataset = "mediabiasgroup/BABE-v3"
ann = LimeExplainer(model_checkpoint, dataset)
ann.explain_dataset(class_=1)
ann.get_wordcloud(silhouette_path="data/silh1.jpg")
