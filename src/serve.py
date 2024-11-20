import os
from transformers import pipeline
import json

src_basepath = "/shared/src"
def init(context):
    
    model_name = os.environ.get("MODEL_NAME", "eurovoc-classifier-110")
    model = context.project.get_model(model_name)
    path = model.download()

    config = [f for f in model.status.files if f["name"] == "config.json"]
    rel_path = os.path.dirname(config[0]["path"])
    path = os.path.join(path, rel_path)

    print(f"model saved to {path}")

    # Retrieve the path to the model from the environment variables
    device = os.environ.get("DEVICE", "cuda:0")
    pred_type = os.environ.get("PRED_TYPE", "label")
    language = os.environ.get("MODEL_LANGUAGE", "it")
    threshold = os.environ.get("THRESHOLD", None) # If None, the pipeline will use top_k
    top_k = os.environ.get("TOP_K", None) # If None, all the labels will be returned. Only considered if threshold is None
    
    if top_k is not None:
        top_k = int(top_k)
    if threshold is not None:
        threshold = float(threshold)
        top_k = None

    classifier = pipeline(
        "text-classification",
        model=path,
        tokenizer=path,
        config=os.path.join(path, "config.json"),
        device=device,
        top_k=top_k,
    )
    setattr(context, "model", classifier)

    if pred_type == "label":
        with open(f"{src_basepath}/config/label_mappings_tc/{language}.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
            setattr(context, "labels", labels)
            
def serve(context, event):
    tokenizer_kwargs = {"padding": "max_length", "truncation": True, "max_length": 512}

    threshold = os.environ.get("THRESHOLD", None) # If None, the pipeline will use top_k
    pred_type = os.environ.get("PRED_TYPE", "id")
    
    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    else:
        body = event.body

    text = body["text"]
    predictions = context.model(text, **tokenizer_kwargs)
    if not threshold:
        # If no threshold is specified, return all the predictions
        if pred_type == "id":
            return {"predictions": predictions[0]}
        elif pred_type == "label":
            # Map the label ids to the actual labels
            to_return = {"predictions": []}
            for pred in predictions[0]:
                try:
                    to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": context.labels[pred["label"]]})
                except KeyError:
                    # If the label is not found in the label mappings, return an empty string
                    to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": ""})
            return to_return
    else:
        # If a threshold is specified, return only the predictions with a score higher than the threshold
        if pred_type == "id":
            return {"predictions": [pred for pred in predictions[0] if pred["score"] >= threshold]}
        elif pred_type == "label":
            # Map the label ids to the actual labels
            to_return = {"predictions": []}
            for pred in predictions[0]:
                try:
                    if pred["score"] >= threshold:
                        to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": context.labels[pred["label"]]})
                except KeyError:
                    # If the label is not found in the label mappings, return an empty string
                    to_return["predictions"].append({"label": pred["label"], "score": pred["score"], "term": ""})
            return to_return