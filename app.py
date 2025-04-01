# app.py (updated)
from flask import Flask, render_template, request, jsonify
import torch
from hydra import initialize, compose
from omegaconf import OmegaConf
from utils.data_manager import DataManager
from utils.evaluation import Evaluator
from utils.model_loader import load_models 



app = Flask(__name__)

# Initialize Hydra config
with initialize(version_base=None, config_path="./configs"):
    cfg = compose(config_name="data_config")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize components
data_manager = DataManager(cfg)
base_model, unlearned_model = load_models(device)
tokenizer = data_manager.tokenizer
evaluator = Evaluator(unlearned_model, tokenizer)

# Load datasets once at startup
forget_set = data_manager.load_forget_set()
retain_set = data_manager.load_retain_set()

@app.route('/compare', methods=['POST'])
def compare_models():
    data = request.json
    text = data['text']
    
    base_response = DataManager.generate_response(base_model, tokenizer, text)
    unlearned_response = DataManager.generate_response(unlearned_model, tokenizer, text)
    
    metrics = evaluator.knowledge_retention_score(
        forget_set=forget_set,
        retain_set=retain_set
    )
    
    return jsonify({
        'base_response': base_response,
        'unlearned_response': unlearned_response,
        'metrics': metrics,
        'diff': calculate_diff(base_response, unlearned_response)
    })

def calculate_diff(text1, text2):
    """Calculate text differences using difflib"""
    from difflib import ndiff
    diff = list(ndiff(text1.split(), text2.split()))
    return ' '.join([d for d in diff if d[0] in ('+', '-')])

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/api/compare', methods=['POST'])
def api_compare():
    data = request.json
    text = data['text']
    
    base_response = DataManager.generate_response(base_model, tokenizer, text)
    unlearned_response = DataManager.generate_response(unlearned_model, tokenizer, text)
    
    metrics = evaluator.knowledge_retention_score(
        forget_set=forget_set,
        retain_set=retain_set
    )
    
    return jsonify({
        'base_response': base_response,
        'unlearned_response': unlearned_response,
        'metrics': metrics,
        'diff': calculate_diff(base_response, unlearned_response)
    })
@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    metrics = evaluator.knowledge_retention_score(
        forget_set=forget_set,
        retain_set=retain_set
    )
    return jsonify(metrics)


if __name__ == '__main__':
    app.run(debug=True)
    