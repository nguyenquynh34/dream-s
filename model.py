import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import matplotlib.pyplot as plt
import clip

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

labels_places = ['pub', 'restaurant', 'grocery store', 'supermarket', 'party']
labels_emotions = ['Happy', 'Angry', 'Enjoyable', 'Relaxed', 'Neutral']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_predictions(image_path, labels):
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of {label}") for label in labels]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    predictions = [labels[idx] for idx in indices]
    scores = [100 * val.item() for val in values]

    return predictions, scores

def get_top_predictions(predictions, scores, top_n=2):
    sorted_predictions = sorted(zip(predictions, scores), key=lambda x: x[1], reverse=True)
    top_predictions = [pred for pred, _ in sorted_predictions[:top_n]]
    top_scores = [score for _, score in sorted_predictions[:top_n]]

    return top_predictions, top_scores

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predictions_places, scores_places = get_predictions(filepath, labels_places)
            predictions_emotions, scores_emotions = get_predictions(filepath, labels_emotions)
            top_places, top_scores_places = get_top_predictions(predictions_places, scores_places)
            top_emotions, top_scores_emotions = get_top_predictions(predictions_emotions, scores_emotions)

            # Create plots
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.bar(labels_places, scores_places, color='blue')
            plt.title('Dự đoán địa điểm')
            plt.ylabel('Xác suất (%)')

            plt.subplot(1, 2, 2)
            plt.bar(labels_emotions, scores_emotions, color='green')
            plt.title('Dự đoán cảm xúc')
            plt.ylabel('Xác suất (%)')

            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chart.png')
            plt.savefig(chart_path)
            plt.close()

            return render_template('results.html', image_url=url_for('static', filename='uploads/' + filename), 
                                   chart_url=url_for('static', filename='uploads/chart.png'),
                                   places=top_places, scores_places=top_scores_places,
                                   emotions=top_emotions, scores_emotions=top_scores_emotions)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
