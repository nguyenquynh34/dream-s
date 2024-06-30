from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from model import get_predictions, get_top_predictions, labels_places, labels_emotions

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predictions_places, scores_places = get_predictions(filepath, labels_places)
            predictions_emotions, scores_emotions = get_predictions(filepath, labels_emotions)

            top_places, top_scores_places = get_top_predictions(predictions_places, scores_places)
            top_emotions, top_scores_emotions = get_top_predictions(predictions_emotions, scores_emotions)

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.bar(top_places, top_scores_places, color='blue')
            plt.title('Places Prediction')
            plt.xlabel('Places')
            plt.ylabel('Scores')

            plt.subplot(1, 2, 2)
            plt.bar(top_emotions, top_scores_emotions, color='green')
            plt.title('Emotions Prediction')
            plt.xlabel('Emotions')
            plt.ylabel('Scores')

            plt.tight_layout()
            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chart.png')
            plt.savefig(chart_path)
            plt.close()

            return render_template('index.html', filename=filename, chart='chart.png',
                                   top_places=top_places, top_scores_places=top_scores_places,
                                   top_emotions=top_emotions, top_scores_emotions=top_scores_emotions, zip=zip)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
