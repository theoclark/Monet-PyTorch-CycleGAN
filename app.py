from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from Model.model import Model, Generator, Residual_block, Upsample_block, Downsample_block
import os
import shutil
from werkzeug.utils import secure_filename

image_folder = './static/Images'
allowed_extensions = {'jpg'}

app = Flask(__name__)

model = Model('./Model/G_xy.pt', './static/Images/input_image.jpg', './static/Images/output_image.jpg')

SECRET_KEY = os.urandom(12)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = image_folder

@app.route("/")
def index():
    remove_image_directory()
    return render_template('index.html', show_images=False)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def remove_image_directory():
    if os.path.isdir('./static/Images'):
        shutil.rmtree('./static/Images')

@app.route('/prediction', methods=['POST'])
def upload_file():
    remove_image_directory()
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        os.mkdir('./static/Images')
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'original_image.jpg'))
        # model.predict("./static/Images/original_image.jpg" )
    return render_template('index.html', show_images=True)

@app.route('/download')
def download():
    path = "./static/Images/output_image.jpg"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
     app.run(debug=True)
