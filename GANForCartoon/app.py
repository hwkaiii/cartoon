# app.py
import os
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from main import get_face, get_mask_U2net, get_face_white_bg_photo2cartoon, get_cartoon_face_photo2cartoon

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['IMAGE_FOLDER'] = 'static/images/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# 确保上传和输出的文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' in request.files:
        image = request.files['file']
        if image.filename != '':
            if image and allowed_file(image.filename):
                img_name = secure_filename(image.filename)
                original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
                image.save(original_image_path)

                # 读取原始图像并转换为RGB格式
                img = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)

                # 调用 main.py 中的函数进行图像处理
                face = get_face(img, 0.8, img_name)
                mask = get_mask_U2net(face, img_name)
                face_white_bg = get_face_white_bg_photo2cartoon(np.dstack((face, mask)), img_name)
                # 处理卡通效果
                cartoon_face = get_cartoon_face_photo2cartoon(face_white_bg, mask, img_name)

                # 保存处理后的图像
                cv2.imwrite(os.path.join(app.config['IMAGE_FOLDER'], 'original_' + img_name),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(app.config['IMAGE_FOLDER'], 'face_' + img_name),cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(app.config['IMAGE_FOLDER'], 'face_white_bg_' + img_name),cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(app.config['IMAGE_FOLDER'], 'cartoon_' + img_name),cv2.cvtColor(cartoon_face, cv2.COLOR_RGB2BGR))

                # 渲染模板显示原始图像和卡通化后的图像
                return render_template('display.html',
                                       original_img='original_' + img_name,
                                       face_img='face_' + img_name,
                                       face_white_bg_img='face_white_bg_' + img_name,
                                       cartoon_img='cartoon_' + img_name)
    return 'No file uploaded.'


@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)