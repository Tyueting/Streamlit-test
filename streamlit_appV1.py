#streamlit run streamlit_app.py

import streamlit as st
import os
import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import swin_tiny_patch4_window7_224
from PIL import Image
import os
import rarfile
import csv
import pandas as pd
import cv2

#st.set_page_config(page_title="Demo", initial_sidebar_state="auto", layout="wide")
#@st.cache(allow_output_mutation=True)
if not os.path.exists("uploads"):
    os.makedirs("uploads")

progress_status = 0

def pre_process(image_path):

    img = cv2.imread(image_path)

    ####颜色分割
    #对图片进行模糊处理减少背景颜色影响
    blur = cv2.blur(img,(5,5))
    blur0=cv2.medianBlur(blur,5)
    blur1= cv2.GaussianBlur(blur0,(5,5),0)
    blur2= cv2.bilateralFilter(blur1,9,75,75)
    #转换为HSV图片
    hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([10, 70, 0])#hsv三个通道的最小值
    high_yellow = np.array([35, 255, 255])#hsv三个通道的最大值
    mask = cv2.inRange(hsv, low_yellow, high_yellow)
    res = cv2.bitwise_and(img,img, mask= mask)

    ####滤波剪裁
    blur_2 = cv2.blur(res,(5,5))
    img_blur=cv2.medianBlur(blur_2,5)

    height, width = img.shape[:2]  # 原始分辨率
    # pix = 1200
    pix = 600
    # 等比例缩放到pix=400
    scale = pix / width
    # 缩放后分辨率
    width = pix
    height = int(height * scale)
    img_resize = cv2.resize(img_blur, (width, height))

    # 上下左右填充的一个大小值
    top_size, bottom_size, left_size, right_size = (300, 300, 300, 300)

    constant = cv2.copyMakeBorder(img_resize, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)
    height, width = constant.shape[:2]
    h=int(height/2)
    w = int(width / 2)
    img_crop = constant[h-320:h+320, w-320:w+320]
    return img_crop

def predict_image(image_path, model):
        # 打开并预处理图片
        img = pre_process(image_path)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = transform(img).unsqueeze(0)  # 添加batch维度

        # 使用模型进行预测
        with torch.no_grad():
            output = model(img)
            probabilities = F.softmax(output, dim=1)
        return probabilities

def model_compile(target_file_path):
        # 加载模型和权重
    model = swin_tiny_patch4_window7_224(num_classes=3)
    model.load_state_dict(torch.load('./best-65-0.9895833333333334.pth'))   ###./
    model.eval()  # 设置模型为评估模式

    results = []
    filename = os.path.basename(target_file_path)
    # 文件夹路径
    probabilities = predict_image(target_file_path, model)
    max_prob_index = torch.argmax(probabilities)
    if max_prob_index == 0:
        label = 'C2F'
    elif max_prob_index == 1:
        label = 'C3F'
    elif max_prob_index == 2:
        label = 'C4F'
    # 输出预测结果
    results.append((filename, probabilities, label))
    return results

def batch_model_compile(target_file_path):
        # 加载模型和权重
    model = swin_tiny_patch4_window7_224(num_classes=3)
    model.load_state_dict(torch.load('./best-65-0.9895833333333334.pth'))
    model.eval()  # 设置模型为评估模式

    # 文件夹路径
    rar_file_path = target_file_path
    extract_path = os.getcwd()
    try:
        with rarfile.RarFile(rar_file_path) as rf:
            rf.extractall(path=extract_path)
        print(f"Successfully extracted {rar_file_path} to {extract_path}")
    except rarfile.Error as e:
        print(f"Failed to extract {rar_file_path}: {e}")
    file_name_without_extension = os.path.splitext(os.path.basename(target_file_path))[0]
    folder_path = os.path.join(extract_path, file_name_without_extension)
    #folder_name = os.listdir(folder)
    results = []
    #for name in folder_name:
    #    folder_path = os.path.join(folder,str(name))
    #    print(folder_path)
    # 遍历文件夹内的图片
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            probabilities = predict_image(image_path, model)
            max_prob_index = torch.argmax(probabilities)
            if max_prob_index == 0:
                label = 'C2F'
            elif max_prob_index == 1:
                label = 'C3F'
            elif max_prob_index == 2:
                label = 'C4F'
            # 输出预测结果
            results.append((filename, probabilities, label))
    return results

def save_results_to_csv(results, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['文件名', '类别概率', '类别'])
        writer.writerows(results)

def main():
    st.title('烟叶分级数据评测系统')
    st.selectbox(label = '请输入评测模型', options = ('VGG', 'ResNet', 'MobileNet', 'DenseNet', 'EfficientNetV1', 'EfficientNetV2', 'Vision Transformer', 'Swin Transformer', 'ConvNeXt', 'Mobile Vision Transformer'), index = 7, format_func = str, help="不同模型")
    mode_option = st.selectbox(label = '请输入评测方式', options = ('批量评测', '单张评测'), format_func = str)
    
    st.write("上传文件：")
    uploaded_file = st.file_uploader("None", label_visibility='collapsed', type=None)
    if uploaded_file is not None:  
        # 指定你希望存放文件的目录
        target_dir = "./uploads"
        # 构建目标文件的完整路径
        target_file_path = os.path.join(target_dir, uploaded_file.name)
        st.write("上传时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))       
        # 将文件从临时目录移动到目标目录
        #shutil.copy(uploaded_file, target_file_path)
        with open(target_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())        
        # 现在target_file_path包含了文件的绝对路径
        #print(f"文件已移动到: {target_file_path}")
    else:
        print("没有上传文件。")
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(formatted_time)

    if st.button("一键评测"):
        if mode_option == '单张评测':
            results = model_compile(target_file_path)
            csv_path = './results.csv'
            save_results_to_csv(results, csv_path)
            df = pd.read_csv(csv_path, encoding='gbk')
            st.dataframe(df)
            image = Image.open(target_file_path)
            st.image(image, use_column_width=True)            
        elif mode_option == '批量评测':
            results = batch_model_compile(target_file_path)
            csv_path = './results.csv'
            save_results_to_csv(results, csv_path)
            df = pd.read_csv(csv_path, encoding='gbk')
            csv = df.to_csv(index=False, encoding='utf-8')             ######################下载乱码
            st.write("本次批量评测结果已保存到文件中")
            st.download_button(label="下载CSV 文件", data=csv, file_name="predict.csv", mime="text/csv")
            st.dataframe(df)
    else:
        st.stop()

if __name__ == '__main__':
    main()
#streamlit run streamlit_appV1.py