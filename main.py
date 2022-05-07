import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QLabel, QPushButton, QLineEdit
from PyQt5.uic import loadUi
from libs.harris import apply_harris_operator, map_indices_to_image
from libs.feature_match import apply_feature_matching, calculate_ncc, calculate_ssd
from libs.SIFT import Sift
import cv2
import numpy as np
import time
from pathlib import Path


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("gui.ui",self)
        
        # Define harris widgets
        self.browse = self.findChild(QPushButton, "browse")
        self.apply_harris = self.findChild(QPushButton, "apply_harris")
        self.input_image = self.findChild(QLabel, "input_image_1")
        self.output_image = self.findChild(QLabel, "output_image_1")
        self.sensitivity = self.findChild(QLineEdit, "lineEdit_sensitivity")
        self.threshold = self.findChild(QLineEdit, "lineEdit_threshold")
        self.harris_time = self.findChild(QLabel, "harris_time")
    
        self.browse.clicked.connect(self.browsefiles_harris)
        self.apply_harris.clicked.connect(self.harris_operator)
        
        
        # Define feature_matching widgets
        self.browse_image1_feature = self.findChild(QPushButton, "browse_2")
        self.browse_image2_feature = self.findChild(QPushButton, "browse_3")
        self.apply_ncc = self.findChild(QPushButton, "apply_ncc")
        self.apply_ssd = self.findChild(QPushButton, "apply_ssd")
        
        self.input_image1 = self.findChild(QLabel, "input_feature_1")
        self.input_image2 = self.findChild(QLabel, "input_feature_2")
        self.output_ncc = self.findChild(QLabel, "output_ncc")
        self.output_ssd = self.findChild(QLabel, "output_ssd")
        
        self.ncc_time = self.findChild(QLabel, "ncc_time")
        self.ssd_time = self.findChild(QLabel, "ssd_time")
    
        self.browse_image1_feature.clicked.connect(self.browsefiles_featureTab1)
        self.browse_image2_feature.clicked.connect(self.browsefiles_featureTab2)
        
        self.apply_ncc.clicked.connect(self.ncc_operator)
        self.apply_ssd.clicked.connect(self.ssd_operator)
        

        # Define sift widgets
        self.browse_sift = self.findChild(QPushButton, "browse_4")
        self.apply_sift = self.findChild(QPushButton, "apply_sift")
        self.input_sift = self.findChild(QLabel, "input_sift")
        self.output_sift = self.findChild(QLabel, "output_sift")
        self.sift_time = self.findChild(QLabel, "sift_time")
    
        self.browse_sift.clicked.connect(self.browsefiles_sift)
        self.apply_sift.clicked.connect(self.sift_operator)


        self.show()


    ##### ---------------- HARRIS FUNCTIONS ----------------- #####
    def browsefiles_harris(self):
        fname, file_format = QFileDialog.getOpenFileName(self, 'Load image', 'images/', 'All files (*);;PNG Files (*.png);;JPG Files(*.jpg);;PNG Files (*.jpeg)')
        
        # If the file is loaded successfully
        if fname != "":
            self.file_path_1.setText(fname.split('/')[-1]) # image_name
            print(f"image path: {fname}")
            
            self.pixmap_1 = QPixmap(fname)        
            self.input_image.setPixmap(self.pixmap_1)
            ## Read the input image
            self.img_bgr = cv2.imread(fname)
            
            
    def harris_operator(self):        
        sensitivity = float(self.sensitivity.text())
        threshold = float(self.threshold.text())
        
        # Calculate function run time
        start_time = time.time()
        harris_response = apply_harris_operator(source=self.img_bgr, k=sensitivity)
        src_img = map_indices_to_image(source=self.img_bgr, harris_response=harris_response, threshold=threshold)
        end_time = time.time()
        
        output_path = 'images/output_harris.jpg' 
        cv2.imwrite(output_path,src_img)
        
        path = Path(__file__).parent.absolute()
        output_image_path = str(path) + '/' + str(output_path)
        
        self.pixmap_2 = QPixmap(output_image_path)        
        self.output_image.setPixmap(self.pixmap_2)
        
        harris_time = format(end_time - start_time, '.5f')
        print(f'Harris computation time = {harris_time} sec')
        self.harris_time.setText(str(harris_time) + ' sec')

        
    
    ##### ---------------- FEATURE MATCHING FUNCTIONS ----------------- #####
    def browsefiles_featureTab1(self):
        fname, file_format = QFileDialog.getOpenFileName(self, 'Load image', 'images/', 'All files (*);;PNG Files (*.png);;JPG Files(*.jpg);;PNG Files (*.jpeg)')
        
        # If the file is loaded successfully
        if fname != "":            
            self.pixmap_feature_1 = QPixmap(fname)
            self.input_image1.setPixmap(self.pixmap_feature_1)
            
            ## Read the input image
            self.img_bgr1 = cv2.imread(fname)            
            self.img_gray1 = cv2.cvtColor(self.img_bgr1, cv2.COLOR_BGR2GRAY)
            
    def browsefiles_featureTab2(self):
        fname, file_format = QFileDialog.getOpenFileName(self, 'Load image', 'images/', 'All files (*);;PNG Files (*.png);;JPG Files(*.jpg);;PNG Files (*.jpeg)')
        
        # If the file is loaded successfully
        if fname != "":            
            self.pixmap_feature_2 = QPixmap(fname)
            self.input_image2.setPixmap(self.pixmap_feature_2)
            
            ## Read the input image
            self.img_bgr2 = cv2.imread(fname)            
            self.img_gray2 = cv2.cvtColor(self.img_bgr2, cv2.COLOR_BGR2GRAY)
        
        
    def feature_matching_operator(self):
        sift = cv2.SIFT_create()
        self.keypoints_1, self.descriptors_1 = sift.detectAndCompute(self.img_gray1, None)
        self.keypoints_2, self.descriptors_2 = sift.detectAndCompute(self.img_gray2, None)
        
  
    def ncc_operator(self):
        self.feature_matching_operator()
        start_ncc = time.time()
        matches_ncc = apply_feature_matching(self.descriptors_1, self.descriptors_2, calculate_ncc)
        matches_ncc = sorted(matches_ncc, key=lambda x: x.distance, reverse=True)
        matched_image_ncc = cv2.drawMatches(self.img_gray1, self.keypoints_1, self.img_gray2, self.keypoints_2, matches_ncc[:30], self.img_gray2, flags=2)
        end_ncc = time.time()
             
        ncc_output_path = 'images/ncc_output.jpg' 
        cv2.imwrite(ncc_output_path, matched_image_ncc)
        
        path = Path(__file__).parent.absolute()
        output_ncc_path = str(path) + '/' + str(ncc_output_path)
        
        self.pixmap_ncc = QPixmap(output_ncc_path)        
        self.output_ncc.setPixmap(self.pixmap_ncc)     
        
        ncc_time = format(end_ncc - start_ncc, '.5f')
        print(f'Computation time of Normalized Cross Correlation = {ncc_time} sec')
        self.ncc_time.setText(str(ncc_time) + ' sec')
        
        
    def ssd_operator(self):
        self.feature_matching_operator()
        start_ssd = time.time()
        matches_ssd = apply_feature_matching(self.descriptors_1, self.descriptors_2, calculate_ssd)
        matches_ssd = sorted(matches_ssd, key=lambda x: x.distance, reverse=True)
        matched_image_ssd = cv2.drawMatches(self.img_gray1, self.keypoints_1, self.img_gray2, self.keypoints_2, matches_ssd[:30], self.img_gray2, flags=2)
        end_ssd = time.time()
        
        ssd_output_path = 'images/ssd_output.jpg' 
        cv2.imwrite(ssd_output_path, matched_image_ssd)

        path = Path(__file__).parent.absolute()
        output_ssd_path = str(path) + '/' + str(ssd_output_path)
        
        self.pixmap_ssd = QPixmap(output_ssd_path)        
        self.output_ssd.setPixmap(self.pixmap_ssd)
        
        ssd_time = format(end_ssd - start_ssd, '.5f')
        print(f'Computation time of Sum Square Distance = {ssd_time} sec')
        self.ssd_time.setText(str(ssd_time) + ' sec')
        
        
    ##### ---------------- SIFT FUNCTIONS ----------------- #####
    def browsefiles_sift(self):
        fname, file_format = QFileDialog.getOpenFileName(self, 'Load image', 'images/', 'All files (*);;PNG Files (*.png);;JPG Files(*.jpg);;PNG Files (*.jpeg)')
        
        # If the file is loaded successfully
        if fname != "":            
            self.pixmap_sift = QPixmap(fname)
            self.input_sift.setPixmap(self.pixmap_sift)
            
            ## Read the input image
            self.img_bgr_sift = cv2.imread(fname, 1)         

        
    def sift_operator(self):
        start_sift = time.time()
        kp, dc =Sift(self.img_bgr_sift)
        end_sift = time.time()
        
        ###### GET THE OUTPUT IMAGE OF SIFT !!
        
        sift_output_path = 'images/sift_output.jpg' 
        # cv2.imwrite(sift_output_path, sift_output_image)
        
        path = Path(__file__).parent.absolute()
        output_sift_path = str(path) + '/' + str(sift_output_path)
        
        self.pixmap_sift = QPixmap(output_sift_path)        
        self.output_sift.setPixmap(self.pixmap_sift)
        
        
        sift_time = format(end_sift - start_sift, '.5f')
        print(f'Computation time of Normalized Cross Correlation = {sift_time} sec')
        self.sift_time.setText(str(sift_time) + ' sec')
        
        
app = QApplication(sys.argv)
ui = MainWindow()
sys.exit(app.exec_())


