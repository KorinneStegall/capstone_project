# Detecting and Localizing Tumors in Brain MRI Scans
![image](https://github.com/KorinneStegall/capstone_project/assets/69525188/c6d83b4c-b9aa-491e-a3cd-a4209a0f2168)


## Research Question:
Can machine learning algorithms accurately detect and localize brain tumors in MRI scans, leading to early diagnosis and treatment?

## Context and Background:
Brain tumors pose a significant health risk; thus, early detection is crucial for patient outcomes. These tumors can vary widely in size, location, and malignancy, making accurate detection crucial for effective treatment planning. Traditional methods of tumor detection in MRI scans can be time-consuming and prone to human error. Radiologists must carefully review each scan, identifying potential abnormalities and assessing their characteristics to determine whether they represent tumors. These processes can be particularly challenging when tumors are small or located in complex regions of the brain. In this context, machine learning (ML) algorithms offer a promising solution to enhance brain tumor detection and localization in MRI scans. By leveraging vast amounts of imaging data, ML models can learn complex patterns and features indicative of brain tumors, enabling automated and efficient analysis of medical images. If we can automate and improve the accuracy of tumor detection, this could lead to faster diagnosis and treatment initiation. This not only improves patient outcomes but can also reduce healthcare costs associated with delayed diagnosis and treatment. The proposed project will aim to develop and evaluate ML-based algorithms for brain tumor detection and localization in MRI scans. 

## Proposed Solution:
The proposed solution involves implementing machine learning to improve the speed and accuracy of brain tumor detection and localization in MRI scans. The analytical method employed will primarily consist of training convolutional neural network (CNN) architectures, specifically the ResNet-50 residual neural network (ResNet) for tumor classification and a deep residual U-net (ResUNet) for tumor segmentation. These network architectures will be used on a dataset of MRI scans and their relative tumor location masks, to perform tumor detection and localization. The solution will begin by gathering data and preprocessing it. Once processed, data exploration will be conducted to gather an understanding of the dataset. Then, the data will be split into training, validation, and testing groups with their respective data generators to be passed through an amplified ResNet-50 architecture. After the best model is revealed, it will undergo an assessment to determine the accuracy, and classify whether the machine learning model is able to accurately detect a tumor within the image. The next step is to localize the tumor within the MRI. To do this, a deep residual U-net will be created and used. The model will predict tumor location at the pixel level which will then be visualized and compared to the original tumor mask. 

The solution addresses the research question by leveraging machine learning to detect and localize brain tumors in MRI scans. By training the models on the MRI data, the solution aims to automate the process of tumor detection, thereby reducing the time and effort required for manual interpretation by radiologists. The implementation of ML-based tumor detection algorithms aligns with the goal of facilitating early diagnosis and treatment planning for brain tumors, ultimately improving patient outcomes and healthcare efficiency. In summary, the solution encompasses data collection, data exploration, model training and evaluation stages, with the overarching goal of improving diagnostic accuracy. 

## Tools and Environments 
* Programming Language: Python
    * Python provides a wide range of libraries and frameworks for deep learning and image processing tasks.
*	Integrated Development Environment: Visual Studio Code
    *	Visual Studio Code offers a versatile environment for coding, debugging, and version control, facilitating the development of both regular Python files and Jupyter Notebook files.
*	Version Control: GitHub
    *	Github serves as a platform for version control, allowing for the management of code repositories, tracking changes made to project files and seamless integration with Visual Studio Code.
*	Libraries:
    *	NumPy: NumPy is used for numerical computations and array manipulation.
    *	Pandas: Pandas is utilized for data manipulation and analysis.
    *	Plotly, Matplotlib and Seaborn: These libraries are all used for data visualization.
    * OpenCV: OpenCV is used for image processing tasks.
    *	TensorFlow and Keras: TensorFlow with the Keras API serves as the primary deep learning framework for building and training convolutional neural networks and residual U-nets.
    *	Scikit-learn: Scikit-learn is utilized for the machine learning tasks beyond deep learning, such as data preprocessing, feature engineering and evaluation metrics calculation. 

## Statistical Significance
#### 1.	ResNet-50 for Image Classification:
  * Model Type: Supervised image classification using a residual neural network (ResNet-50).
  *	Algorithm: Implemented using TensorFlow/Keras
  *	Metrics:
    *	Accuracy: 0.98
    *	Precision: 0.98
    *	Recall: 0.99
    *	F1-score: 0.99
  *	Benchmark:
    *	Success Criterion: Accuracy exceeds 90%, precision, recall, and F1-score for tumor detection are above 0.85, and confusion matrix contains less than 10 false negatives and 10 false positives.
  *	Conclusion:
    *	The accuracy score of 0.98 exceeds the benchmark of 90%, indicating successful brain tumor detection in MRI scans.
    *	Precision, recall, and F1-score values exceed 0.85, further confirming the model’s effectiveness in accurately classifying tumor and non-tumor regions.
    *	The confusion matrix reveals only 2 false negatives and 6 false positives, which is within the defined benchmark, supporting the successful detection of brain tumors.
#### 2.	Residual U-Net for Image Segmentation:
  *	Model Type: Supervised image segmentation using Residual U-Net.
  *	Algorithm: Developed using TensorFlow/Keras.
  *	Metrics:
    *	Dice Similarity Coefficient: 81%
  *	Benchmark:
    *	Success Criterion: Dice similarity coefficient exceeds 75%, and qualitative assessment confirms coherence of predicted regions with ground truth.
  *	Conclusion:
    *	The dice similarity coefficient of 81% surpasses the benchmark of 75%, indicating successful brain tumor localization in MRI scans.
    *	Qualitative inspection also confirms the coherence and plausibility of predicted tumor regions, supporting the accurate localization of brain tumors.

**Overall Conclusion: Both ResNet-50 for image classification and Residual U-Net for image segmentation have demonstrated statistically significant performance in detecting and localizing brain tumors in MRI scans. The metrics used for evaluation consistently meet or exceed the predefined benchmarks, providing strong support for the hypothesis that machine learning techniques can effectively improve the speed and accuracy of brain tumor detection and localization.**
