# Brain-Tumor-GoogLeNet

Inception V3, also known as GoogLeNet, stands as a formidable architecture in the realm of convolutional neural networks (CNNs), revered for its intricate design and remarkable performance in various image classification tasks. Leveraging its deep layers and sophisticated inception modules, Inception V3 has found particular utility in medical imaging, notably for the critical task of brain tumor classification. In the medical domain, precise and rapid diagnosis is paramount, and Inception V3 rises to the occasion with its ability to discern between different types of brain tumors with a high degree of accuracy.

In the context of brain tumor classification, Inception V3 is trained on a dataset comprising images of glioma, meningioma, pituitary tumors, as well as images depicting the absence of tumors, referred to as 'notumor'. This diverse dataset is instrumental in enabling the model to learn the intricate features and patterns characteristic of each tumor type, thereby facilitating its ability to make informed classifications.

The training process for Inception V3 involves the utilization of sophisticated data augmentation techniques, a critical component in mitigating overfitting and enhancing the model's generalization capabilities. The ImageDataGenerator class, a staple tool in the arsenal of deep learning practitioners, is employed to preprocess the training and testing datasets. Through rescaling the pixel values of the images to a range between 0 and 1, normalization is achieved, which aids in stabilizing the training process and expedites convergence.

Furthermore, employing techniques such as random rotation, shifting, and flipping, augmentation injects variability into the training data, thereby enriching the model's exposure to diverse perspectives of the tumor images. This augmentation strategy is pivotal in enhancing the model's robustness and its ability to generalize well to unseen data, a crucial requirement in real-world medical applications.

The training and validation datasets are organized into directories, facilitating seamless integration with the flow_from_directory method. This method, an integral component of the Keras deep learning framework, streamlines the process of feeding data into the model during training and validation. By specifying parameters such as target size, batch size, and class mode, the datasets are efficiently prepared for consumption by Inception V3, ensuring optimal training and evaluation performance.

![Model](https://github.com/GOVINDFROMINDIA/Brain-Tumor-GoogLeNet/assets/79012314/a53ba215-493e-49c7-80b1-e3c107b27638)

![ROC](https://github.com/GOVINDFROMINDIA/Brain-Tumor-GoogLeNet/assets/79012314/4ba11392-c698-4f09-b6ed-a90d5e49e16f)
