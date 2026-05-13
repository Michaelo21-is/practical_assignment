run model:
    During runtime, I run the image through the YOLO model at least twice. In each run,
    I check how many bounding boxes were created and whether the boxes overlap between the runs. 
    This allows me to conclude that the result is more stable, and therefore the analysis is likely more accurate and reliable.
    If it does not find a matching stable result, I print "Model result is not stable".
    However, the logic is flexible. It does not have to stop there. 
    I can choose to return the first YOLO result as a fallback, or I can run the function again until a stable match is found.
train model: 
    I train the model for 100 epochs. The goal is to give the model enough time to learn the arrow patterns, but this value may need fine-tuning because I have not tested yet if it causes overfitting or underfitting.
    I use YOLO11n because it is a lightweight YOLO model, which makes it suitable for faster training and inference, especially if the final solution needs to run on limited hardware.
    I set the image size to 640 because it is a common input size for YOLO models and gives a good balance between accuracy and performance.
    I set the batch size to 16, but this can be changed depending on the available GPU memory and RAM.
    I decided to train the model on GPU because it gives better performance and is much more time-efficient than CPU training. However, this depends on the hardware available during training.
    After training, I would evaluate the model using metrics such as precision, recall, and mAP, and also test it visually on new technical drawings that were not used during training.
Why I chose the YOLO algorithm:
    I chose the YOLO algorithm because I needed a fast object detection solution that can return results in real time. YOLO is suitable for this task because it detects objects and directly returns bounding boxes and coordinates, which is exactly what was required in the assignment.
    Another reason I chose YOLO is that it can run very fast while still providing good detection results. By using a lightweight version of YOLO, such as YOLO11n, the model can also be suitable for limited hardware, including machines with around 4GB of RAM, depending on the image size, model size, and runtime environment.