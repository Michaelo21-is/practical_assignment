from ultralytics import YOLO
class Training:
    def train_model(self):
        # declare the model
        model = YOLO("yolo11n.pt")
        model.train(
            data="data.yaml",  # Path to your dataset config
            epochs=100,  # how many time to image will be used for training
            imgsz=640,  # Input image size
            batch=16,  # is how many image will be trained at a time is depends on your GPU memory and ram
            device="0",  # for using GPU set device to 0
            name = "size_arrow_model" # after the training is done the model will be saved in runs/detect/size_arrow_model/weights
        )
        # after runing thie train model it should open a new file with the training progress called runs/detect/size_arrow_model/weights
