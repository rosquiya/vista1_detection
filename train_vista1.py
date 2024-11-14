if __name__ == '__main__':
    import torch
    from ultralytics import YOLO


    model = YOLO('yolov8n.yaml')

    results = model.train(
        data='data.yaml',   
        epochs=1000,         
        patience=50      
    )
