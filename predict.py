import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import argparse
import sys


def process_image(image):
    return tf.image.resize(np.squeeze(image), (224, 224)) / 255.0

def predict(image_path, model, top_k):
    image = process_image(np.asarray(Image.open(image_path)))
    pred = model.predict(np.expand_dims(image, axis=0))
    top_values, top_indices = tf.math.top_k(pred, top_k)
    top_classes = [str(value+1) for value in top_indices.numpy()[0]]
    return top_values.numpy()[0], top_classes


if __name__ == "__main__":
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--top_k', type=int)
    my_parser.add_argument('--category_names', type=str)
    args = my_parser.parse_args(sys.argv[3:])
    
    top_k = 5
    if args.top_k is not None:
        top_k = args.top_k
    
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)
    
    if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        print([class_names[i] for i in classes])


