from face_keras import predict_image
import sys

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 3:
        print("nope")
        exit(1)
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    predicted = predict_image(model_path, image_path)
    print(predicted)
