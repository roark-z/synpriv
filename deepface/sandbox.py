from deepface import DeepFace

path = "../datasets/small/images/test_0001.jpg"

obj = DeepFace.analyze(img_path = path, 
        actions = ['age', 'emotion']
)

print(obj)