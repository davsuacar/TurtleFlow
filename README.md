TURTLEFLOW
==========
Application to recognize and identify people based in a Convolutional Neural Network designed using TensorFlow.

To execute the application we have to follow the next steps

## Create images

```
python src/take_face_square.py /<path>/<to>/<project>/data/shape_predictor_68_face_landmarks.dat
```

## Load data to CSV

```
python src/create_convolutional_dataset.py /<path>/<to>/<project>/img
```

## Execute Video Streaming and People Identification application.

```
jupyter notebook
```

Execute notebook called notebooks/Convolutional Neural Network.ipynb


### Tech
TurtleFlow technologies requirements:

  - [OpenCV] - Open Source computer vision.
  - [Dlib] - Machine Learning library.
  - [TensorFlow] - TensorFlow is an Open Source Software Library for Machine Intelligence.
  - [Pandas] - Data manage python library.
  - [Scikit-learn] - Machine Learning library.
  - [Numpy] - Data manage library.

   [OpenCV]: <http://opencv.org/>
   [Dlib]: <http://dlib.net/>
   [TensorFlow]: <https://www.tensorflow.org/>
   [Pandas]: <http://pandas.pydata.org/>
   [Scikit-learn]: <http://scikit-learn.org/stable/>
   [Numpy]: <http://www.numpy.org/>

### Doc
Here you can find a pdf document with the full research process and project architecture.
[Full Documentation](docs/compiled/turtleflow.pdf)

### Video Tutorial
Video tutorial explaining the project structure.
[Video Tutorial](https://youtu.be/C1K6JdAoAjA)
### Demo
Real demo identifying two different people.
[Demo](https://youtu.be/RxMD2-Wopdw)