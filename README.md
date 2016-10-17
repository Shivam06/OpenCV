# OpenCV


<h2 style="color: blue"><u>PROJECTS :</u></h2>

<ol>
<li style="color: green"><b><u>Hand detection and Fingers counting</u></b></li>
<h4><b>Contents :</b></h4>
<ul>
<li><b>hand.py</b> - Script for hand detection and fingers counting in a video.</li>
<li><b>1.png, 2.png, 3.png, 4.png, 5.png</b> are screenshots taken from the webcam video</li>
</ul>
<br/>
<li style="color: green"><b><u>Sudoku Solver</u></b></li>
<h4><b>Contents :</b></h4>
<ul>
<li><b>imageProcessing.py</b> Script to segement the image of sudoku square <b>"imag.jpg"</b> from original image <b>"sudoku.jpg"</b></li>
<li><b>imageToData.py</b> Script to make user mark correct labels (1-9) for numbers represented in image <b>"numbers2.png"</b> and <b>"numbers.png"</b> and storing the feature vectors of all the numbers (feature matrix) shown in the image in 2 text files, one for training set - "inputs_train.data" and other for test set - "inputs_test.data"</li>
<li><b>KNearestNeighbour.py</b> - Module containing KNN class which has been used in <b>main.py</b> to check the accuracy over test set after training the knn model over training data.</li>
<li><b>main.py</b> - As mentioned above, just to examine performance of KNN model (10 nearest neighbor) over test set. And as expected it gave 100% accuracy because both training and test set have same font.</li>
<li><b>sudoku.py</b> - Main part of project to recognize the numbers, but here accuracy came out to be 90% because numbers in actual sudoku have different font in comparison to the font in training set. And also KNN having many disadvantages of its own, expectedly did not perform well. <b>NOT COMPLETED YET</b>.</li>
</ul>
</ol>

<br/>
<h2 style="color: red">TODO - Sudoku Solver</h2>

<ul>
<li>Improve Accuracy for recognition either by training more and different fonts of training data or use different algorithm. Will try <b>Artificial Neural Network (ANN)</b> model trained over MNIST dataset (hand written digits).</li>
<li>Debug and complete <b>sudoku.py</b></li>
<li>Solve sudoku using <b>backtracking algorithm</b>.</li>
<li>Document all the codes.</li>
</ul>


