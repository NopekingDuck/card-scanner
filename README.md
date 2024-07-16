<h1 align="center">Card Scanner</h1>
<p>Card Scanner is a project to extract text from a set of cards belonging to a board game. The cards are provided in a pdf format and the extracted text of each card added to a csv.</p>
<br>
<h2>Features</h2> 
<ul>
  <li>Convert pdf pages to jpegs</li>
  <li>Crop white space from around the main image of a jpeg</li>
  <li>Slice jpegs</li>
  <li>Run pytesseract on a card image to get the text</li>
</ul>
<h2>Screenshot</h2>

![screenshot of a cropped page of cards](https://github.com/NopekingDuck/card-scanner/assets/124461571/d788ef05-61c1-496a-adf7-4cb93cce262c)

![screenshot of an individual card](https://github.com/NopekingDuck/card-scanner/assets/124461571/d21b6ed5-a6ba-40b3-9dac-5bc55fcf265d)
![screenshot of an individual card](https://github.com/NopekingDuck/card-scanner/assets/124461571/f90fd3d0-d3db-4f62-a808-b15d23c27544)

<h2>Getting Started</h2>
<ol>
  <li>Clone the repository</li>
  <li>pip install -r requirements.txt</li>
  <li>run main.py</li>
</ol>


<h2>Future Improvements</h2>
<ul>
  <li>Smarter edge detecton on text areas to more precisely find text</li>
</ul>


