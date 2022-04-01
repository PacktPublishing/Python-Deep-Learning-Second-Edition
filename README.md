## Python Deep Learning - Second Edition

<a href="https://www.packtpub.com/big-data-and-business-intelligence/python-deep-learning-second-edition?utm_source=github&utm_medium=repository&utm_campaign="><img src="https://www.packtpub.com/sites/default/files/B11133_cover.png" alt="Python Deep Learning" height="256px" align="right"></a>

This is the code repository for [Python Deep Learning - Second Edition](https://www.packtpub.com/product/python-deep-learning-second-edition/9781789348460), published by Packt.

**Exploring deep learning techniques and neural network architectures with PyTorch, Keras, and TensorFlow**

## About the Book
With the surge in artificial intelligence in applications catering to both business and consumer needs, deep learning is more important than ever for meeting current and future market demands. With Python Deep Learning Second Edition, you’ll explore deep learning, and learn how to put machine learning to use in your projects.

This book covers the following exciting features:
* Grasp the mathematical theory behind neural networks and deep learning processes
* Investigate and resolve computer vision challenges using convolutional networks and capsule networks
* Solve generative tasks using variational autoencoders and Generative Adversarial Networks
* Implement complex NLP tasks using recurrent networks (LSTM and GRU) and attention models
* Explore reinforcement learning and understand how agents behave in a complex environment
* Get up to date with applications of deep learning in autonomous vehicles

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/B07KQ29CQ3/) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
import torch

torch.manual_seed(1234)

hidden_units = 5

net = torch.nn.Sequential(
 torch.nn.Linear(4, hidden_units),
 torch.nn.ReLU(),
 torch.nn.Linear(hidden_units, 3)
)
```

**Following is what you need for this book:**
This book is for data science practitioners, machine learning engineers, and those interested in deep learning who have a basic foundation in machine learning and some Python programming experience. A background in mathematics and conceptual understanding of calculus and statistics will help you gain maximum benefit from this book.

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| All | Python 3.6, Anaconda 5.2, Jupyter Notebook | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/9781789348460_ColorImages.pdf).

### Related products
*  Python Deep Learning Projects[[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/python-deep-learning-projects?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/9781788997096)

*  Advanced Deep Learning with Keras[[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/advanced-deep-learning-keras?utm_source=github&utm_medium=repository&utm_campaign=9781788629416) [[Amazon]](https://www.amazon.com/dp/9781788629416)

## Get to Know the Authors
**Ivan Vasilev** started working on the first open source Java Deep Learning library with GPU support in 2013. The library was acquired by a German company, where he continued its development. He has also worked as a machine learning engineer and researcher in the area of medical image classification and segmentation with deep neural networks. Since 2017 he has focused on financial machine learning. He is working on a Python open source algorithmic trading library, which provides the infrastructure to experiment with different ML algorithms. The author holds an MSc degree in Artificial Intelligence from The University of Sofia, St. Kliment Ohridski.

**Daniel Slater** started programming at age 11, developing mods for the id Software game Quake. His obsession led him to become a developer working in the gaming industry on the hit computer game series Championship Manager. He then moved into finance, working on risk- and high-performance messaging systems. He now is a staff engineer working on big data at Skimlinks to understand online user behavior. He spends his spare time training AI to beat computer games. He talks at tech conferences about deep learning and reinforcement learning; and the name of his blog is Daniel Slater's blog. His work in this field has been cited by Google.

**Gianmario Spacagna** is a senior data scientist at Pirelli, processing sensors and telemetry data for the internet of things (IoT) and connected-vehicle applications. He works closely with tire mechanics, engineers, and business units to analyze and formulate hybrid, physics-driven, and data-driven automotive models. His main expertise is in building ML systems and end-to-end solutions for data products. He holds a master's degree in telematics from the Polytechnic of Turin, as well as one in software engineering of distributed systems from KTH, Stockholm. Prior to Pirelli, he worked in retail and business banking (Barclays), cyber security (Cisco), predictive marketing (AgilOne), and did some occasional freelancing.

**Peter Roelants** holds a master's in computer science with a specialization in AI from KU Leuven. He works on applying deep learning to a variety of problems, such as spectral imaging, speech recognition, text understanding, and document information extraction. He currently works at Onfido as a team leader for the data extraction research team, focusing on data extraction from official documents.

## Other books by the authors
[Python Deep Learning Projects](https://www.packtpub.com/big-data-and-business-intelligence/python-deep-learning-projects)

[Advanced Deep Learning with Keras](https://www.packtpub.com/big-data-and-business-intelligence/advanced-deep-learning-keras)


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.


