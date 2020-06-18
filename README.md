# [Inklusive](http://inklusive.xyz/): Think about your Ink
![alt text](./image/website_front.jpg)

*Recommend tattoo artists based user-submitted image and a curated list of inclusive Montreal tattoo studios*

### What is Inklusive

Inklusive works to match users with the right artists from Montreal's most loved studios. Further, it emphasizes the importance of studio atmosphere. It can often be intimidating walking into a tattoo studio, particularly for women, gender fluid and non-binary people. Being welcomed and treated with respect by the artists and studio is crucial to having a good tattoo experience. In order to achieve this, 1.2K women/female-identifying/gender fluid people were polled on facebook as to their most positive and inclusive experiences getting tattooed in Montreal. This resulted in 12 studios, all of which have 4.8+ stars on Google Reviews!

### The studios:

* [Paradise](https://www.paradisemtl.com)
* [Minuit Dix](http://minuitdixtattoo.com)
* [Tattoo 1974](https://en.tattoo1974.com)
* [Bloodline Tattoo](https://bloodlinemtl.com)
* [DFA Tattoos](https://www.dfatattoos.com)
* [Studio Artease](http://www.studioartease.com/en/)
* [Loveless](https://www.lovelesstattoo.ca)
* [Tattoo Abyss](https://tattooabyss.com)
* [Tatouage Royal](https://tatouageroyal.com)
* [Le Chalet](https://lechalettattooshop.com)
* [Saving Grace Tattoo](https://www.savinggracetattoo.com)
* [Black Rose Tattoo](https://www.theblackrosetattoo.ca)

### The images:

By scraping the Instagram pages for each of the 12 studios, I put together thousands of images. Along with the images the artist's instagram handle was also saved (only images with a single @mention were saved). From there, I had to further classify if an image was of a tattoo or not. 

![alt text](./image/demo_tatclass.jpg)

In order to do this I created two custom datasets: one of tattoo images, and the other non-tattoo images (N=5254 for each). Then I built a Convolutional Neural Network with binary encoding to distinguish between the two datasets. It achieved a validation accuracy of 94%! 

Now that I had the CNN model to classify the images, I ran all of the instagram images through, storing the information of those classified as a tattoo in a Pandas dataframe. 

### Image similarity:

Next is the essence of the product: train a model to extract key features from the images, and then train a nearest neighbours model on those features. This way, when a user uploads an image, it is similarly passed through both models, with an output of *k* similar images. 






