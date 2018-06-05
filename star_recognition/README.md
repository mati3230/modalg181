# Manual to download images from google images

* Go to google image search
* Enter the name of the object
* Open developer console
* Scroll down till enough images are found

* Paste the code 1) and press enter
* Paste the code 2) and press enter
* Paste the code 3) and press enter

* A .txt file with the urls will be saved

* Place them into a directory ../datasets/stars_from_google_images

* Choose the name {0}_{1}_urls.txt
{0}: first name of the star in lower case
{1}: last name of the star in lower case

## Code 1), 2), 3)
// 1)
// pull down jquery into the JavaScript console
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

// 2)
// grab the URLs
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });

// 3)
// write the URls to file (one per line)
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();