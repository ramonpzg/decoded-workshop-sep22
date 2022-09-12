
import urllib.request, os

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'
path = os.path.join('data', 'raw')
filename = 'SeoulBikeData.csv'

if not os.path.exists(path): os.makedirs(path)
        
urllib.request.urlretrieve(url, os.path.join(path, filename))
print("File Downloaded Successfully!")
