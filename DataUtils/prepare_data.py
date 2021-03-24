import urllib.request

def download(nums=''):
    with open(nums+".txt", "r") as f:
        classes = f.readlines()
    classes = [c.replace('\n', '').replace(' ', '_') for c in classes]
    print(classes)
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        cls_url = c.replace('_', '%20')
        path = base+cls_url+'.npy'
        print(path)
        urllib.request.urlretrieve(path, './Data/'+c+'.npy')

download('train')
download('val')
download('test')
