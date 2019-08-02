# DeepOCTNet

Download OCT dataset from the following two links
1.  https://data.mendeley.com/datasets/rscbjbr9sj/2
2.  https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1

Merge the OCT images according to disease class.

Now in the directory '.../Datasets/OCT2017_NOR' create 'train', 'validation' and 'test' folder. 

In each newly created folder create four folders by named 'AMD', 'CNV', 'DME', 'NORMAL'.

Keep 242 AMD OCTs in the AMD folder,  242 CNV OCTs in the CNV folder, 242 DME OCTs in the DME folder, and 242 Normal
OCTs in the NORMAL folder in the test folder. Put different 242 OCT images of each kind, total of 984 images, in validation 
folder accordingly. And keep all the remailing OCTs in the training folder accordingly. 

In dataset 1, you will find AMD in the Drusen folder. 

Four class classification using Inception v3:
Now run the save_best_model_classify_inception_V3.py 
it will save weights in the current directory. 
Now run load_best_model_classify_inception_V3.py from the same directory.

Four class classification using VGG16:
Now run the save_best_model_classify_VGG16.py 
it will save weights in the current directory. 
Now run load_best_model_classify_VGG16.py from the same directory.


Four class classification using VGG19:
Now run the save_best_model_classify_VGG19.py 
it will save weights in the current directory. 
Now run load_best_model_classify_VGG19.py from the same directory.

For obtaining results for four class classification on a single dataset code will remain same as described above and put OCT images from a single dataset instead of usning combined dataset. 

For three class OCT classification run code from 'Three_class' folder.
For two class OCT classification run code from 'Two_class' folder.
