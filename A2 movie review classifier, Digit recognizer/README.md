# Assignment 2

 - The pdf **ass2.pdf** contains the problem statement.    
 - The data folder contains the training data for both IMDB ratings and MNIST handwritten digits. 
 - Folders 1 and 2 contains scripts for running different models in problem statement. 
 - **report.pdf** contains a brief write up for the whole assignment. 
 - **links.json** contains the google drive ids for different models(pickle dumps and libsvm models). 
 - **jsonParser.py** downloads links in the links.json and puts them in respective directories. 
 - **run.sh** is used to predict test data for different models.

To run enter - <br>
```
./run.sh <Question_number> <model_number> <input_file_name> <output_file_name>
```

model_number for Q1

     1) NB model corresponding to part-a i.e. without stemming and stopword removal.

     2) NB model corresponding to part-d i.e. with stemming and stopword removal.

     3) NB model corresponding to part-e, i.e. your best model.

<br>  
  
model_number for Q2:

	1) Pegasos model corresponding to part b.

	2) Libsvm model (linear kernel) corresponding to part c.

	3) Best Libsvm model (rbf kernel) corresponding to part d.