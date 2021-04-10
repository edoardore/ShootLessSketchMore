# ShootLessSketchMore
* Progetto di Data and Document Mining. 
* Realizzazione di una rete Siamese che permetta di eseguire il Few-shot Learning, nello specifico One-Shot learning con N-Way variabile.



## Come eseguire il codice
``` shell
python3 train.py --dataset tuberlin
python3 train.py --dataset miniquickdraw

python3 evaluation.py --dataset miniquickdraw
python3 evaluation.py --dataset tuberlin

````
### train.py
* Addestra una Siamese net che prende in input dal data loader due immagini con label 0 se di classi differenti o label 1 se della stessa classe. 
* Ottimizzatore Adam con learning rate 10^-3
* Binary Cross Entropy with Logits
* Test set 80%, Validation set 20%
* Salva il modello addestrato a fine esecuzione e mostra i grafici con l'andamento della loss

### evaluation.py
* Carica il modello addestrato precedentemente
* Il data loader fornisce una immagine e un set di N-Way immagini di cui una della stessa classe della prima. Inserendo nella rete addestrata l'immagine e le altre N-Way si trova quella con similarita' maggiore. 
* Viene mostrata la performance di predizione nel task N-Way One-Shot

## Datasets utilizzati
### TUBerlin 
* 161 classi train
* 41 validation
* 48 test
* Formato .png dimensione 84x84 8-bit

### MiniQuickDraw
* Sottoinsieme delle 50 milioni di immagini di Google QuickDraw! ricavabile tramite il codice presente in DataUtils
* 32.700 disegni in totale in 109 classi (300 per classe)
* 70 classi train
* 18 validation
* 21 test
* formato compresso array .npz numpy, unidimensionale di 748, modificate in 28x28 nel main.py

