# ShootLessSketchMore
* Progetto di Data and Document Mining. 
* Siamese Network per Few-shot Learning, nello specifico One-Shot learning con N-Way modificabile
![siamesenet](https://github.com/edoardore/ShootLessSketchMore/blob/main/Siamese.PNG)



## Come eseguire il codice
* Addestramento della Siamese net:
``` shell
python3 train.py --dataset tuberlin
python3 train.py --dataset miniquickdraw
```
* Valutazione dell'accuratezza di predizione N-Way:
```shell
python3 evaluation.py

````
### train.py
* Addestra una Siamese Net che prende in input dal data loader due immagini con label 0 se di classi differenti o label 1 viceversa
* Rete per classificazione binaria
* Due immagini in input a due reti convoluzionali che condividono i pesi
* Differenza in valore assoluto ---> layer fully connected ---> sigmoide che indica la similarit√† (rete in model.py)
* Ottimizzatore Adam con learning rate 10^-3
* Binary Cross Entropy Loss with Logits
* Epoche di addestramento modificabili in config.py
* A termine esecuzione salva il modello addestrato e mostra i grafici con l'andamento della loss (Esempio seguente del train in 15 epoche con dataset Mini Quick Draw)

![loss plot](https://github.com/edoardore/ShootLessSketchMore/blob/main/Schermata%20da%202021-04-29%2009-09-43.png)


### evaluation.py
* Carica il modello addestrato precedentemente (file .pt)
* Il data loader fornisce una immagine e un set di N-Way immagini di cui una della stessa classe della prima 
* Inserendo nella rete addestrata l'immagine principale e una immagine del set alla volta si trova quella con similarit√† maggiore
* Viene mostrata la performance di predizione nel task N-Way One-Shot mediata su 12 iterazioni

<img src="https://github.com/edoardore/ShootLessSketchMore/blob/main/fewShotExample.PNG" width="400">

* L'immagine principale a sinistra viene posta in input alla Siamese Net assieme ad una immagine alla volta tra quelle di destra
* In output dalla rete si ha il valore di predizione, si decide di assegnare l'appartenenza alla stessa classe dell'immagine di sinistra con una di quelle a destra per cui il valore di predizione in output dalla Siamese Net √® il maggiore tra tutti
```python
for i, testImg in enumerate(imgSets):
            output = model(mainImg, testImg)
            if output > predVal:
                pred = i
                predVal = output
        if pred == label:
            correct += 1
```
* Output finale: performance di predizione per i due modelli in configurazione 2-Way, 5-Way, 10-Way
![output](https://github.com/edoardore/ShootLessSketchMore/blob/main/Schermata%20da%202021-05-30%2013-28-23.png)
## Datasets utilizzati
### TUBerlin 
* 161 classi train
* 41 validation
* 48 test
* Formato .png dimensione 84x84 8-bit
* File di supporto: ./TUBerlin, dataset_n_way.py, train_val_splitter.py, dataset.py

### MiniQuickDraw
* Sottoinsieme delle 50 milioni di immagini di Google QuickDraw! ricavabile tramite il codice presente in DataUtils
* 32.700 disegni in totale in 109 classi (300 per classe)
* 70 classi train
* 18 validation
* 21 test
* formato compresso array .npz numpy, unidimensionale di 748, modificate in 28x28 nel main.py
* File di supporto ./DataUtils, dataset_n_way.py

