# ShootLessSketchMore
DDM Project


## Come eseguire il codice
``` shell
python3 main.py --dataset tuberlin

python3 main.py --dataset miniquickdraw
````

## Datasets
### TUBerlin 
161 classi train
41 validation
48 test
Formato .png dimensione originale 1111x1111 8-bit

### MiniQuickDraw
Sottoinsieme delle 50 milioni di immagini di Google QuickDraw! ricavabile tramite il codice presente in DataUtils
32.700 disegni in totale in 109 classi (300 per classe)
70 classi train
18 validation
21 test
formato compresso array .npz numpy, unidimensionale di 748, modificate in 28x28 nel main.py

