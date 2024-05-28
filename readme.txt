data: per conservare i dati del dataset MOT17. in una cartella fuori da quella che carichiamo su github
dentro la cartella data seguire il seguente percorso

data/gt/mot_challenge/           Che contiene i dati di ground truth.
data/trackers/mot_challenge/     Che contiene i risultati del tracker.

    data_path = 'data/videos'
    gt_path = 'data/gt/mot_challenge'

models: per script legati ai modelli, come DETR e eventuali reti aggiuntive.
utils: per funzioni di utilità, come le funzioni per il calcolo delle metriche.
results: per salvare output intermedi e risultati finali.

models/detr_model.py per gestire il caricamento e l'utilizzo del modello DETR
utils/track_management.py,  funzioni per gestire le tracce





____________________________________________________________________________________________________________________
dico in teoria perchè mi dà alcuni errori nell'esecuzione
In teoria eseguendo trackeval_setup.py, valuterà le prestazioni del sistema di tracciamento rispetto al dataset di ground truth
I risultati ti forniranno una serie di metriche, come HOTA, CLEAR MOT, Identity e altre, che ti aiuteranno a comprendere 
come il tuo tracker sta performando e dove potrebbe essere migliorato.

____________________________________________________________________________________________________________________

INSTALLAZIONE DA TERMINALE

pip install torch torchvision
pip install scipy

git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval
pip install -r requirements.txt
pip install .