"""
1.  Értelmezd az adatokat!!!
    A feladat megoldásához használd a NJ transit + Amtrack csv-t a moodle-ból.
    A NJ-60k az a megoldott. Azt fogom használni a modellek teszteléséhez, illetve össze tudod hasonlítani az eredményedet.    
2. Írj egy osztályt a következő feladatokra:  
     2.1 Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     2.2 Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba 
     2.3 Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el  
     2.4 Dobjuk el a from és a to oszlopokat, illetve azokat a sorokat ahol van nan és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el  
     2.5 A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el   
     2.6 Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el  
         4:00-7:59 -- early_morning  
         8:00-11:59 -- morning  
         12:00-15:59 -- afternoon  
         16:00-19:59 -- evening  
         20:00-23:59 -- night  
         0:00-3:59 -- late_night  
    2.7 A késéseket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0min <= x < 5min   --> 0  
         5min <= x          --> 1  
    2.8 Dobd el a felesleges oszlopokat 'train_id' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    2.9 Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'. 
    2.10 Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'
3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re. 
    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik (0), ha 5p <= x --> késik (1).
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.
    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.
    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp 
4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el. 
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!
HAZI-
    HAZI06-
        -NJCleaner.py
        -HAZI06.py
##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################
"""

import pandas as pd

from src.DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

col_name = ['stop_sequence', 'from_id', 'to_id', 'status', 'line','type','day','part_of_the_day', 'delay']
data = pd.read_csv('data\\NJ.csv', skiprows=1, header=None, names=col_name)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split=100, max_depth=11)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))

# Sample = [90,100,110,120,130]
# Depth = [9,10,11,12,13]

# for i in range(0,len(Depth)):
#     for j in range(0,len(Sample)):
        
#         classifier = DecisionTreeClassifier(min_samples_split=Sample[j], max_depth=Depth[i])
#         classifier.fit(X_train, Y_train)
#         Y_pred = classifier.predict(X_test)
#         accuracy = accuracy_score(Y_test, Y_pred)        
#         print(f'Split: {Sample[j]}, Depth: {Depth[i]} -> Accuracy: {accuracy}')
    

# Ha a samples_split túl alacsony, könnyen vezethet overfittinghez. Ha a max_depth túl magas, akkor is ugyan így overfitting léphet fel.
# Természetesen fordított esetben ugyan ez könnyen elmondható az underfitting-re is.
# Tesztelés során egy for ciklusba ágyazva "gyorsítottam" a tesztelést.

# A fentebbiekből kifolyólag találtam egy formulát amely az adott adat mennyiségétől függően határozza meg a min splitet és a depthet egy bizonyos range-ben.
# Ezzel a módszerrel igazából kapásból 78%-os volt a pontosság. Best case --->>> Split: 110, Depth: 11 -> Accuracy: 80.275% volt.

# Tests:
# Split: 30, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 30, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 50, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 30, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 10, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 70, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 10, Depth: 3 -> Accuracy: 0.7839166666666667
# Split: 90, Depth: 2 -> Accuracy: 0.7823333333333333
# Split: 50, Depth: 3 -> Accuracy: 0.7839166666666667
# Split: 30, Depth: 3 -> Accuracy: 0.7839166666666667
# Split: 90, Depth: 3 -> Accuracy: 0.7839166666666667
# Split: 70, Depth: 3 -> Accuracy: 0.7839166666666667
# Split: 30, Depth: 5 -> Accuracy: 0.7885833333333333
# Split: 10, Depth: 5 -> Accuracy: 0.7885833333333333
# Split: 70, Depth: 5 -> Accuracy: 0.7885833333333333
# Split: 50, Depth: 5 -> Accuracy: 0.7885833333333333
# Split: 10, Depth: 7 -> Accuracy: 0.7935
# Split: 90, Depth: 5 -> Accuracy: 0.7885833333333333
# Split: 50, Depth: 7 -> Accuracy: 0.7935833333333333
# Split: 30, Depth: 7 -> Accuracy: 0.7936666666666666
# Split: 90, Depth: 7 -> Accuracy: 0.7940833333333334
# Split: 70, Depth: 7 -> Accuracy: 0.7936666666666666
# Split: 30, Depth: 8 -> Accuracy: 0.7955
# Split: 10, Depth: 8 -> Accuracy: 0.7955833333333333
# Split: 70, Depth: 8 -> Accuracy: 0.7955
# Split: 50, Depth: 8 -> Accuracy: 0.7954166666666667
# Split: 90, Depth: 9 -> Accuracy: 0.79825
# Split: 90, Depth: 8 -> Accuracy: 0.7965
# Split: 110,Ddepth: 9 -> Accuracy: 0.79825
# Split: 100, Depth: 9 -> Accuracy: 0.7986666666666666
# Split: 130, Depth: 9 -> Accuracy: 0.7975
# Split: 120, Depth: 9 -> Accuracy: 0.7975
# Split: 100, Depth: 10 -> Accuracy: 0.80225
# Split: 90, Depth: 10 -> Accuracy: 0.8018333333333333
# Split: 110, Depth: 10 -> Accuracy: 0.8014166666666667
# Split: 120, Depth: 10 -> Accuracy: 0.8006666666666666
# Split: 90, Depth: 11 -> Accuracy: 0.803
# Split: 130, Depth: 10 -> Accuracy: 0.7993333333333333
# Split: 130,Depth: 11 -> Accuracy: 0.801
# Split: 90, Depth: 12 -> Accuracy: 0.80175
# Split: 100,Depth: 12 -> Accuracy: 0.8025
# Split: 110,Depth: 12 -> Accuracy: 0.8016666666666666
# Split: 120,Depth: 12 -> Accuracy: 0.80125
# Split: 130,Depth: 12 -> Accuracy: 0.8005833333333333
# Split: 90, Depth: 13 -> Accuracy: 0.79975
# Split: 100,Depth: 13 -> Accuracy: 0.8009166666666667
# Split: 110,Depth: 13 -> Accuracy: 0.7996666666666666
# Split: 130,Depth: 13 -> Accuracy: 0.7985833333333333
# Split: 120,Depth: 13 -> Accuracy: 0.7990833333333334