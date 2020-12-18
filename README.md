# Onderzoeksrapport PI7 Data Science


## Inleiding
Voor de minor Data Science pi7 zijn 6 opdrachten gemaakt verdeelt in 3 fases. 
1. Fase 1: Multiple regression en Logistic regression
2. Fase 2: Random Forests en Neurale netwerken 
3. Fase 3: Support vector machines en Bayesian networks

### Dataset Beschrijving attributen en target
Voor de verschillende opdrachten in de drie fases, is gekozon om onderstaande dataset toe te passen. De dataset, "Dataset Carprices", is een set aan data die bestaat uit 26 attributen die iets zeggen over 205 type auto's.


|  1 |           Car_ID          |                                                 Unique id of each   observation (Interger)                                                |
|:--:|:-------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|
|  2 |         Symboling         | Its assigned insurance risk   rating, A value of +3 indicates that the auto is risky, -3 that it is   probably pretty safe.(Categorical)  |
|  3 |         carCompany        |                                                    Name of car company   (Categorical)                                                    |
|  4 |          fueltype         |                                              Car fuel type i.e gas or   diesel (Categorical)                                              |
|  5 |         aspiration        |                                                  Aspiration used in a car   (Categorical)                                                 |
|  6 |         doornumber        |                                                  Number of doors in a car   (Categorical)                                                 |
|  7 |          carbody          |                                                         body of car (Categorical)                                                         |
|  8 |         drivewheel        |                                                    type of drive wheel   (Categorical)                                                    |
|  9 |       enginelocation      |                                                   Location of car engine   (Categorical)                                                  |
| 10 |         wheelbase         |                                                         Weelbase of car (Numeric)                                                         |
| 11 |         carlength         |                                                          Length of car (Numeric)                                                          |
| 12 |          carwidth         |                                                           Width of car (Numeric)                                                          |
| 13 |         carheight         |                                                          height of car (Numeric)                                                          |
| 14 |         curbweight        |                                        The weight of a car without occupants or baggage. (Numeric)                                        |
| 15 |         enginetype        |                                                      Type of engine.   (Categorical)                                                      |
| 16 |       cylindernumber      |                                                 cylinder placed in the car   (Categorical)                                                |
| 17 |         enginesize        |                                                           Size of car (Numeric)                                                           |
| 18 |         fuelsystem        |                                                     Fuel system of car   (Categorical)                                                    |
| 19 |         boreratio         |                                                         Boreratio of car (Numeric)                                                        |
| 20 |           stroke          |                                               Stroke or volume inside the   engine (Numeric)                                              |
| 21 |      compressionratio     |                                                    compression ratio of car   (Numeric)                                                   |
| 22 |         horsepower        |                                                            Horsepower (Numeric)                                                           |
| 23 |          peakrpm          |                                                           car peak rpm (Numeric)                                                          |
| 24 |          citympg          |                                                         Mileage in city (Numeric)                                                         |
| 25 |         highwaympg        |                                                       Mileage on highway   (Numeric)                                                      |
| 26 | price(Target variable)    |                                                           Price of car (Numeric)                                                          |


|  symboling |  wheelbase  |  carlength  | carwidth   | carheight        | \            |
|:----------:|:-----------:|:-----------:|------------|------------------|--------------|
|    count   |  205.000000 |  205.000000 | 205.000000 | 205.000000       | 205.000000   |
|    mean    |   0.834146  |  98.756585  | 174.049268 | 65.907805        | 53.724878    |
|     std    |   1.245307  |   6.021776  | 12.337289  | 2.145204         | 2.443522     |
|     min    |  -2.000000  |  86.600000  | 141.100000 | 60.300000        | 47.800000    |
|     25%    |   0.000000  |  94.500000  | 166.300000 | 64.100000        | 52.000000    |
|     50%    |   1.000000  |  97.000000  | 173.200000 | 65.500000        | 54.100000    |
|     75%    |   2.000000  |  102.400000 | 183.100000 | 66.900000        | 55.500000    |
|     max    |   3.000000  |  120.900000 | 208.100000 | 72.300000        | 59.800000    |
| curbweight |  enginesize |  boreratio  | stroke     | compressionratio | \            |

| curbweight | enginesize  | boreratio   | stroke     | compressionratio | \            |
|------------|-------------|-------------|------------|------------------|--------------|
|    count   |  205.000000 |  205.000000 | 205.000000 | 205.000000       | 205.000000   |
|    mean    | 2555.565854 |  126.907317 | 3.329756   | 3.255415         | 10.142537    |
|     std    |  520.680204 |  41.642693  | 0.270844   | 0.313597         | 3.972040     |
|     min    | 1488.000000 |  61.000000  | 2.540000   | 2.070000         | 7.000000     |
|     25%    | 2145.000000 |  97.000000  | 3.150000   | 3.110000         | 8.600000     |
|     50%    | 2414.000000 |  120.000000 | 3.310000   | 3.290000         | 9.000000     |
|     75%    | 2935.000000 |  141.000000 | 3.580000   | 3.410000         | 9.400000     |
|     max    | 4066.000000 |  326.000000 | 3.940000   | 4.170000         | 23.000000    |

| horsepower |   peakrpm   |   citympg   | highwaympg | price            |              |
|:----------:|:-----------:|:-----------:|------------|------------------|--------------|
|    count   |  205.000000 |  205.000000 | 205.000000 | 205.000000       | 205.000000   |
|    mean    |  104.117073 | 5125.121951 | 25.219512  | 30.751220        | 13276.710571 |
|     std    |  39.544167  |  476.985643 | 6.542142   | 6.886443         | 7988.852332  |
|     min    |  48.000000  | 4150.000000 | 13.000000  | 16.000000        | 5118.000000  |
|     25%    |  70.000000  | 4800.000000 | 19.000000  | 25.000000        | 7788.000000  |
|     50%    |  95.000000  | 5200.000000 | 24.000000  | 30.000000        | 10295.000000 |
|     75%    |  116.000000 | 5500.000000 | 30.000000  | 34.000000        | 16503.000000 |
| max        | 288.000000  | 6600.000000 | 49.000000  | 54.000000        | 45400.000000 |

#### Data preperatie
##### Heatmap analyse
Voordat begonnen is aan normalisatie en standaardisatie is gekeken of dit daadwerkelijk nodig was. Er is begonnen door eerst een heatmap te genereren om te kijken of er waardes zijn met een hoge correlatie, zodat deze eruit gefilterd kunnen worden om een bias te voorkomen. Zie de heatmap. Te zien is hoe "highwaympg" en "citympg" een correlatie hebben van 0.97. In het model laten we deze dan ook weg.

![](HeatmapmlrCarPrices.png)

##### Standaardisatie
Als tweede stap is gekeken naar standaardisatie, zijn er kolommen die aangepast moeten worden om tot een beter resultaat te komen? Alle kolommen die geen nummerieke waarden bevatten zijn omgezet naar tabellen die wel nummerieke waarde bevatten, doormiddel van de "get_dummies()" functie van Pandas. Echter is een kolom, "CarName", niet efficiënt om op deze manier te standaardiseren. De kolom "CarName" heeft 205 waardes die bestaan uit unieke auto merken en types. Als deze kolom gestandaardiseerd word, resulteert dit in 205 nieuwe kolommen met 204 nullen en één 1. Dit leidde in de versie 1, tot een lage r2 score en een hoge rmse bij een test set van 30 procent. Om de kolom "CarName" te verbeteren is er gekozen om alle merken te categoriseren, zodoende werden alle type auto's van hetzelfde merk onder één naam gezet.

##### Normalisatie
Als derde stap is gekeken of normalisatie nodig zou zijn. In eerste instantie waren er geen kolommen die uitschietende waardes hadden. Alleen de target kolom had hoge waardes omdat er prijzen gehanteerd worden, maar aangezien dit de target kolom was zou zijn bij multiple linear regression zou deze sowieso niet meegenomen worden in de normalisatie. Om toch te kijken of normalisatie een positief effect zou hebben, in het geval dat er een andere target kolom gehanteerd zou worden, is deze toegepast. Echter waren de waardes na normalisatie zo abnormaal dat normalisatie niet is toegepast.

| Na Normalisatie bij multiple linear regression |                         |
|------------------------------------------------|-------------------------|
| rmse:                                          | 30267727458.953026      |
| r2:                                            | -2.7177251408947733e+22 |

## Fase 1
### Multiple linear regression
Bij deze opdracht is de te voorspellen waarde "price". Dit is de prijs van een type auto.

#### Code
~~~
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:02:10 2020

@author: Rutger
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import math



df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

#Heatmap
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.heatmap(df.corr().round(2),square=True,cmap="RdYlGn",annot=True)


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
print(df.describe())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

print(df.info())

#Normalisatie (n.v.t.)
#df = (df-df.min())/(df.max()-df.min())
      
y = df.price
x = df.drop('price', 1)



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

lg = LinearRegression()

lg.fit(x_train, y_train)

y_pred = lg.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rtwo = r2_score(y_test, y_pred)
print('\nrmse: ',math.sqrt(mse), '\nr2: ', rtwo)

plt.scatter(y_test, y_pred)
    
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")

plt.show()
~~~

#### Output
- Random_state: 7
- Data test set: 30 procent
- Data train set: 70 procent

- rmse:  3031.144264619707 
- r2:  0.8320279333635224

![](LinearverbandPredyTesty.png)

#### Conclusie
Het uiteindelijke resultaat laat zien dat er een rmse is van ongeveer 3000 euro met r2 score van ongeveer 83 procent. Dit laat zien dat er een goed verband is tussen alle attributen en dat er een vrij goede voorspelling gedaan kan worden over de dataset.

#### Feedback
In de feedback momenten, werd er vooral aangekaart dat we meer aandacht moesten bieden aan het voorbereiden van de data. De geschreven code was prima, maar er moest meer gefocussed worden op het analyseren van de verbanden in de dataset om te kijken hoe de uitkomst het beste zou worden.
Zo is er aangeraden om de kolom "CarName" op te splitsen in merknamen i.p.v. type auto's. Ook werd als tip gegeven om aan de hand van de heatmap te kijken welke attributen een goede correlatie hadden en dus een biassed uitkomst konden leveren.

### Logistic regression
Voor de opdracht van logistic regression is er een dataset toegepast over de kans op een hartaanval; https://www.kaggle.com/nareshbhat/health-care-data-set-on-heart-attack-possibility.
Deze dataset is gevonden op de website van kaggle, dit leek betrouwbaar aangezien het studieboek, Data Science Design Manual, deze bron vaker gebruikt om theorieën toe te lichten.

#### Beschrijving
Deze dataset bevat van ongeveer 300 patienten, 13 attributen die kunnen inschatten of een patient een mindere- of hogere kans heeft op een hartaanval. Deze dataset is passend omdat de kans op een hartaanval binair wordt gepresenteerd met een 1 of 0.

| Attribuut     | Toelichting                                                                     |
| ------------- |:-------------------------------------------------------------------------------:|
| age           | Leeftijd van patiënt                                                            |
| sex           | Geslacht van patiënt                                                            |
| cp            | Borstkast pijn (0 - 3)                                                          |
| trest bps     | Bloeddruk bij rust                                                              |
| chol          | serumcholestoraal in mg / dl                                                    |
| fbs           | Bloedsuiker nuchter > 120 mg / dl                                               |
| restecg       | Elektrocardiografische resultaten bij rust (0 - 2)                              |
| thalach       | Maximale hartslag bereikt                                                       |
| exang         | Oefening geïnduceerde angina                                                    |
| oldpeak       | Oldpeak = ST-depressie veroorzaakt door inspanning ten opzichte van rust        |    
| slope         | De helling van het ST-segment met piekoefening                                  |
| ca            | Aantal grote bloed vaten (0 - 3) gekleurd door flourosopie                      |
| thal          | Thalassemia: 0 = normaal; 1 = vast; 2 = omkeerbaar defect                       |
| target        | Target: 0 = minder kans op een hartaanval; 1 = meer kans op een hartaanval      |

#### Data opzet
| patient | age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | target |
|---------|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|--------|
| 1       | 63  | 1   | 3  | 145      | 233  | 1   | 0       | 150     | 0     | 23      | 0     | 0  | 1    | 1      |
| 2       | 37  | 1   | 2  | 130      | 250  | 0   | 1       | 187     | 0     | 35      | 0     | 0  | 2    | 1      |
| 3       | 41  | 0   | 1  | 130      | 204  | 0   | 0       | 172     | 0     | 14      | 2     | 0  | 2    | 1      |
| 4       | 56  | 1   | 1  | 120      | 236  | 0   | 1       | 178     | 0     | 8       | 2     | 0  | 2    | 1      |
| 5       | 57  | 0   | 0  | 120      | 354  | 0   | 1       | 163     | 1     | 6       | 2     | 0  | 2    | 1      |
|..       |..   |..   |..  |..        |..    |..   |..       |..       |..     |..       |..     |..  |..    |..      |

#### Code
~~~~
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:16:21 2020

@author: Rutger
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Dataset Heart.csv')
df.head()

LogReg = LogisticRegression()

y = df['target']
x = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]

LogReg.fit(x, y)

y_pred = LogReg.predict(x)

score = LogReg.score(x, y)

i = 0
while i < len(y_pred):
    print('Person ', i, ': ', y_pred[i])
    i += 1

print('ROC score: ', score)

sns.pairplot(df)
~~~~


#### Output
##### Code  output
~~~~
Person  0 :  1
Person  1 :  1
Person  2 :  1
Person  3 :  1
Person  4 :  1
Person  5 :  1
..
ROC score:  0.8514851485148515
~~~~
##### Visualisatie output pairplot


#### Conclusie
Op basis van de 13 attributen uit de dataset heeft dit model met een ROC score van ongeveer 85 procent betrouwbaarheid een voorspelling kunnen maken op de kans van een hartaanval.

#### Feedback

## Fase 2
### Random forests

### Neurale netwerken 

## Fase 3
### Support vector machines

### Bayesian networks

## Auteurs
- Rutger de Groen https://rutgerfrans.com/
- Maroche Delnoy https://www.linkedin.com/in/maroche-delnoy-788ab9195/


 


