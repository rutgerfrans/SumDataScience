# Onderzoeksrapport PI7 Data Science

- [Inleiding](#Inleiding)
- [Data selectie](#Dataselectie)
    - [Data beschrijving](#Datasetbeschrijving)
    - [Data preperatie](#Datapreperatie)
- [Fase 1](#Fase1)
    - [Multiple Linear Regression](#mlr)
    - [Logistic Regression](#lr)
- [Fase 2](#Fase2)
    - [Random Forests](#rf)
    - [Neural Networks](#nn)
- [Fase 3](#Fase3)
    - [Support Vector Machines](#svm)
    - [Bayesian Networks](#bn)
- [Conclusie](#AlgeheleConclusie)
- [Auteurs](#Auteurs)




## <a name="Inleiding"></a> Inleiding
- *Mooie inleiding schrijven met context*

Voor de minor Data Science pi7 zijn 6 opdrachten gemaakt verdeelt in 3 fases. 
1. Fase 1: Multiple regression en Logistic regression
2. Fase 2: Random Forests en Neurale netwerken 
3. Fase 3: Support vector machines en Bayesian networks

### <a name="Dataselectie"></a> Data selectie
#### Onderbouwing
- *Wat voor dataset hebben we gekozen en waarom?*
#### <a name="Datasetbeschrijving"></a> Dataset Beschrijving attributen en target
Voor de verschillende opdrachten in de drie fases, is gekozen om onderstaande dataset toe te passen. De dataset, "Dataset Carprices", is een set aan data die bestaat uit 26 attributen die iets zeggen over 205 type auto's.


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


| feature          | count    | mean         | std         | min     | 25%     | 50%      | 75%          | max         |
|------------------|----------|--------------|-------------|---------|---------|----------|--------------|-------------|
| symboling        | 205.0    | 0.834146     | 1.245307    | -2.00   | 0.00    | 1.00     | 2.00         | 3.00        |
| wheelbase        | 205.0    | 98.756585    | 6.021776    | 86.60   | 94.50   | 97.00    | 102.40       | 120.90      |
| carlength        | 205.0    | 174.049268   | 12.337289   | 141.10  | 166.30  | 173.20   | 183.10       | 208.10      |
| carwidth         | 205.0    | 65.907805    | 2.145204    | 60.30   | 64.10   | 65.50    | 66.90        | 72.30       |
| carheight        | 205.0    | 53.724878    | 2.443522    | 47.80   | 52.00   | 54.10    | 55.50        | 59.80       |
| curbweight       | 205.0    | 2555.565854  | 520.680204  | 1488.00 | 2145.00 | 2414.00  | 2935.00      | 4066.00     |
| enginesize       | 205.0    | 126.907317   | 41.642693   | 61.00   | 97.00   | 120.00   | 141.00       | 326.00      |
| boreratio        | 205.0    | 3.329756     | 0.270844    | 2.54    | 3.15    | 3.31     | 3.58         | 3.94        |
| stroke           | 205.0    | 3.255415     | 0.313597    | 2.07    | 3.11    | 3.29     | 3.41         | 4.17        |
| compressionratio | 205.0    | 10.142537    | 3.972040    | 7.00    | 8.60    | 9.00     | 9.40         | 23.00       |
| horsepower       | 205.0    | 104.117073   | 39.544167   | 48.00   | 70.00   | 95.00    | 116.00       | 288.00      |
| peakrpm          | 205.0    | 5125.121951  | 476.985643  | 4150.00 | 4800.00 | 5200.00  | 5500.00      | 6600.00     |
| price            | 205.0    | 13276.710571 | 7988.852332 | 5118.00 | 7788.00 | 10295.00 | 16503.00     | 45400.00    |

#### <a name="Datapreperatie"></a> Data preperatie
##### Heatmap analyse
Voordat begonnen is aan normalisatie en standaardisatie is gekeken of dit daadwerkelijk nodig was. Er is begonnen door eerst een heatmap te genereren om te kijken of er waardes zijn met een hoge correlatie, zodat deze eruit gefilterd kunnen worden om een bias te voorkomen. Zie de heatmap. Te zien is hoe "highwaympg" en "citympg" een correlatie hebben van 0.97. In het model laten we deze dan ook weg.

![](HeatmapmlrCarPrices.png)

##### Standaardiseren
Als tweede stap is gekeken naar standaardisatie, zijn er kolommen die aangepast moeten worden om tot een beter resultaat te komen? Alle kolommen die geen nummerieke waarden bevatten zijn omgezet naar tabellen die wel nummerieke waarde bevatten, doormiddel van de "get_dummies()" functie van Pandas. Echter is een kolom, "CarName", niet efficiënt om op deze manier te standaardiseren. De kolom "CarName" heeft 205 waardes die bestaan uit unieke auto merken en types. Als deze kolom gestandaardiseerd word, resulteert dit in 205 nieuwe kolommen met 204 nullen en één 1. Dit leidde in de versie 1, tot een lage r2 score en een hoge rmse bij een test set van 30 procent. Om de kolom "CarName" te verbeteren is er gekozen om alle merken te categoriseren, zodoende werden alle type auto's van hetzelfde merk onder één naam gezet.

##### Normalisatie
Als derde stap is gekeken of normalisatie nodig zou zijn. In eerste instantie waren er geen kolommen die uitschietende waardes hadden. Alleen de target kolom had hoge waardes omdat er prijzen gehanteerd worden, maar aangezien dit de target kolom was zou zijn bij multiple linear regression zou deze sowieso niet meegenomen worden in de normalisatie. Om toch te kijken of normalisatie een positief effect zou hebben, in het geval dat er een andere target kolom gehanteerd zou worden, is deze toegepast. Echter waren de waardes na normalisatie zo abnormaal dat normalisatie niet is toegepast.

| Na Normalisatie bij multiple linear regression |                         |
|------------------------------------------------|-------------------------|
| rmse:                                          | 30267727458.953026      |
| r2:                                            | -2.7177251408947733e+22 |

Hier toelichten dat dit per model anders is!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## <a name="Fase1"></a> Fase 1
### <a name="mlr"></a> Multiple linear regression
#### Beschrijving
Multiple linear regression is een machine learning model die binnen supervised learning wordt toegepast. Dit model zoekt zoekt een relatie tussen afhankelijke en een of meer onafhankelijke variabelen door de best passende rechte lijn te plaatsen, ookwel de regressielijn. Aan de hand van deze regressielijn kan het model voorspellingen maken op basis van nieuwe data. Hoe minder variantie deze regressielijn heeft, hoe beter de voorspelling. In deze toepassing zoekt het model in de dataset "Carprices" een lineair verband tussen de afhankelijke variabele, carprice, en de overige features die zijn over gebleven na de data preperatie.

#### Code
~~~
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
      
y = df.price
x = df.drop('price', 1)

#Normalisatie (n.v.t.)
#x = (x-x.min())/(x.max()-x.min())

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
- RMSE:  3041.884027168188 
- R2:  0.8308355282333387

![](LinearverbandPredyTesty.png)

#### Evaluatie
Het uiteindelijke resultaat laat zien dat er een rootmeansquarederror(RMSE) is van ongeveer 3000 euro met R2 score van ongeveer 83 procent. Dit laat zien dat er een vrij goed lineair verband is tussen "Carprice" en alle andere features. Er kan dus met ongeveer 83 procent zekerheid worden voorspeld wat de prijs van een auto zal zijn op basis van deze dataset. De RMSE ziet er ook vrij normaal uit aangezien de gemiddelde prijs van een auto rond de 13000 euro ligt. Een rmse van 3000 euro ziet er daarom niet gek uit. Tevens is er ook een logisch verband te zien in de plot waarbij de voorspelde target tegenover de daadwerkelijk target is gezet. Zo is te zien dat de regressielijn weinig uitschieters heeft en niet al te veel variantie.

#### Feedback
In de feedback momenten, werd er vooral aangekaart dat we meer aandacht moesten bieden aan het voorbereiden van de data. De geschreven code was prima, maar er moest meer gefocust worden op het analyseren van de verbanden in de dataset om te kijken hoe de resultaat het beste zou worden.
Zo is er aangeraden om de kolom "CarName" op te splitsen in merknamen i.p.v. type auto's. Ook werd als tip gegeven om aan de hand van de heatmap te kijken welke attributen een goede correlatie hadden en dus een biassed uitkomst konden leveren.

### <a name="lr"></a> Logistic regression
#### Beschrijving
Logistic regression is een machine learning model die binnen supervised learning wordt toegepast voor classificatie. In de statistiek wordt logistische regressie gebruikt om een geclassificeerde uitkomstvariabele te relateren aan een of meer variabelen. Hierbij is de geclassificeerde uitkomstvariabele dichotoom. Een logistisch model bepaalt dit door i.p.v. een lineaire regressie lijn, een s-vormige log-functie toe te passen. Hierbij kan een logistisch model de kans weergeven of de uitkomstvariabele het een of het ander is gebaseerd op de onafhankelijke inputvariabelen. In dit geval is het belangrijk dat deze uitkomstvariabele omgezet wordt naar een binaire vorm, zo zijn de prijzen opgedeeld in hoog of laag. Afhankelijk van het gemiddelde is bepaald of de prijs van een auto laag of hoog is.

#### Code
~~~~
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
#print(df.describe())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

ToBinairize = ['price']# alleen target binairizen

q =0
while q < len(ToBinairize):
    bins = (int(min(df[ToBinairize[q]])-1), int(np.mean(df[ToBinairize[q]])), int(max(df[ToBinairize[q]])+1))
    group_names = [0, 1]
    df[ToBinairize[q]] = pd.cut(df[ToBinairize[q]], bins = bins, labels=group_names)
    q+=1

#print(df.info())
      
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = LogisticRegression(max_iter=662)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

probs = model.predict_proba(x_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)

print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Accuracy score:\n",accuracy_score(y_test, y_pred))
~~~~


#### Output
AUC score: 
0.95

Confusion matrix:
|              | Predicted True | Predicted False |
|--------------|----------------|-----------------|
| Actual True  |       41       |        3        |
| Actual False |        4       |        14       |

Classification Report:
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            |    0.91   |  0.93  | 0.92     | 44      |
| 1            |    0.82   |  0.78  | 0.80     | 18      |
| accuracy     |           |        | 0.89     | 62      |
| macro avg    | 0.87      | 0.85   | 0.86     | 62      |
| weighted avg | 0.89      | 0.89   | 0.89     | 62      |

Accuracy score:
0.8870967741935484
 
ROC curve:

![](ROClogisticregression.png)


#### Evaluatie
Het uiteindelijke resultaat laat zien dat er een accuracy score is van ongeveer 88 procent. Dit laat zien dat er geen goed verband is tussen de geclassificeerde output variabele en de andere features. Tevens geeft de ROC score een mooie lijn aan die snel stijgt en een AUC score heeft van 0.95.

#### Feedback
Tijdens het feedbackmoment werd er aangekaart dat de evaluatie technieken werden toegepast over het getrainde model. Bij logistic regression hoort een confusion matrix en ROC curve toegepast te worden om te evalueren hoe goed het model presteert.

## <a name="Fase2"></a> Fase 2
### <a name="rf"></a> Random forests 
#### Beschrijving
Randomforests is een voorbeeld van ensemble-leermethodes voor classificatie en regressie, die werken door een veelvoud aan beslissingsbomen te construeren tijdens het trainen van een model. Deze beslissingsbomen worden opgebouwd op basis van een bootstrapped dataset, iedere beslissingsboom heeft zijn eigen bootstrapped dataset. Deze bootstrapped datasets bestaan uit dezelfde records als de originele dataset, alleen worden de records willekeurig uitgekozen en kunnen ze vaker dan een keer voor komen in de bootstrapped dataset. De beslissingsbomen krijgen vervolgens een willekeurige root feature toegewezen om de beslissingboom mee te laten beginnen. Deze stap wordt herhaald tot dat alle features zijn gebruikt. Om nu een uitkomstvariable te verspellen word elk record door iedere beslissingsboom gelopen. Alle uitkomsten van ieder record worden naast elkaar gelegd om vervolgens op basis van het gemiddelde of de modus hiervan de uitkomstvariabele te voorspellen. Om nu te controleren of deze voorspelling accuraat is kunnen de out-of-bag records door de beslessingsbomen lopen. Deze out-of-bag records zijn de records die niet in de bootstrap datasets zijn meegenomen. Alle out-of-bag records vormen samen een out-of-bag dataset die gebruikt kan worden om de accuraatheid van de random forest te meten. Inprincipe wordt de voorspelde uitkomstvariabele naast de daadwerkelijke variabale van de out-of-bag record gelegd om te kijken of de voorspelling accuraat was. Het aantal fout voorspelde resultaten wordt ook wel "out-of-bag-error" genoemd.

Een belangrijke parameter bij randomforests zijn het aantal decisiontrees die van toepassing zijn in het model om de beste score te krijgen. Zoals te zien hier onder, is geanalyseerd welke hoeveelheid aan desicion trees benodigd zou zijn voor de beste score bij een regressor-, en classifiermodel. 

##### Aantal bomen regressor model:
Te zien in onderstaande resultaten lopen de scores langzaam op. Interessant om te zien is dat het model blijkbaar bij 100 bomen even slechter presteert en vervolgens bij duizend bomen weer een stukje beter presteert. Er is in dit model gekozen voor 10000 bomen aangezien de bereken tijd voor het resultaat van 10000 bomen niet rendabel is voor het verschil dat het levert.

| Trees  | R2-score           |
|--------|--------------------|
| 20     | 0.8776208936835751 |
| 30     | 0.8839495624935526 |
| 50     | 0.8853196824674148 |
| 100    | 0.8818481683280802 |
| 1000   | 0.8859202099069733 |
| 10000  | 0.8874014607427503 |
| 100000 | 0.8875302857180773 |

##### Aantal bomen classifier model:
1000 en 10000 decision trees hebben uiteindelijk de beste score. 1000 trees zal worden gehanteerd binnen de code omdat daarvan de compile tijd korter zal zijn. Interessant om te zien is dat bij een random forest van 100000 trees de score weer lager wordt, dit heeft waarschijnlijk te maken met overfitting.

| Trees  | Accuracy score     |
|--------|--------------------|
| 20     | 0.8548387096774194 |
| 30     | 0.8548387096774194 |
| 50     | 0.8548387096774194 |
| 100    | 0.8548387096774194 |
| 1000   | 0.8709677419354839 |
| 10000  | 0.8709677419354839 |
| 100000 | 0.8548387096774194 |

#### Code Regressor
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
#(df.describe())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

#print(df.info())
     
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = RandomForestRegressor(n_estimators=10000, random_state=1)


model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 score:', r2_score(y_test,y_pred))
~~~~

#### Code Classifier
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
#print(df.describe())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

ToBinairize = ['price']

q =0
while q < len(ToBinairize):
    bins = (int(min(df[ToBinairize[q]])-1), int(np.mean(df[ToBinairize[q]])), int(max(df[ToBinairize[q]])+1))
    group_names = [0, 1]
    df[ToBinairize[q]] = pd.cut(df[ToBinairize[q]], bins = bins, labels=group_names)
    q+=1

#print(df.info())
      
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = RandomForestClassifier(n_estimators=1000, random_state=1)


model.fit(x_train, y_train)

y_pred = model.predict(x_test)

probs = model.predict_proba(x_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)


print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Accuracy score:\n",accuracy_score(y_test, y_pred))
~~~~
#### Output Regressor
- RMSE: 2481.7295515565033
- R2: 0.8874014607427503

#### Output Classifier
AUC score:
0.93

Confusion matrix:
|              | Predicted True | Predicted False |
|--------------|----------------|-----------------|
| Actual True  |       40       |        4        |
| Actual False |        4       |        14       |

Classification Report:
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.91      | 0.91   | 0.91     | 44      |
| 1            | 0.78      | 0.78   | 0.78     | 18      |
| accuracy     |           |        | 0.87     | 62      |
| macro avg    | 0.84      | 0.84   | 0.84     | 62      |
| weighted avg | 0.87      | 0.87   | 0.87     | 62      |

Accuracy score:
0.8709677419354839

ROC curve:

![](ROCrandomforestclassifier.png)

#### Evaluatie
##### Regressor:
Het resultaat geeft weer dat het model een R2-score geeft van ongeveer 88 procent en een RMSE van 2481 euro. Dit houdt in dat de randomforest regressor beter werkt op de dataset dan de multiple linear regression. Dit houdt in dat de randomforest regressor minder variantie heeft dan de multiple linear regression en dus betere voorspellingen kan maken.

##### Classifier
Het classifier resultaat laat zien dat er een accuracy is van ongeveer 87 procent bij een randomforest van 1000 bomen. Net als voorgaande modellen laat dit model zien dat er een goed verband is tussen de target feature en de andere features. Net als bij de logistic regression is hier een goede ROC curve te zien, waarbij de lijn vrij snel stijgt en een AUC score aanwezig is van 0.93. Echter geeft het logistic regression model een accurater model weer met een verschil van 1 procent. Tevens is de AUC score ook hoger met een verschil van 3 procent.

#### Feedback
Bij het feedbackmoment zijn er een paar vragen gestelt m.b.t. de visualisatie van het resultaat. Wij waren er nog niet mee bekend dat we de confusion matrix en de ROC curve bij alle classificatie modellen konden gebruiken voor de visualisatie. Na het feedback moment zijn deze toegevoegd voor verduidelijking van het resultaat.

### <a name="nn"></a> Neurale netwerken
#### Beschrijving
Een neuraal netwerk is een techniek die binnen de machinelearning wordt toegepast om op basis van complexen datasets voorspellingen te maken. Neurale netwerken bestaan uit meerdere lagen die nodes kunnen bevatten. Zo zijn er inputlayers, hiddenlayers en outputlayers. In deze layers bestaan nodes die inputvariable omzetten naar outputvariabelen op basis van het gewicht/bias en de activatie functie. Dit is een iteratief proces waarbij na ieder proces het gewicht/bias wordt bijgewerkt om een beter resultaat te krijgen. Dit doet het neurale netwerk aan de hand van "backpropagation", dit is een techniek die de richtingscoëfficient van de errorfunctie bepaald.

Om neurale netwerken in de praktijk toe te passen zijn er een aantal parameters die belangrijk zijn. In dit model definiëren we het aantal hiddenlayers, het aantal nodes binnen de hiddenlayers, de activatiefunctie en de optimizer functie. Per parameter is er gekeken wat het beste resultaat leverde, beginnend bij de hiddenlayers en eindigend bij de optimizer functie. Hierbij is in het begin gebruik gemaakt van de default options tot dat alle parameters onderzocht waren. 

##### Regressor parameters
Bij het selecteren van de parameters werd al vrij snel duidelijk dat weinig activatie- en solver functies goed werken op deze dataset. Er was maar één solver functie die logsiche resultaten weergaf en dat was 'lbfgs'. Op basis van deze functie is besloten de andere parameters te bepalen.

Voor de activatie functie is er gekozen voor relu aangezien deze de meest logische scores weergaf.

|          | RMSE    |  R2-score    |
|----------|---------|--------------|
| Relu     | 3619    | 0.761        |
| Tanh     | 7508    | -0.0306      |
| Logistic | 7548    | -0.0417      |

Voor het aantal hiddenlayers is voor 1 gekozen aangezien het probleem niet complex genoeg is voor meerdere layers. Als meer layers worden toegepast leid dit tot overfitting. Wat wel van belang is bij dit model zijn het aantal nodes in de hiddenlayer. Gekozen is voor 5 nodes aangezien dit het minste verschil gaf en het beste resultaat. 1-3 nodes geven dusdanige resultaten dat het om underfitting gaat, en na 6 nodes worden de resultaten zo wisselvallig dat het om overfitting gaat.

| Nodes   | R2-score       | b/v tradeoff |
|---------|----------------|--------------|
| 1 -2    | -0.031         | Underfitting |
| 3       | 74% - 85%      | Underfitting |
| 4       | 78% - 86%      | Underfitting |
| 5       | 82% - 90%      |              |
| 6       | 78% - 90%      | Overfitting  |
| 7       | 73% - 68%      | Overfitting  |

Uiteindelijke parameters:
- Hiddenlayers (hl): (1)
- Nodes binnen hl: (5)
- Activation rule: 'relu'
- Solver: 'lbfgs'

##### Classifier parameters:
We hebben gekozen voor 1 hiddenlayer om dat er niet meer nodig waren aangezien ons probleem niet zo complex is. op het moment dat 2 of meer hiddenlayers worden toegepast in dit model, krijgen we te maken met overfitting. 

Dit zelfde geldt ook voor het aantal nodes binnen de hiddenlayers. Aan onderstaande resultaten is te zien dat het model het beste resultaat geeft met 3 tot 10 nodes binnen de hiddenlayers. 7 nodes gaven in dit model het beste resulaat.

| Nodes   | Accuracy score | b/v tradeoff |
|---------|----------------|--------------|
| 1 - 2   | 30% - 90%      | Underfitting |
| 3 - 10  | 85% - 91%      |              |
| 10 >    | < 85%          | Overfitting  |

Bij het bepalen van de juiste activatie functies hebben we gekeken welke het beste past bij de dataset. De activation parameter bevat 3 functies, 'relu', 'logistic' en 'tanh'. In de resultaten is te zien dat tanh slechtere resulaten levert dan relu en logistic, daarentegen geven relu en logistic ongeveer dezelfde resultaten weer. Omdat de dataset vooral continue waardes bevat is er gekozen voor de relu functie.

| Ac. Functie | Accuracy score |
|-------------|----------------|
| Relu        | 85% - 90%      |
| Logistic    | 85% - 90%      |
| Tanh        | < 83%          |

Ten slotte de optimizer functie, hierbij zijn 3 mogelijkheden: 'adam', 'lbfgs' en 'sgd'. Hierbij zijn we tot de conclusie gekomen dat de adam solver de minste variatie heeft in resultaten. Ondanks dat de stochastische gradient-descent methode het hoogste resultaat leverde, varieerde deze meer in resultaat dan adam en lbfgs. Wij denken dat dit komt door overfitting.

| Sol. Functie | Accuracy score |
|--------------|----------------|
| Adam         | 85% - 90%      |
| Lbfgs        | 79% - 85%      |
| Sgd          | 85% - 94%      |

Uiteindelijke parameters:
- Hiddenlayers (hl): (1)
- Nodes binnen hl: (7)
- Activation rule: 'relu'
- Solver: 'adam'

#### Regressor Code
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
#print(df.describe().transpose())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

#print(df.info())
      
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = MLPRegressor(hidden_layer_sizes=(5), activation='relu', solver='lbfgs', max_iter=500)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rtwo = r2_score(y_test, y_pred)
print('\nrmse: ', math.sqrt(mse), '\nr2: ', rtwo)
~~~~

#### Classifier Code
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
print(df.describe().transpose())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

ToBinairize = ['price']

q =0
while q < len(ToBinairize):
    bins = (int(min(df[ToBinairize[q]])-1), int(np.mean(df[ToBinairize[q]])), int(max(df[ToBinairize[q]])+1))
    group_names = [0, 1]
    df[ToBinairize[q]] = pd.cut(df[ToBinairize[q]], bins = bins, labels=group_names)
    q+=1

#print(df.info())
      
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = MLPClassifier(hidden_layer_sizes=(7), activation='relu', solver='adam', max_iter=600)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

probs = model.predict_proba(x_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)


print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Accuracy score:\n",accuracy_score(y_test, y_pred))
~~~~

#### Regressor ouput
- RMSE: 2826.0780483729486
- R2: 0.8539867877302934 

#### Classifier output
AUC score:
0.93

Confusion matrix:
|              | Predicted True | Predicted False |
|--------------|----------------|-----------------|
| Actual True  |       41       |        3        |
| Actual False |        5       |        13       |

Classification Report:
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.89      | 0.93   | 0.91     | 44      |
| 1            | 0.81      | 0.72   | 0.76     | 18      |
| accuracy     |           |        | 0.87     | 62      |
| macro avg    | 0.85      | 0.83   | 0.84     | 62      |
| weighted avg | 0.87      | 0.87   | 0.87     | 62      |

Accuracy score:
0.8709677419354839

ROC curve:

![](ROCneuralnetworkclassifier.png)

#### Evaluatie
##### Regressor
Het resultaat geeft weer dat er een vrij goed verband is tussen de target value en de onafhankelijke features. Met een RMSE van 2826 en een R2-score van ongeveer 85% presteert dit model slechter dan de randomforest regressor maar beter dan de multiple linear regressor. Echter zijn de resultaten zo wisselvallig dat het multiple linear regressor wel consistenter presteert. Er kan dus geconcludeerd worden dat de neuralnetwork regressor niet de beste machinelearning methode is om toe te passen op deze dataset.

##### Classifier
Uiteindelijk kan geconstateerd worden dat ook in dit model een goed verband is tussen de target value en de onafhankelijke features. Bovenstestaande score is het gemiddelde van de variatie in scores die naar voren kwamen. Met een accuraatheid van ongeveer 87 procent en een AUC score van 93 procent komen de scores overeen met die van de randomforest, wat inhoudt dat het logistische model beter presteert. 

#### Feedback
In het feedback moment is geconcludeerd dat 3 hiddenlayers te veel zijn en dat deze zorgen voor overfitting, omdat er sprake was van een hoge variance en een lage bias. In de toelichting van het model is de onderbouwing van het aantal hiddenlayers geupdate. Op basis van deze gedachtegang hebben we ook de andere parameters benaderd en onderbouwd.

## <a name="Fase3"></a> Fase 3
### <a name="svm"></a> Support vector machines
#### Beschrijving
Support vector machines is een machinelearning methode die wordt toegepast om, netzoals de logistische regressie, uitkomstvariabelen te classificeren. De uitkomstvariabelen is wederom dichotoom en bestaat dus uit een 1 of 0, hoog of laag, etc. De support vector machine maakt hierbij geen gebruik van kansen en ook niet van een s-curve om de classificatie te bepalen. Support vector machines maken gebruik van hyperplanes om datapunten te classificeren. Deze hyperplanes scheiden de datapunten aan de hand van een margin, dit is de maximale afstand van de support vectors aan beide kanten van de scheiding. Support vectors zijn de vectoren/datapunten die het dichtstbij de scheiding liggen.

Supportvector machines kunnen in meerdere dimensies worden toegepast. Dit kan in 1 dimensie, waarbij er één lijn is waar de datapunten zich op bevinden en deze gescheiden worden door een hyperplane in de vorm van punten op de lijn. Maar ook in 2 dimensies kunnen support vector machines worden toegepast. Hierbij is er een x en y geplot waarin de datapunten zijn verdeeld en gescheiden worden door een hyperplane die bestaat uit drie lijnen. De middelste representeerd de scheiding/classificatie en de buitenste lijnen zijn de parallel lopende support vector lijnen die de margin bepalen, ook wel een 1 dimensionale hyperplane. Bij 3 dimensies zijn de datapunten verdeeld over drie assen waarbij deze gescheiden worden door een 2 dimensionale hyperplane. Dit zijn drie vlakken waarvan het middelste vlak de classifier is en de twee buitenste vlakken de supportvectorvlakken vormen die de margin weergeven. Om te ontdekken welke dimensie van toepassing is of in welke dimensie de beste resultaten naar voren komen wordt daar kernel functies voor toegepast.

Net als bij neurale netwerken, zijn hier ook parameters van belang die de accuraatheid van de resultaten beïnvloeden. Bij supportvector machines zijn dat: 'c', 'gamma' en 'kernel'. C staat voor de mate aan misclassifacitie binnen de margin, gamma is de grote in margin en kernel geeft de functie aan die toegepast wordt om de beste resultaten te vinden in de verschillende dimensies. Voor de grote in margin is gekozen om de gamma op 'auto' te zetten aangezien deze functie de optimale margin voor ons berekend bij beide modellen.

##### Regressor parameters
Beginnend bij de kernel functie valt al direct op dat alle resultaten vrij slecht zijn, dit indiceert al dat suportvector machine regressor niet een perfect model zal zijn om voorspellingen mee te maken op deze dataset. Maar om het model toch een kans te geven hebben we gekeken of er betere resultaten uitkwamen bij het gebruik van de lineare kernel functie, aangezien deze de minst slechte score had.

| Kernel      | R2-score       |
|-------------|----------------|
| linear      | -0.019         |
| poly (d=1)  | -0.037         |
| poly (d=2)  | -0.037         |
| poly (d=3)  | -0.037         |
| rbf         | -0.037         | 
| sigmoid     | -0.037         |
| precomputed | n.v.t          |

Bij de mate van misclassificatie is bij onderstaande resultaten voor een waarde van 10000 gekozen. Dit is een erg hoge waarde, wat inhoudt dat misclassificatie enorm vermeden wordt maar dit geeft wel een goed resultaat weer. bij een misclassificatie waarde van 20000 wordt de margin waarde waarschijnlijk zo klein dat er een overfitting plaatsvindt waardoor de R2-score weer daalt.

| C         | R2-score       | b/v tradeoff |
|-----------|----------------|--------------|
| 1         | -0.019         | Underfitting |
| 5         | 0.060          | Underfitting |
| 10        | 0.144          | Underfitting |
| 100       | 0.559          | Underfitting |
| 1000      | 0.820          | Underfitting |
| 10000     | 0.891          |              |
| 20000     | 0.880          | Overfitting  |

##### Classifier parameters
Bij de mate van misclassificatie is naar aanleiding van onderstaande tabel gekozen voor een C van 1.0. De range van 1.0 tot 5.0 gaven allemaal dezelfde score van ongeveer 90 procent weer. Besloten is om 1.0 toe te passen aangezien niet hoger nodig is voor het model en om te veel misclassificatie te voorkomen. Na 5.0 is interessant om te zien dat de accuraatheid weer daalt dit komt waarschijnlijk door overfitting.

| C         | Accuracy score | b/v tradeoff |
|-----------|----------------|--------------|
| 0.1       | 0.70           | Underfitting |
| 0.5       | 0.85           | Underfitting |
| 1.0 - 5.0 | 0.90           |              |
| 5.1 >     | < 0.87         | Overfitting  |

Op basis van onderstaande tabel is de kernel functie bepaald. Het resultaat geeft weer dat de polynominale functie met een degree van 1 de beste resultaten uit het model haalt.

| Kernel      | Accuracy score |
|-------------|----------------|
| linear      | 0.89           |
| poly (d=1)  | 0.91           |
| poly (d=2)  | 0.70           |
| poly (d=3)  | 0.70           |
| rbf         | 0.90           | 
| sigmoid     | 0.90           |
| precomputed | n.v.t          |


Uiteindelijke parameters:
- C: (1.0)
- Gamma: 'auto'
- Kernel: 'poly'
- degree: (1.0)

#### Regressor Code
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
#print(df.describe().transpose())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

#print(df.info())
      
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = svm.SVR(C=10000.0, gamma='auto', max_iter=-1, kernel='linear')


model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rtwo = r2_score(y_test, y_pred)
print('\nrmse: ', math.sqrt(mse), '\nr2: ', rtwo)
~~~~

#### Classifier Code
~~~~
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


df = pd.read_csv('Dataset Carprices.csv')
df.head()
df = df.drop(['car_ID', 'highwaympg', 'citympg'], 1)

targetkolom = 'price'


#Preperatie op CarName
i =0
while i < len(df.CarName):
    df.CarName[i] = df.CarName[i].split()[0]
    i += 1
    
pd.set_option('display.max_columns', 200)
print(df.describe().transpose())

#Dataset standaardiseren
df = pd.get_dummies(df, columns=['CarName','fueltype','aspiration','doornumber','carbody',
                                 'drivewheel','enginelocation','enginetype','cylindernumber',
                                 'fuelsystem'], prefix="", prefix_sep="")

ToBinairize = ['price']

q =0
while q < len(ToBinairize):
    bins = (int(min(df[ToBinairize[q]])-1), int(np.mean(df[ToBinairize[q]])), int(max(df[ToBinairize[q]])+1))
    group_names = [0, 1]
    df[ToBinairize[q]] = pd.cut(df[ToBinairize[q]], bins = bins, labels=group_names)
    q+=1

#print(df.info())
      
y = df[targetkolom]
x = df.drop(targetkolom, 1)

#Normalisatie (n.v.t.)
x = (x-x.min())/(x.max()-x.min())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3 ,random_state=7)

model = svm.SVC(C=1.0, gamma='auto', max_iter=-1, kernel='poly', degree=1, probability=True)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

probs = model.predict_proba(x_test)

probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)

print('AUC: %.2f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr, tpr)

print("Confusion matrix:\n", confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Accuracy score:\n",accuracy_score(y_test, y_pred))
~~~~

#### Regressor output
- RMSE: 2437.4477553297274 
- R2: 0.8913838302854323 

#### Classifier output
AUC score:
0.96

Confusion matrix:
|              | Predicted True | Predicted False |
|--------------|----------------|-----------------|
| Actual True  |       40       |        4        |
| Actual False |        2       |        16       |

Classification Report:
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.95      | 0.93   | 0.94     | 44      |
| 1            | 0.84      | 0.89   | 0.86     | 18      |
| accuracy     |           |        | 0.92     | 62      |
| macro avg    | 0.90      | 0.91   | 0.90     | 62      |
| weighted avg | 0.92      | 0.92   | 0.92     | 62      |

Accuracy score:
0.9193548387096774

ROC curve:

![](ROCsvm.png)

#### Evaluatie
##### Regressor
Bij het regressor model is te zien dat er een vrij sterk verband is tussen de target value en de onafhankelijke features. Ondanks het feit dat er een extreem hoge mate nodig was om misclassification te voorkomen waren de resultaten vrij goed: RMSE van 2437 en een R2-score van ongeveer 89 procent. Dit houdt in dat dit model ongeveer even goed presteert als de randomforest regressor.

##### Classifier
Bij dit model valt te concluderen dat er een erg goed verband is tussen de target value en de onafhankelijke features. Met een accuracy score van ongeveer 91 procent en een AUC score van 96 procent is dit de best toegepaste techniek tot nu toe. Ook is het sterke verband aan de snelle stijging van ROC curve te zien.

#### Feedback
Tijdens het feedback moment zijn de parameters toegelicht, aangezien deze onduidelijk waren. Op basis van de toelichting zijn we de parameters gaan analyseren om de beste resultaten te vinden.

### <a name="bn"></a> Bayesian networks
#### Beschrijving 
Bayesian network is een model binnen de machinelearning die toegepast wordt voor "probabilistic reasoning". Hierbij worden de kanstheorie gecombineerd met logisch beredeneren. Bayesian networks worden dan ook vooral toegepast op problemen met onzekerheid. Door het toepassen van kansberekening kan de kans dat een event plaatsvindt worden berekend. Dit wordt gedaan door een netwerk aan nodes te creëren die onderling een bepaald verband hebben. Zo kan de ene node een childnode zijn en de ander een parentnode zijn. Dit houdt in dat de childnode afhankelijk is van de parentnode, en dat de kans dat de childnode plaatsvindt beïnvloed wordt door de parentnode. In voorbeeld1.0 en voorbeeld2.0 wordt de praktijk van bayesian networks beter toeglicht. 

Bij deze voorbeelden zijn onderstaande bronnen benaderd:
- https://www.youtube.com/watch?v=SkC8S3wuIfg&t=1504s
- https://www.youtube.com/watch?v=4fcqyzVJwHM

#### Voorbeeld 1.0
Dit voorbeeld laat zien wat de kansen zijn rondom de toelating van een denkbeeldige studie. Hierbij zijn vier nodes tezien: Examlevel, IQlevel, Marks en Admission.

- Examlevel: e0 = moeilijk, e1 = makkelijk
- IQlevel: i0 = hoog, i1 = laag
- Marks: m0 = voldoende, m1 = onvoldoende
- Admission: a0 = toegelaten, a1 = niet toegelaten

Hierbij is Admission de child node van Marks, wat inhoud dat Admission afhankelijk is van Marks. Tevens is Marks de child node van Examlevel en IQlevel, wat indiceert dat Marks afhankelijk is van Examlevel en IQlevel. Examlevel en IQlevel zijn onafhankelijk van elkaar omdat er geen directe link tussen deze twee nodes bestaat.

![](bayesiannetworkvoorbeeld1.png)
##### Uitwerking
###### Kans op voldoende en onvoldoende

p(m0) = p(m0 | i0 n e0) * p(i0 n e0) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | i0 n e1) * p(i0 n e1) +<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | i1 n e0) * p(i1 n e0) +<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | i1 n el) * p(i1 n e1)<br/>

p(m0) = 0.6 * 0.56 +<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.9 * 0.24 +<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.5 * 0.14 +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.8 * 0.06<br/>

p(m0) = 0.336 + 0.214 + 0.07 + 0.048

p(m0) = 0.668

p(ml) = 0.332

###### Kans op toelating en geen toelating

p(a0) = p(a0 | m0) * p(m0) + p(a0 | m1) * p(m1) 

p(a0) = 0.6 * 0.668 + 0.9 * 0.332

p(a0) = 0.4008 + 0.2988

p(a0) = 0.6996

p(al) = 0.3004

#### Voorbeeld 2.0
Dit voorbeeld is hetzelfde als voorbeeld1.0 echter uitgebreid. Te zien in de uitwerking van voorbeeld1.0 is de kans op toelating ongeveer 70 procent, wat vrij hoog is. Te zien aan de nodes is de kans op een voldoende afhankelijk van maar twee factoren, namelijk: Examlevel en IQlevel. Hierbij wordt alleen gekeken naar de moeilijkheidsgraad van het examen en het niveau van intilligentie van de student. Als toevoeging hierop is Studylevel toegevoegd, omdat hiermee verwacht wordt dat de kans op toelating kleiner en iets realistischer wordt. 

- Studylevel: q0 = goed gestudeerd, q1 = slecht geleerd

![](bayesiannetworkvoorbeeld2.png)
##### Uitwerking
###### Kans op voldoende en onvoldoende
p(m0) = p(m0 | e0, q0, i0) * P(e0 n q0 n i0) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(mO | e0, q1, i0) * p(e0 n q1 n i0) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | e0, q0, i1) * P(e0 n q0 n i1) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | e0, q1, i1) * P(e0 n q1 n i1) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | e1, q0, i0) * p(e1 n q0 n i0) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | e1, q1, i0) * p(e1 n q1 n i0) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | e1, q0, i1) * p(e1 n q0 n i1) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(m0 | e1, q1, i1) * p(e1 n q1 n i1)<br/>

p(m0) = 0,7 * (0,7 * 0,2 * 0,8) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,7 * (0,7 * 0,8 * 0,8) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,7 * (0,7 * 0,2 * 0,2) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,7 * (0,7 * 0,8 * 0,2) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,3 * (0,3 * 0,2 * 0,8) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,3 * (0,3 * 0,8 * 0,8) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,3 * (0,3 * 0,2 * 0,2) +<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0,3 * (0,3 * 0,8 * 0,2)<br/>

p(m0) = 0,0784 + 0,3136 + 0,0196 + 0,0784 + 0,0144 + 0.0576 + 0,0036 + 0,0144

p(m0) = 0 58

p(m1) = 0.42

###### Kans op toelating en geen toelating

p(a0) = p(a0 | m0) * m0 + <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p(a0 | m1) * m1

p(a0) = 0.6 * 0.58 + 0.9 * 0.42

p(a0) = 0.348 + 0.378

p(a0) = 0.726

p(a1) = 0.274

#### Evaluatie
Te zien in voorbeeld1.0 is dat de kans om toegelaten te worden op basis van 3 factoren ongeveer 70% is. Dit leek in eerste instantie vrij hoog als toelating voor een studie. Daarom is een tweede voorbeeld uitgewerkt waarbij een extra factor, Studylevel, is toegevoegd. Dit om te onderzoeken of het niveau van studiewerk impact zou hebben op de toelatingskans van de denkbeeldige studie. Wat interessant is om te zien, is dat met de toegevoegde factor, in voorbeeld2.0, de kans op toelating met 2% stijgt. Dit kan bijvoorbeeld komen door het fenomeen dat wanneer er features worden toegevoegd de resulaten altijd een beetje hoger worden.

#### Feedback
In het feedback moment is de uitleg over bayesian networks toegelicht en kregen we als feedback om onze bronnen bij de uitleg te vermelden. Ook hadden we een vraagteken rondom het verschil in resultaat van voorbeeld1.0 en voorbeeld2.0 waarbij aangegeven werd dat de R2-score hoger wordt als er nieuwe features worden toegevoegd.

## <a name="AlgeheleConclusie"></a> Conclusie
### Conclusie regressor models
Te zien in onderstaande tabel, uit alle RMSE- en R2-scores van de regressie modellen, dat de Support vector machine het beste scored. Echter kom het model van de randomforest erg dichtbij alleen is de RMSE daarvan een stuk hoger. Het neural network model heeft ook een vrij hoge score, maar zoals eerder al geconcludeerd bij de evaluatie was deze zo wisselvallig dit model waarschijnlijk niet van toepassing is op deze dataset.

|                | Linear regression   |Random forests  | Neural network | Support vector machine |
|----------------|---------------------|----------------|----------------|------------------------|
| RSME           | 3041.88             | 2481.73        | 2826.08        | **2437.45**            |
| R2             | 0.831               | 0.887          | 0.854          | **0.891**              |


### Conclusie classification models
Als we alle accuracy en AUC scores naast elkaar leggen is duidelijk te zien dat het supportvector machine model het beste heeft gepresteerd.

|                | Logistic regression | Random forests | Neural network | Support vector machine |
|----------------|---------------------|----------------|----------------|------------------------|
| Accuracy score | 0.887               | 0.871          | 0.871          | **0.919**              |
| AUC score      | 0.95                | 0.93           | 0.93           | **0.96**               |


## <a name="Auteurs"></a> Auteurs
- Rutger de Groen https://rutgerfrans.com/
- Maroche Delnoy https://www.linkedin.com/in/maroche-delnoy-788ab9195/
