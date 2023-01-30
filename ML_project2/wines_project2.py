# -*- coding: utf-8 -*-
"""
Uczenie maszynowe - ML PROJECT 2

Dataset 'wines_SPA.csv' pochodzi ze strony kaggle.com,
zas czesc implementacji oparta jest na rozwiazaniach pochodzacych z wykladu

Roznice w porownaniu z projektem 1:
W projekcie pierwszym przewidywane byly wartosci w kolumnie 'price',
natomiast w tym projekcie skupilem sie na stworzeniu klasy win najlepiej ocenianych
przy wykorzystaniu kolumny 'rating'
"""


#%% import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")


#%% przygotowanie srodowiska pracy i wczytanie danych

KATALOG_PROJEKTU = os.path.join(os.getcwd(), "projekt_winery")
KATALOG_DANYCH = os.path.join(KATALOG_PROJEKTU, "dane")

plik = 'wines_SPA.csv'
df = pd.read_csv(os.path.join(KATALOG_DANYCH, plik))


#%% przygotowanie danych
"""
Wybrany dataset zawiera dane o 7500 hiszpanskich winach,
okreslone zostaly podstawowe informacje o trunkach takie jak ich pochodzenie,
ceny, oceny konsumentow, liczba konsumentow oceniajacych trunek, a takze w pewnej skali ich walory smakowe
"""

print('\nStarting shape of dataset\n', df.shape ,'\n')
print(df.info())

"""
czesc preprocessingu - kasowanie rekordow z brakujacymi danymi
    
w datasecie znajduja sie pozycje bez wartosci co mozna wywnioskowac dzieki poleceniu df.info(),
dodatkowo kolumna 'country' przyjmuje tylko wartosc Spain przez to ze dataset jest o winach tylko z Hiszpanii,
dlatego dataset zostaje pomniejszony o te dane - po przeprowadzeniu dropu, nasz dataset ma 6070 elementow
"""
df['year'] = df['year'].replace('N.V.', np.NaN)
df = df.dropna()
df = df.drop(columns=['country'])
print('\nShape of dataset after dropping values\n', df.shape)


#%% analiza danych
"""
WSTEPNA ANALIZA DANYCH W KOLUMNACH

acidity - dosyc malo roznorodna kolumna, zdecydowanie dominuje wartosc 3.0
"""
print('\nAcidity - number of values\n',df['acidity'].value_counts())

"""
rating - dominuje wartosc 4.2 (najnizsza ze wszystkich), natomiast im wyzsza wartosc rating,
tym mniejsza jest liczba win do nich zakwalifikowanych, czyli wystepuje tendencja 
im cos jest rzadsze, tym jest tego mniej
"""
print('\nRating - number of values\n',df['rating'].value_counts())

"""
price - wykres ponizej ukazuje 25 najczesciej wystepujacych cen dla win z datasetu,
do 21 cen zostalo przypisanych po okolo 200 win, przy czym rozstrzal miedzy tymi cenami
jest dosyc spory (najnizsza cena to 16.76 a najwyzsza 77.36), natomiast najczesciej wystepujaca cena to 37.90;
daje to nam okolo 4200 rekordow przypisanych do 21 cen, reszta rekordow jest zas przypisana 
do mniej popularnych cen
"""
plt.title('Number of wine bottles in most common prices')
sns.countplot(y='price', data=df, palette="Purples", order=df['price'].value_counts().iloc[:25].index)
plt.show()

"""
heatmapa korelacji miedzy kolumnami - najwyzszy wspolczynnik korelacji na poziomie 0.55
maja kolumny rating i price
"""
plt.title('Heatmap showing correlation between columns')
sns.heatmap(df.corr(), annot=True, cbar_kws = {'label': 'correlation scale'}, cmap='Oranges')
plt.show()

"""
ze wzgledu na wystepowanie najwyzszej korelacji przy ratingu i problemy przy przeprowadzaniu klasyfikacji,
zdecydowalem sie na stworzenie klasy top do ktorej zaliczane jest 10% najlepiej ocenianych win 
(czyli pozycje wystepujace najwyzej w kolumnie 'rating'), reszta win jest zaliczana do klasy drugiej

(podzial nastapil ze wzgledu na zauwazalne roznice wystepujace miedzy winami w klasie top a poza nią)
"""
df['top'] = 0
df['top'].loc[(df['rating'] > np.percentile(df.rating,90))] = 1

"""
zalicza sie do tej klasy 624 wina, czyli okolo 10% rekordow datasetu
"""

"""
(PROBLEM)
Czy algorytmy klasyfikacji dobrze się sprawdzają do rozrozniania najlepiej ocenianych win.
"""


#%% preprocessing czesc dalsza
"""
kolejna czesc PREPROCESSINGU

(1) zamiana danych nienumerycznych na liczby
"""
from sklearn.preprocessing import LabelEncoder

for column in df:
    if df[column].dtypes == 'object':
        df[column] = LabelEncoder().fit_transform(df[column])
        
#print('\nData after encoding\n', df.head())
        
"""
(2) przeprowadzenie standaryzacji danych przed podzialem
"""
df = ( df - df.mean() ) / df.std()
#print('\nData after standarization\n', df.head())

"""
(3) podzial danych na zbiory trenujace i testujace
w tym miejscu takze wybrana zostaje kolumna 'price' jako zmienna y
"""
from sklearn.model_selection import train_test_split

X = df.drop(columns=['top', 'rating'],axis=1)
y = df['top']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
(4) skalowanie danych i transformacja
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)


#%% test algorytmow
"""
TEST ALGORYTMOWOW ZESPOLOWYCH
przeprowadzenie testu, ktory z algorytmow da najlepsze wyniki dla okreslonego celu w tym zbiorze danych
"""
from sklearn.metrics import accuracy_score
accuracy_test_scores = {}

"""
logistic regression
"""
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(solver='liblinear', random_state=42)
log_clf.fit(X_train, y_train)

y_pred = log_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['Logistic Regression'] = ac_score

"""
random forest classifier
"""
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=10,random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred = rnd_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['RandomForestClassifier'] = ac_score

"""
SVC
"""
from sklearn.svm import SVC

svm_clf = SVC(gamma='auto', probability=True, random_state=42)
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['SVC'] = ac_score

"""
(1) Voting Classifier
"""
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft'
        )

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['VotingClassifier'] = ac_score

"""
(2) Bagging Classifier
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, random_state=42
        )

bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['Bagging Classifier'] = ac_score

"""
(3) Ada Boost classifier (boosting)
"""
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5
        )

ada_clf.fit(X_train, y_train)

y_pred = ada_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['AdaBoost Classifier'] = ac_score

"""
(4) Stacking Classifier
    (niestety nie bylem w stanie sprawdzic algorytmu stacking ze wzgledu na problemy
     z wersja oprogramowania w moim komputerze, dlatego pozostawiam zakomentowane)
"""

"""
from sklearn.ensemble import StackingClassifier

stack_clf = StackingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=200)

stack_clf.fit(X_train, y_train)

y_pred = stack_clf.predict(X_test)
ac_score = accuracy_score(y_test, y_pred)
accuracy_test_scores['Stacking Classifier'] = ac_score
"""

"""
Podsumowanie accuracy
"""
print('\nAccuracy scores:\n', accuracy_test_scores)

"""
Najlepszy wynik Accuracy osiaga klasyfikator zespolowy AdaBoostClassifier
Warto tez dodac ze random forest radzi sobie lepiej od klasyfikatora bagging w przypadku drzew
"""
    

#%% CROSS-VALIDATION
"""
przeprowadzenie tego testu pozwolic ocenic jeszcze dokladniej ktory z algorytmow najbardziej
nadaje sie do pracy na wybranym datasecie o hiszpanskich winach
"""

from sklearn.model_selection import cross_val_score
cv_rmse={}
cv_std={}

"""
logistic regression
"""
log_clf_scores = cross_val_score(log_clf, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
log_clf_rmse_scores = np.sqrt(-log_clf_scores)

cv_rmse['LogisticRegression'] = log_clf_rmse_scores
cv_std['LogisticRegression'] = log_clf_rmse_scores.std()

"""
SVC
"""
svm_clf_scores = cross_val_score(svm_clf, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
svm_clf_rmse_scores = np.sqrt(-svm_clf_scores)

cv_rmse['SVC'] = svm_clf_rmse_scores
cv_std['SVC'] = svm_clf_rmse_scores.std()

"""
random forest classifier
"""
rnd_clf_scores = cross_val_score(rnd_clf, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
rnd_clf_rmse_scores = np.sqrt(-rnd_clf_scores)

cv_rmse['RandomForestClassifier'] = rnd_clf_rmse_scores
cv_std['RandomForestClassifier'] = rnd_clf_rmse_scores.std()


"""
(1) Voting Classifier
"""
voting_clf_scores = cross_val_score(voting_clf, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
voting_clf_rmse_scores = np.sqrt(-voting_clf_scores)

cv_rmse['Voting Classifier'] = voting_clf_rmse_scores
cv_std['Voting Classifier'] = voting_clf_rmse_scores.std()

"""
(2) Bagging Classifier
"""
bag_clf_scores = cross_val_score(bag_clf, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
bag_clf_rmse_scores = np.sqrt(-bag_clf_scores)

cv_rmse['Bagging Classifier'] = bag_clf_rmse_scores
cv_std['Bagging Classifier'] = bag_clf_rmse_scores.std()

"""
(3) AdaBoost Classifier
"""
ada_clf_scores = cross_val_score(ada_clf, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
ada_clf_rmse_scores = np.sqrt(-ada_clf_scores)

cv_rmse['AdaBoost Classifier'] = ada_clf_rmse_scores
cv_std['AdaBoost Classifier'] = ada_clf_rmse_scores.std()


"""
Wyniki cross-validation
"""
print('\nCross validation - RMSE values:\n', cv_rmse)
print('\nCross validation - Standard deviation values:\n', cv_std)


"""
Najmniejszy rmse w wiekszosci przypadkow i odchylenie standardowe uzyskuje Voting Classifier, 
lecz niedużo gorsze rezultaty osiąga AdaBoost Classifier, ktory osiagal najlepsze accuracy
w poprzednim tescie, dlatego tez AdaBoost zostaje wybrany jako algorytm
prezentujacy najlepsze wartosci
"""


#%% tuning hyperparametrow
"""
Przeprowadzenie tuningu hyperparametrow dla modelu AdaBoost Classifier
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid={
        'n_estimators': [100, 200, 300],
        'algorithm': ["SAMME.R"], 
        'learning_rate': [0.4, 0.5, 0.6],
        }
        
ada_clf = AdaBoostClassifier(random_state=42)

grid_search = GridSearchCV(ada_clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)
final_model = grid_search.best_estimator_


#%% przedstawienie i ocena finalnego wyniku
"""
Wyliczenie finalnego bledu RMSE dla najlepszej wersji algorytmu AdaBoost Classifier
dla zbioru testowego z predykcja
"""
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)

print('\nBest RMSE for AdaBoost Classifier at test and predicted set: ', 
      np.sqrt(final_mse))

y_pred = rnd_clf.predict(X_test)

print('\nBest accuracy score for AdaBoost Classifier at test and predicted set: ',
      accuracy_score(y_test, y_pred))

"""
Wartosci RMSE oscyluja w okolicach 0.23, co oznacza ze sa jak najbardziej akceptowalne, jest to stosunkowo dobry wynik
dla duzych datasetow

tak samo Accuracy Score, ktore oscyluje w okolicach powyzej 0.93 - co oznacza ze jest blisko 1, do ktorej chcemy sie 
zblizyc jak najbardziej
"""


#%% podsumowanie

"""
Przewidywanie czy dana butelka wina nalezy do jednych z najlepiej ocenianych w zbiorze danych
zostalo zakonczone pomyslnie

Wybrany finalnie algorytm pozwala na przewidywanie przynaleznosci do klasy z zadowalajaca precyzja,
aczkolwiek mozliwe byloby poprawienie wynikow poprzez wykorzystanie jeszcze bardziej zaawansowanych metod
lub dopieszczenie tych wykorzystanych
"""













