# -*- coding: utf-8 -*-
"""
Uczenie maszynowe - ML PROJECT 1

Dataset 'wines_SPA.csv' pochodzi ze strony kaggle.com,
zas czesc implementacji oparta jest na rozwiazaniach pochodzacych z wykladu
"""


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
"""
Wybrany dataset zawiera dane o 7500 hiszpanskich winach,
okreslone zostaly podstawowe informacje o trunkach takie jak ich pochodzenie,
ceny, oceny konsumentow, liczba konsumentow oceniajacych trunek, a takze w pewnej skali ich walory smakowe
"""
df = pd.read_csv('wines_SPA.csv')
print('\nStarting shape of dataset\n',df.shape ,'\n')
#print(df.info())

"""
czesc preprocessingu - kasowanie rekordow z brakujacymi danymi
    
w datasecie znajduja sie pozycje bez wartosci co mozna wywnioskowac dzieki poleceniu df.info(),
dodatkowo kolumna 'country' przyjmuje tylko wartosc Spain przez to ze dataset jest o winach tylko z Hiszpanii,
dlatego dataset zostaje pomniejszony o te dane - po przeprowadzeniu dropu, nasz dataset ma 6070 elementow
"""
df['year'] = df['year'].replace('N.V.', np.NaN)
df = df.dropna()
df = df.drop(columns=['country'])
print('\nShape of dataset after dropping values\n',df.shape)


#%%
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


#%%
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

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
(4) skalowanie danych
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#%%
"""
TEST ALGORYTMOW REGRESYJNYCH

przeprowadzenie testu, ktory z algorytmow do regresji da najlepsze wyniki dla tego datasetu

do okreslenia ktory z nich moze byc najbardziej trafny wykorzystany zostanie wspolczynnik r^2,
ktory im jest wyzszy tym lepiej
"""

from sklearn.metrics import r2_score
model_r2_scores = {}

"""
(1) regresja liniowa
"""
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
model_r2_scores['LinearRegression'] = r2

"""
(2) regresja wedlug drzewa decyzji
"""
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
model_r2_scores['DecisionTreeRegressor'] = r2

"""
(3) regresja oparta o support vector machines
"""
from sklearn.svm import SVR

svm_reg = SVR(gamma='scale')
svm_reg.fit(X_train, y_train)

y_pred = svm_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
model_r2_scores['SVM'] = r2

"""
(4) regresja oparta o random forest regressor
"""
from sklearn.ensemble import RandomForestRegressor

rnd_forest_reg = RandomForestRegressor(n_estimators=100)
rnd_forest_reg.fit(X_train, y_train)

y_pred = rnd_forest_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
model_r2_scores['RandomForestRegressor'] = r2

"""
(5) regresja oparta o k-neighbors regressor
"""
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)

y_pred = knn_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
model_r2_scores['KNeighborsRegressor'] = r2

"""
Podsumowanie testu r2 algorytmow - najwyzsza wartosc wspolczynnika r^2 zapewnil 
random forest regressor, warto dodac ze nieznacznie od niego gorszy okazal sie kneighbors regressor
"""
print('\nR2 values:\n', model_r2_scores)


#%%
"""
CROSS-VALIDATION
przeprowadzenie tego testu pozwolic ocenic jeszcze dokladniej ktory z algorytmow najbardziej
nadaje sie do pracy na wybranym datasecie o hiszpanskich winach
"""

from sklearn.model_selection import cross_val_score
cv_rmse={}
cv_std={}

"""
(1) regresja liniowa
"""
lin_reg_scores = cross_val_score(lin_reg, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
lin_reg_rmse_scores = np.sqrt(-lin_reg_scores)

cv_rmse['LinearRegression'] = lin_reg_rmse_scores
cv_std['LinearRegression'] = lin_reg_rmse_scores.std()

"""
(2) drzewo decyzji
"""
tree_reg_scores = cross_val_score(tree_reg, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
tree_reg_rmse_scores = np.sqrt(-tree_reg_scores)

cv_rmse['DecisionTreeRegressor'] = tree_reg_rmse_scores
cv_std['DecisionTreeRegressor'] = tree_reg_rmse_scores.std()

"""
(3) support vector machines
"""
svm_reg_scores = cross_val_score(svm_reg, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
svm_reg_rmse_scores = np.sqrt(-svm_reg_scores)

cv_rmse['SVM'] = svm_reg_rmse_scores
cv_std['SVM'] = svm_reg_rmse_scores.std()

"""
(4) random forest regressor
"""
rnd_forest_scores = cross_val_score(rnd_forest_reg, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
rnd_forest_rmse_scores = np.sqrt(-rnd_forest_scores)

cv_rmse['RandomForestRegressor'] = rnd_forest_rmse_scores
cv_std['RandomForestRegressor'] = rnd_forest_rmse_scores.std()

"""
(5) k-neighbors regressor
"""
knn_reg_scores = cross_val_score(knn_reg, X_train, y_train,
                              scoring="neg_mean_squared_error", cv=5)
knn_reg_rmse_scores = np.sqrt(-knn_reg_scores)

cv_rmse['KNeighborsRegressor'] = knn_reg_rmse_scores
cv_std['KNeighborsRegressor'] = knn_reg_rmse_scores.std()

"""
Najmniejszy rmse w wiekszosci przypadkow uzyskuje random forest regressor, czyli jest
najlepszy po raz drugi, wartosci odchylenia standardowego tez osiaga jedne z najnizszych,
choc odrobine nizsze osiaga regresja oparta na drzewie decyzji
"""

print('\nCross validation - RMSE values:\n', cv_rmse)
print('\nCross validation - Standard deviation values:\n', cv_std)


#%%
"""
Przeprowadzenie tuningu hyperparametrow dla modelu Random Forest Regressor
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

param_grid=[
        {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]},
        {'bootstrap': [False], 'n_estimators': [30,300], 'max_features':[2, 3, 4]},
            ]

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)
final_model = grid_search.best_estimator_

"""
Wyliczenie finalnego bledu RMSE dla najlepszej wersji algorytmu RandomForestRegressor
dla zbioru testowego z predykcja
"""
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
print('\nBest RMSE for Random Forest Regressor at test and predicted set: ', np.sqrt(final_mse))




















