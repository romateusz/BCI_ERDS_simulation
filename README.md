# BCI_ERDS_simulation

## Autor

👤 **Mateusz Roman**

## Spis treści

- [BCI\_ERDS\_simulation](#bci_erds_simulation)
  - [Autor](#autor)
  - [Spis treści](#spis-treści)
  - [Opis](#opis)
    - [Przykładowe metryki dla ruchu:](#przykładowe-metryki-dla-ruchu)
  - [Architektura](#architektura)
  - [Wymagania](#wymagania)
  - [Instalacja](#instalacja)
  - [Uruchomienie oprogramowania](#uruchomienie-oprogramowania)
    - [1. Część wczytania i obróbki danych](#1-część-wczytania-i-obróbki-danych)
    - [2. Część Analizy i Treningu](#2-część-analizy-i-treningu)
      - [Wybór danych EEG](#wybór-danych-eeg)
      - [Wybór komponentów](#wybór-komponentów)
      - [Wynik](#wynik)

## Opis
--> Poprawić opis!! <--

1. Celem projektu jest wykrywanie zamiaru, jak i faktycznego poruszenia lewego lub prawego palca wskazującego na podstawie sygnału EEG.
   
2. Analiza opiera się na przetworzeniu sygnału na spektrogramy, w celu jakościowej oceny wczytanych sygnałów.  

-> Opisać spektrogramy <-

Oprogramowanie to wczytuje i przetwarza dane EEG w celu klasyfikacji prawdziwego ruchu i zamiaru poruszenia palca wskazującego:

- lewego,
- prawego.


Następnie skorzystano z metody Common Spatial Patterns (CSP), w celu wyodrębnienia cech z kory ruchowej, umożliwiającej znalezienie przestrzennych wzorców związanych z aktywacją kory ruchowej. Na podstawie wektora własnego macierzy `W` i wartości własnych `Lambda` obliczonych za pomocą CSP, wybierana jest reprezentacja najlepiej różnicująca klasy ruchu.

Dane zostały podzielone na zbiór treningowy i testowy (w celu zasymulowania prawdziwego BCI, gdzie częśc danych stałaby się kalibracyjna, a cześć zostałaby użyta już po kalibracji, symulując nie jako użycie na żywo). W celu wykonania klasyfikacji użyto Regresji Logistycznej i metody XGBOOST.

Dla ruchu uzyskane metryki określające jakość wytrenowanego modelu oscylowały w okolicach 75%, natomiast w przypadku wyobrażeń  wynik bardzo zależał od wylosowanego zbioru treningowego i testowego. Aby poprawność wskazań modelu wzrosła, należałoby zwiększyć ilość danych treningowych/kalibracyjnych. Możnaby również pomyśleć nad lepszym wytrenowaniem badanego oraz być może wybrać silniejszy/łatwiejszy do wyobrażenia dla badanego ruch. Nie mniej jednak palce to dobry pomysł, gdyż duże obszary kory odpowiadają za ruchy dłoni i pałców (jest to ściśle związane z ich niezwykłą precyzją i czułością).

### Przykładowe metryki dla ruchu:

**Confusion Matrix:**

```
[[5 1]
 [2 4]]
```

**Classification XGBOOST Report:**

```
              precision    recall  f1-score   support

           0       0.71      0.83      0.77         6
           1       0.80      0.67      0.73         6

    accuracy                           0.75        12
   macro avg       0.76      0.75      0.75        12
weighted avg       0.76      0.75      0.75        12
```

## Architektura

1. **Główna funkcja uruchamiająca różne moduły**
2. **ERDS_readData.ipynb** – Zbieranie danych EEG
3. **ERDS_utils.py** – Moduł z rysowaniem obrazków i funkcjami pomocniczymi
4. **ERDS_CSP.py** – Moduł odpowiedzialny za obliczenie macierzy przejścia `W` oraz wartości własnych `Lambda` za pomocą danych treningowych, wybór dominujących wektorów własnych, Transformacje sygnału EEG treningowego i testowego macierzą `W`.
5. **ERDS_training.py** – Klasyfikacja ruchu i tylko jego zamiaru w oparciu o przekształcone cechy za pomocą CSP.

## Wymagania

- Python 3.8+
- numpy
- scipy
- matplotlib
- scikit-learn
- xgboost

## Instalacja

```bash
git clone https://github.com/romateusz/BCI_ERDS_simulation.git .
pip install -r requirements.txt

```

## Uruchomienie oprogramowania

### 1. Część wczytania i obróbki danych

Należy wykonać notebook w środowisku Jupyter: `ERDS_readData.py`, po wykonaniu notebooka zapisane zostaną dane:

- mati_ruch_dane.npy
- mati_wyobrazenie_dane.npy

### 2. Część Analizy i Treningu

```bash
python3 ERDS_main.py 
```

#### Wybór danych EEG

Wybierz rodziaj danych wpisując: R lub W (Ruch/Wyobrażenie):

```bash
 R
 ```

#### Wybór komponentów

Wybierz właściwy pierwszy komponent na podstawie rysunków (powinien zawierać aktywność na P3-P4):

```bash
 18 
 ```

Wybierz właściwy drugi komponent na podstawie rysunków:

```bash
0
```

#### Wynik

Na wyjściu zostają przedstawione raporty uzyskanych klasyfikacji badanego problemu rozrożnienia ruchu/wyobrażenia lewego od prawego palca wskazującego.