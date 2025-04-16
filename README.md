# BCI_ERDS_simulation

## Autor

👤 **Mateusz Roman**

## Spis treści

- [BCI\_ERDS\_simulation](#bci_erds_simulation)
  - [Autor](#autor)
  - [Spis treści](#spis-treści)
  - [Opis](#opis)
    - [Streszczenie](#streszczenie)
    - ["Krok po kroku", czyli co tak naprawdę zostało wykonane](#krok-po-kroku-czyli-co-tak-naprawdę-zostało-wykonane)
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

### Streszczenie

Celem projektu jest rozróżnienie faktycznego ruchu palca wskazującego:

- lewego,
- prawego,
  
jak i na podstawie ruchu wyobrażonego na korzystając z zebranego sygnału EEG.

### "Krok po kroku", czyli co tak naprawdę zostało wykonane

1. -> Etap zbierania danych <-

2. Oprogramowanie to wczytuje, przetwarza i zapisuje do plików pocięte dane EEG.

3. Następnie umożliwia ich wstępną analizę, która opiera się na spektrogramach stworzonych z wczytanego sygnału EEG. Umożliwia to jakościową ocenę wczytanych sygnałów.

4. Wszelkie rysunki podczas działania programu są zapisywane w katalogu `./ERDS_output`. Między innymi trafiają tam obrazki przedstawiające spektrogramy, rozmieszczone w układzie odpowiadającym lokalizacji elektrod. Spektogramy można doskalować według potrzeby parametrem sigma, który dostosowuje zakres dolny i górny kolorów dla wartości przedstawionych na ilustracjach. Podczas działania programu są generowane trzy typy spektrogramów:

   - Spektrogramy przedstawiające w swoich pikselach, po prostu amplitudę sygnału w dziedzinie czas-częstość,
   - Spektrogramy (jak poprzednio), ale normalizowane średnią z referencyjnego fragmentu sygnału (baselinem),
   - Spektrogramy (jak poprzednio), ale normalizowane Z-score normalizacja względem baseline’u (średnia i std liczone po próbach).

Na usyzkanych spektrogramach można zauważyć:

  - najbardziej interesujące zmiany będą obserwowane nad korą ruchową, czyli elektrody w linii C, nastomiast z racji na przesunięcie czepka, ewidentnie widać, że największe różnice zaobserwowane zostały w linii P,
  
  - W szczególności możnaby się spodziewać, że najbardziej w różnicowania ruchu lewego palca od prawego, posłużą w tym przypadku elektordy P3 i P4. Natomiast w późniejszej fazie projektu okazało się, że w przypadku zebranych (przy tym projekcie) danych największe różnice są obserwowane przez CSP na elektrodzie P4,
  
  - Zaobserwowano znaczące zmiany amplitudy w paśmie alfa (~10 Hz), szczególnie tuż po pojawieniu się bodźca świadczącego o kierunku, co może wskazywać na zaangażowanie uwagi oraz przetwarzanie informacji sensorycznej, co zmusiło badanego do podjęcia reakcji. Bezpośrednio po tej fazie aktywnego przetwarzania zauważalny jest ponowny wzrost amplitudy w zakresie alfa – tzw. alpha rebound, który może odzwierciedlać zakończenie aktywnego przetwarzania oraz powrót do stanu spoczynkowego lub mechanizmów hamowania korowego.

  - Dodatkowo, obserwowane są zmiany amplitudy również w paśmie beta, co może wiązać się z procesami motorycznymi lub poznawczymi towarzyszącymi reakcjom na bodziec.

5. Następnie przetworzone dane EEG (w dziedzinie czasu), dzieli się na dwa zbiory: treningowy i testowy (w celu zasymulowania prawdziwego BCI, gdzie część danych posłużyłaby do kalibracji oprogramowania, a cześć zostałaby użyta już po kalibracji, symulując niejako dane wczytywane na żywo w Interfejsie Mózg-Komputer). Treningową część danych poddaje się metodzie Common Spatial Patterns (CSP), w celu wyodrębnienia cech z kory ruchowej, umożliwiającej znalezienie przestrzennych wzorców związanych z aktywacją kory ruchowej. Na podstawie wektora własnego macierzy `W` i wartości własnych `Lambda` obliczonych za pomocą CSP, wybierana jest interaktywnie przez użytkownika, reprezentacja dwóch najlepiej różnicujących dwa zbiory danych komponentów.

6. Dane trenigowe jak i testowe tranformowano za pomocą uzyskanej macierzy W (tylko na danych treningowych, w celu uniknięcia pewnych przecieków informacji). Następnie dane, będące niejako wytłumaczeniem źródeł, a nie sygnałami zarejestrowanymi z elektrod, przetworzono na spektogramy i wytrenowano dwa modele: Regresji Logistycznej i XGBOOST.

7. Uzyskane modele poddano weryfikacji  jakości uzyskanego modelu za pomocą danych testowych. Dla ruchu uzyskane metryki określające jakość wytrenowanego modelu oscylowały w okolicach 75%, natomiast w przypadku wyobrażeń  wynik bardzo zależał od wylosowanego zbioru treningowego i testowego. Aby poprawność wskazań modelu wzrosła, należałoby zwiększyć ilość danych treningowych/kalibracyjnych. Należałoby również pomyśleć nad zwiększeniem czasu, gdzie badany trenowałby oraz być może wybrać silniejszy/łatwiejszy do wyobrażenia dla badanego ruch. Nie mniej jednak palce to dobry pomysł, gdyż duże obszary kory odpowiadają za ruchy palców i dłoni (jest to ściśle związane z ich niezwykłą precyzją i czułością).

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