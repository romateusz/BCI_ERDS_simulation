# BCI_ERDS_simulation

## Spis treści

- [BCI\_ERDS\_simulation](#bci_erds_simulation)
  - [Spis treści](#spis-treści)
  - [Opis](#opis)
  - [Architektura](#architektura)
  - [Wymagania](#wymagania)
  - [Instalacja](#instalacja)
  - [Uruchomienie](#uruchomienie)
    - [wybór kanałów np.:](#wybór-kanałów-np)

## Opis

System przetwarza dane EEG w celu klasyfikacji prawdziwego ruchi i zamiaru poruszenia palca wskazującego:
- lewego,
- prawego.

Projekt ten służy do wykrywania zamiaru poruszenia lewego palca lub prawego palca wskazującego na podstawie sygnałów EEG. Analiza opiera się na przetworzeniu sygnału na spektrogramy, w celu jakościowej oceny sygnałów, a następnie wykorzystano metodę Common Spatial Patterns (CSP) w celu wyodrębnienia cech z kory ruchowej.  umożliwiającej znalezienie przestrzennych wzorców związanych z aktywacją kory ruchowej. Na podstawie wektora własnego macierzy W i wartości własnych Lambda oblicoznych za pomocą CSP, wybierana jest reprezentacja najlepiej różnicująca klasy ruchu.

Dane zostały podzielone na zbiór treningowy i testowy (w celu zasymulowania prawdziwego BCI, gdzie częśc danych byłaby kalibracyjna, a cześć zostałaby użyta już po kalibracji). W celu wykonania klasyfikacji użyto Regresji Logistycznej i metody XGBOOST.
Dla ruchu uzyskane metryki określające jakość wytrenowanego modelu oscylowały w okolicach 75%, natomiast w przypadku wyobrażeń była duża losowość i wynik bardzo zależał od wylosowanego zbioru treningowego i testowego, ale poprawność modelu dla problemu rozróżnienia wyobrażonego ruchu można ocenić na 66%.

*Przykład dla ruchu:*
Confusion Matrix:
 [[5 1]
 [2 4]]
Classification Report:
               precision    recall  f1-score   support

           0       0.71      0.83      0.77         6
           1       0.80      0.67      0.73         6

    accuracy                           0.75        12
   macro avg       0.76      0.75      0.75        12
weighted avg       0.76      0.75      0.75        12

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

## Uruchomienie 
```bash
python3 ERDS_main.py 
```

### wybór kanałów np.:

Wybierz właściwy pierwszy komponent na podstawie rysunków (powinien zawierać aktywność na P3-P4):
```bash
 18 
 ```
 
Wybierz właściwy drugi komponent na podstawie rysunków:
```bash
0
```