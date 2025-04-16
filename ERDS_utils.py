import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal

def oblicz_usredniony_spektrogram(dane, fs=256, nperseg=128, noverlap=64):
    """
    Oblicza spektrogram dla każdego kanału EEG i uśrednia po 30 realizacjach.
    Automatycznie oblicza liczbę segmentów czasowych.

    Parametry:
    - dane: ndarray o kształcie (30, 19, 2048) - 30 realizacji, 19 kanałów, 2048 próbek.
    - fs: częstotliwość próbkowania (domyślnie 256 Hz).
    - nperseg: długość segmentu FFT (domyślnie 256).
    - noverlap: nakładanie się okien (domyślnie 128).

    Zwraca:
    - usredniony_spektrogram: ndarray o kształcie (19, liczba częstotliwości, liczba czasowych segmentów)
    - f: wektor częstotliwości
    - t: wektor czasu
    """
    num_realizacje, num_kanaly, num_probki = dane.shape

    # Obliczenie dynamicznej liczby segmentów czasowych
    Nt = (num_probki - nperseg) // (nperseg - noverlap) + 1
    Nf = nperseg // 2 + 1  # Liczba częstotliwości po FFT

    # Inicjalizacja tablicy na spektrogramy
    spektrogramy = np.zeros((num_realizacje, num_kanaly, Nf, Nt))  

    for i in range(num_realizacje):  # Iteracja po 30 realizacjach
        for k in range(num_kanaly):  # Iteracja po 19 kanałach
            f, t, Sxx = signal.spectrogram(dane[i, k], fs=fs, nperseg=nperseg, noverlap=noverlap)
            spektrogramy[i, k] = Sxx  # Zapisywanie spektrogramu

    # Uśrednianie po realizacjach
    usredniony_spektrogram = np.mean(spektrogramy, axis=0)  # (19, Nf, Nt)
    print(usredniony_spektrogram.shape)

    return usredniony_spektrogram, f, t


def oblicz_spektrogram(dane, fs=256, nperseg=128, noverlap=64):
    """
    Oblicza spektrogram dla każdego kanału EEG i uśrednia po 30 realizacjach.
    Automatycznie oblicza liczbę segmentów czasowych.

    Parametry:
    - dane: ndarray o kształcie (30, 19, 2048) - 30 realizacji, 19 kanałów, 2048 próbek.
    - fs: częstotliwość próbkowania (domyślnie 256 Hz).
    - nperseg: długość segmentu FFT (domyślnie 128).
    - noverlap: nakładanie się okien (domyślnie 64).

    Zwraca:
    - spektrogram: ndarray o kształcie (30, 19, liczba częstotliwości, liczba czasowych segmentów)
    - f: wektor częstotliwości
    - t: wektor czasu
    """
    num_realizacje, num_kanaly, num_probki = dane.shape

    # Obliczenie dynamicznej liczby segmentów czasowych
    Nt = (num_probki - nperseg) // (nperseg - noverlap) + 1
    Nf = nperseg // 2 + 1  # Liczba częstotliwości po FFT

    # Inicjalizacja tablicy na spektrogramy
    S = np.zeros((num_realizacje, num_kanaly, Nf, Nt))  

    for i in range(num_realizacje):  # Iteracja po 30 realizacjach
        for k in range(num_kanaly):  # Iteracja po 19 kanałach
            f, t, Sxx = signal.spectrogram(dane[i, k], fs=fs, nperseg=nperseg, noverlap=noverlap)
            S[i, k] = Sxx  # Zapisywanie spektrogramu

    # Uśrednianie po realizacjach
    print(S.shape)

    return S, f, t


def rysunek_glowy(spektrogram_tab, t, f, title, sigma=3):
    # Współrzędne subplotów
    wsp = [("Fp1", 2), ("Fp2", 4), ("F7", 6), ("F3", 7), ("Fz", 8), 
           ("F4", 9), ("F8", 10), ("T3", 11), ("C3", 12), ("Cz", 13), 
           ("C4", 14), ("T4", 15), ("T5", 16), ("P3", 17), ("Pz", 18), 
           ("P4", 19), ("T6", 20), ("O1", 22), ("O2", 24)]
    channels_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 
                      'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 
                      'O1', 'O2']
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=15)
    i = 0

    t_lim = (0, len(t))
    f_lim = (0, np.searchsorted(f, 30))  # Tylko do 30 Hz

    for nazwa, j in wsp:
        plt.subplot(5, 5, j)
        plt.imshow(spektrogram_tab[i, f_lim[0]:f_lim[1], t_lim[0]:t_lim[1]], 
                   aspect='auto', origin='lower', cmap='seismic', 
                   extent=[t[t_lim[0]], t[t_lim[1]-1], f[f_lim[0]], f[f_lim[1]-1]],
                   vmin=-sigma, vmax=sigma)
        plt.colorbar()
        plt.title(channels_names[i])
        i += 1
    
    plt.tight_layout()

    # Tworzenie katalogu jeśli nie istnieje
    output_dir = "ERDS_output"
    os.makedirs(output_dir, exist_ok=True)

    # Zapis wykresu do pliku w tym katalogu
    fig_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(fig_path)
    print(f"Wykres zapisany jako: {fig_path}")


def normowane_sr_baseline(Spektogram):
    for proba in range(Spektogram.shape[0]):
        for ch in range(Spektogram.shape[1]):
            for f in range (Spektogram.shape[2]):
                S_baseline = np.mean(Spektogram[proba, ch, f, :7]) # dla tych ustawień 7
                for t in range (Spektogram.shape[3]):
                    Spektogram[proba, ch, f, t] = (Spektogram[proba, ch, f, t] - S_baseline) / S_baseline
    return Spektogram


def normowane_sigm_baseline(Spektogram):
    tab = np.zeros((Spektogram.shape[1], Spektogram.shape[2], Spektogram.shape[3]))
    for ch in range(Spektogram.shape[1]):
        for proba in range(Spektogram.shape[0]):
            for f in range (Spektogram.shape[2]):
                baseline = Spektogram[:, ch, f, :7] # dla tych ustawień 7
                for t in range (Spektogram.shape[3]):
                    tab[ch, f, t] = (np.mean(Spektogram[:, ch, f, t]) - np.mean(baseline)) / np.std(baseline)
    return tab


def przygotuj_spektrogramy(daneTestowe, numKomponent, fs=256, nperseg=128, noverlap=64):
    """
    zwraca zestaw spektogramów

    Parametry:
    - dane: ndarray o kształcie (30, 19, 2048) - 30 realizacji, 19 kanałów, 2048 próbek.
    - fs: częstotliwość próbkowania (domyślnie 256 Hz).
    - nperseg: długość segmentu FFT (domyślnie 256).
    - noverlap: nakładanie się okien (domyślnie 128).

    Zwraca:
    - ndarray o kształcie (30, 19, liczba częstotliwości, liczba czasowych segmentów)
    - f: wektor częstotliwości
    - t: wektor czasu
    """
    num_realizacje, num_kanaly, num_probki = daneTestowe.shape

    # Obliczenie dynamicznej liczby segmentów czasowych
    Nt = (num_probki - nperseg) // (nperseg - noverlap) + 1
    Nf = nperseg // 2 + 1  # Liczba częstotliwości po FFT

    # Inicjalizacja tablicy na spektrogramy
    spektrogramy = np.zeros((num_realizacje, Nf, Nt))  
        
    # zapisywanie spektogramów, dla komponentu o maksymalnej wartości własnej
    for i in range(num_realizacje):  # Iteracja po 30 realizacjach
        f, t, Sxx = signal.spectrogram(daneTestowe[i, numKomponent], fs=fs, nperseg=nperseg, noverlap=noverlap)
        spektrogramy[i] = Sxx  # Zapisywanie spektrogramu

    f_lim = (0, np.searchsorted(f, 30))  # Tylko do 30 Hz

    return spektrogramy[:,:f_lim[1],:], f[:f_lim[1]], t