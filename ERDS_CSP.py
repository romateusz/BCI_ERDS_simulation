import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

class CSP:
    def __init__(self, Fs, N_channels):
        self.Fs = Fs
        self.N_chan = N_channels
    
    def fit(self, daneLewa, danePrawa):

        N_reps = daneLewa.shape[0]

        R_L = np.zeros((self.N_chan, self.N_chan))
        R_P = np.zeros((self.N_chan, self.N_chan))
        
        start = 2 * self.Fs
        end = 8 * self.Fs
        for r in range(N_reps):
            L = daneLewa[r,:,start:end]
            # tmp = np.cov(L)
            tmp = LedoitWolf().fit(L.T).covariance_
            R_L = R_L + tmp
            
            P = danePrawa[r,:,start:end]
            # tmp = np.cov(P)
            tmp = LedoitWolf().fit(P.T).covariance_
            R_P = R_P + tmp
        
        R_L = R_L / N_reps
        R_P = R_P / N_reps
        
        Lambda, W = eigh(R_L, R_P)

        # tworzy atrybuty klasy: Lambda i W
        self.Lambda = Lambda
        self.W = W

    def componentsHeadImg(self):
        # Funkcja pomocnicza rysująca rysunek wartości komponentów
        def rysunek_komponentu_na_glowie(tab, num):
            wsp = [("Fp1", 2), ("Fp2", 4), ("F7", 6), ("F3", 7), ("Fz", 8), 
                ("F4", 9), ("F8", 10), ("T3", 11), ("C3", 12), ("Cz", 13), 
                ("C4", 14), ("T4", 15), ("T5", 16), ("P3", 17), ("Pz", 18), 
                ("P4", 19), ("T6", 20), ("O1", 22), ("O2", 24)]
            
            v = np.max(np.abs(tab))

            fig, ax = plt.subplots(5, 5, figsize=(10, 10))
            fig.suptitle(f"Komponent: {numKomponent}, Lambda: {self.Lambda[numKomponent]}", fontsize=15)
            ax = ax.flatten()  # Ułatwia indeksowanie 1D
            
            # Na początku wyłączamy wszystkie osie
            for a in ax:
                a.axis('off')
            
            # Teraz wypełniamy tylko te, które mają dane
            for i, (nazwa, j) in enumerate(wsp):
                idx = j - 1
                T = tab[i] * np.ones((10, 10))
                im = ax[idx].imshow(-T, cmap='seismic', vmin=-v, vmax=v)
                fig.colorbar(im, ax=ax[idx])
                ax[idx].set_title(nazwa)
                ax[idx].axis('on')
            
            plt.tight_layout()
            # Tworzenie katalogu jeśli nie istnieje
            output_dir = "ERDS_output"
            os.makedirs(output_dir, exist_ok=True)

            # Zapis wykresu do pliku w tym katalogu
            fig_path = os.path.join(output_dir, f"Rysunek_komponentu_{num}.png")
            plt.savefig(fig_path)
            print(f"Wykres zapisany jako: {fig_path}")

        # topologia
        W_odwrotna = np.linalg.inv(self.W)
        for numKomponent in [0,1,17,18]:
            print(f"\n===================\nKomponent: {numKomponent}\n===================")
            wybranyKomponent = W_odwrotna[numKomponent,:]
            print(wybranyKomponent.shape)
            print(wybranyKomponent)
            rysunek_komponentu_na_glowie(wybranyKomponent, numKomponent)


    def transform(self, dane):
        # dostaje dane zwraca komponenty
        S = np.zeros_like(dane)
        for r in range(dane.shape[0]):
            S[r,:,:] = self.W.T @ dane[r]
        return S