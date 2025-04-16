import numpy as np
import ERDS_utils
import ERDS_CSP
import ERDS_training

# wczytywanie danych
daneRuch = np.load("mati_ruch_dane.npy")
daneWyob = np.load("mati_wyobrazenie_dane.npy")
print(daneRuch.shape)
print(daneWyob.shape)
Fs = 256

daneRuchLewa = daneRuch[0]
daneRuchPrawa = daneRuch[1]
daneWyobLewa = daneWyob[0]
daneWyobPrawa = daneWyob[1]

usr_spektrogramRuchLewa, f, t = ERDS_utils.oblicz_usredniony_spektrogram(daneRuchLewa)
usr_spektrogramRuchPrawa, _, _ = ERDS_utils.oblicz_usredniony_spektrogram(daneRuchPrawa)
usr_spektrogramWyobLewa, _, _ = ERDS_utils.oblicz_usredniony_spektrogram(daneWyobLewa)
usr_spektrogramWyobPrawa, _, _ = ERDS_utils.oblicz_usredniony_spektrogram(daneWyobPrawa)


# ogląd (dla testu) uśrednionych spektrogramów
ERDS_utils.rysunek_glowy(usr_spektrogramRuchLewa, t, f, "Uśredniony Spektogram RuchLewa", sigma=1)
ERDS_utils.rysunek_glowy(usr_spektrogramRuchPrawa, t, f, "Uśredniony Spektogram RuchPrawa", sigma=1)
ERDS_utils.rysunek_glowy(usr_spektrogramWyobLewa, t, f, "Uśredniony Spektogram WyobLewa", sigma=1)
ERDS_utils.rysunek_glowy(usr_spektrogramWyobPrawa, t, f, "Uśredniony Spektogram WyobPrawa", sigma=1)


# obliczanie spektrogramów dla kolejnych realizacji
SRL, f, t = ERDS_utils.oblicz_spektrogram(daneRuchLewa)
SRP, f, t = ERDS_utils.oblicz_spektrogram(daneRuchPrawa)
SWL, f, t = ERDS_utils.oblicz_spektrogram(daneWyobLewa)
SWP, f, t = ERDS_utils.oblicz_spektrogram(daneWyobPrawa)


# spektrogramy normowane średnią (baseline)
SRL_norm = ERDS_utils.normowane_sr_baseline(SRL)
SRP_norm = ERDS_utils.normowane_sr_baseline(SRP)
SWL_norm = ERDS_utils.normowane_sr_baseline(SWL)
SWP_norm = ERDS_utils.normowane_sr_baseline(SWP)

usr_SRL_norm = np.mean(SRL_norm, axis=0)  # (19, Nf, Nt)
usr_SRP_norm = np.mean(SRP_norm, axis=0)  # (19, Nf, Nt)
usr_SWL_norm = np.mean(SWL_norm, axis=0)  # (19, Nf, Nt)
usr_SWP_norm = np.mean(SWP_norm, axis=0)  # (19, Nf, Nt)

ERDS_utils.rysunek_glowy(usr_SRL_norm, t, f, "Uśredniony Spektrogram RuchLewa normowany średnią", sigma=1)
ERDS_utils.rysunek_glowy(usr_SRP_norm, t, f, "Uśredniony Spektogram RuchPrawa normowany średnią", sigma=1)
ERDS_utils.rysunek_glowy(usr_SWL_norm, t, f, "Uśredniony Spektogram WyobLewa normowany średnią", sigma=1)
ERDS_utils.rysunek_glowy(usr_SWP_norm, t, f, "Uśredniony Spektogram WyobPrawa normowany średnią", sigma=1)

# # spektrogramy normowane Z-Score względem baselinu (obliczanie wariancji i dzielenie przez średnią)
# print("SRL")
# SRL_norm_sigm = ERDS_utils.normowane_sigm_baseline(SRL)
# print("SRP")
# SRP_norm_sigm = ERDS_utils.normowane_sigm_baseline(SRP)
# print("SWL")
# SWL_norm_sigm = ERDS_utils.normowane_sigm_baseline(SWL)
# print("SWP")
# SWP_norm_sigm = ERDS_utils.normowane_sigm_baseline(SWP)

# ERDS_utils.rysunek_glowy(SRL_norm_sigm, t, f,"Uśredniony Spektrogram RuchLewa normowany sigma", sigma=3)
# ERDS_utils.rysunek_glowy(SRP_norm_sigm, t, f, "Uśredniony Spektrogram RuchPraw normowany sigma", sigma=3)
# ERDS_utils.rysunek_glowy(SWL_norm_sigm, t, f, "Uśredniony Spektrogram WyobLewa normowany sigma", sigma=3)
# ERDS_utils.rysunek_glowy(SWP_norm_sigm, t, f, "Uśredniony Spektrogram WyobPrawa normowany sigma", sigma=3)

X_train, X_test, y_train, y_test = ERDS_training.podziel_dane_do_uczenia(daneRuchLewa, daneRuchPrawa)
X_train_lewa = X_train[y_train == 0]
X_train_prawa = X_train[y_train == 1]

# CSP
C = ERDS_CSP.CSP(256, 19)
C.fit(X_train_lewa, X_train_prawa)
C.componentsHeadImg()
S_train = C.transform(X_train)
S_test = C.transform(X_test)

print("Wybierz właściwy pierwszy komponent na podstawie rysunków (powinien zawierać aktywność na P3-P4): ", end="")
numKomponent1 = int(input())

SpektogramyRuchTrain1, f_tab, t_tab = ERDS_utils.przygotuj_spektrogramy(S_train, numKomponent1, fs=256, nperseg=128, noverlap=64)
SpektogramyRuchTest1, f_tab, t_tab = ERDS_utils.przygotuj_spektrogramy(S_test, numKomponent1, fs=256, nperseg=128, noverlap=64)
print(SpektogramyRuchTrain1.shape)
print(SpektogramyRuchTest1.shape)

print("Wybierz właściwy drugi komponent na podstawie rysunków: ", end="")
numKomponent2 = int(input())

SpektogramyRuchTrain2, f_tab, t_tab = ERDS_utils.przygotuj_spektrogramy(S_train, numKomponent2, fs=256, nperseg=128, noverlap=64)
SpektogramyRuchTest2, f_tab, t_tab = ERDS_utils.przygotuj_spektrogramy(S_test, numKomponent2, fs=256, nperseg=128, noverlap=64)
print(SpektogramyRuchTrain2.shape)
print(SpektogramyRuchTest2.shape)

# SpłaszczoneSpektogramy
SSRTrain1 = SpektogramyRuchTrain1.reshape(SpektogramyRuchTrain1.shape[0], -1)
SSRTest1 = SpektogramyRuchTest1.reshape(SpektogramyRuchTest1.shape[0], -1)
SSRTrain2 = SpektogramyRuchTrain1.reshape(SpektogramyRuchTrain2.shape[0], -1)
SSRTest2 = SpektogramyRuchTest1.reshape(SpektogramyRuchTest2.shape[0], -1)
print("Test shape: ", SSRTrain1.shape, SSRTrain2.shape)
print("Train shape: ", SSRTest1.shape, SSRTest2.shape)

SSRTrain = np.concatenate([SSRTrain1, SSRTrain2], axis=1)
SSRTest = np.concatenate([SSRTest1, SSRTest2], axis=1)
print(SSRTrain.shape)
print(SSRTest.shape)


print("\n===================\nRegresja Logistyczna\n===================")
modelRL = ERDS_training.treningRegresjaLogistyczna(SSRTrain, y_train)
ERDS_training.testRegresjaLogistyczna(modelRL, SSRTest, y_test)
print("\n===================\nXGBOOST\n====================")
modelXGBOOST = ERDS_training.treningXGBOOST(SSRTrain, y_train)
ERDS_training.testTESTXGBOOST(modelXGBOOST, SSRTest, y_test)